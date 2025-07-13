import math
import jax
import jax.numpy as jnp
import optax
import wandb
from functools import partial
from flax import nnx
from optax import tree_utils as otu
from tqdm.auto import tqdm
from omegaconf.dictconfig import DictConfig
import data, utils
import model as model_lib
import optimizer as optimizer_lib
import stochastic_round


@partial(jax.jit, static_argnames=('model_graphdef', 'pad'))
def loss_fn(model_state, model_graphdef, x, pad=False): # [B, T]
    model = nnx.merge(model_graphdef, model_state)
    y = jnp.roll(x, -1, axis=1)
    loss_mask = data.pad_mask(x) if pad else jnp.ones(x.shape, dtype=bool)
    loss_mask = loss_mask.at[:, -1].set(False)
    logits = model(x) # [B, T, V]
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, y) # [B, T]
    return (losses * loss_mask).sum() / loss_mask.sum()


@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef', 'simulate_bf16'))
def train_step(key, opt_state, opt_graphdef, model_graphdef, batch, simulate_bf16=False):
    # traing step in fp32
    loss, grads = jax.value_and_grad(loss_fn)(opt_state.model, model_graphdef, batch)
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grads)
    opt_state = nnx.state(optimizer)

    # optionally simulate bf16 weights
    # during fwd and bwd pass, we keep all weights in fp32 to force jax to compute fp32 activations and grads
    # after every optimzier step we round model and optimizer state to bf16 to simulate bf16 weights
    if simulate_bf16:
        key_tree = otu.tree_split_key_like(key, opt_state)
        round_leaf = lambda key, x: stochastic_round.to_bf16(key, x).astype(jnp.float32)
        opt_state = jax.tree.map(round_leaf, key_tree, opt_state)
    
    return opt_state, loss


@partial(jax.jit, static_argnames=('opt_graphdef', 'model_graphdef'))
def train_step_grad_acc(opt_state, opt_graphdef, model_graphdef, batches):
    n_batch = len(batches)
    loss_mean = 0
    grad_mean = otu.tree_zeros_like(opt_state.model)
    def step_fn(i , args):
        grad_mean, loss_mean = args
        batch_loss, batch_grads = jax.value_and_grad(loss_fn)(opt_state.model, model_graphdef, batches[i])
        grad_mean = jax.tree.map(lambda m, g: (i*m + g) / (i+1), grad_mean, batch_grads)
        loss_mean = (i*loss_mean + batch_loss) / (i+1)
        return grad_mean, loss_mean
    grad_mean, loss_mean = jax.lax.fori_loop(0, n_batch, step_fn, (grad_mean, loss_mean))
    optimizer = nnx.merge(opt_graphdef, opt_state)
    optimizer.update(grad_mean)
    opt_state = nnx.state(optimizer)
    return opt_state, loss_mean


@partial(jax.jit, static_argnames=('model_graphdef', 'pad'))
def eval_step(model_state, model_graphdef, dataset, pad=False):
    losses = jax.lax.map(partial(loss_fn, model_state, model_graphdef, pad=pad), dataset)
    return losses.mean()


def train_and_evaluate(c: DictConfig):

    # get model and dataset rng seed
    key = jax.random.key(c.seed)
    key, key_model, key_dataset = jax.random.split(key, 3)

    # sharding
    # all devices are aligned across a single mesh axis called 'data'
    # we use FSDP to shard data, model, and optimzier parameters across this axis
    num_fsdp_devices = jax.device_count() // c.num_tp_devices
    mesh = jax.make_mesh((num_fsdp_devices, c.num_tp_devices), ('data', 'model'))
    print('sharding mesh:', ', '.join(f'{k}={v}' for k, v in mesh.shape.items()))

    # model
    c.model.V = int(math.ceil(c.model.V / jax.device_count()) * jax.device_count()) # round V up to enable sharding
    model = model_lib.create_sharded_model(c.model, mesh, key_model)
    model_graphdef, model_state = nnx.split(model)

    # get num. model parameters
    n_params = {
        'n_param_nonembed': 12 * c.model.L * c.model.D**2,
        'n_param_embed': c.model.D * c.model.V,
        'n_param_actual': utils.get_num_model_params(model),
    }
    for k, v in n_params.items():
        print(f'{k}={v:_}')

    # dataset
    if (c.num_tokens_train is None) and (c.tokens_params_ratio is not None):
        c.num_tokens_train = c.tokens_params_ratio * (n_params['n_param_nonembed'] + n_params['n_param_embed'])
    ds_train, ds_valid = data.load_ds(key_dataset, mesh, c.ds_path, c.model.T, c.opt.microbatch_size, c.num_tokens_valid, c.num_tokens_train)
    if (c.num_tokens_train is None): c.num_tokens_train = ds_train.size

    # optimizer
    num_opt_steps = len(ds_train) // c.opt.grad_acc_steps
    tokens_per_opt_step = c.opt.batch_size * c.model.T
    tx = optimizer_lib.get_optimizer(c.opt, model_state, num_opt_steps, tokens_per_opt_step)
    optimizer = nnx.Optimizer(model, tx)
    opt_graphdef, opt_state = nnx.split(optimizer)

    # start wandb
    if jax.process_index() == 0:
        wandb.init(project=c.wandb_project, config=utils.flatten_dict(c), mode=c.wandb_mode, name=c.run_name)
        wandb.summary.update(n_params)

    # training loop
    train_loss_sum, train_loss_num = jnp.zeros([]), 0
    with mesh:
        pbar = range(num_opt_steps)
        if jax.process_index() == 0: pbar = tqdm(pbar)
        for step in pbar:
            
            # training step (no accumulation)
            if c.opt.grad_acc_steps == 1:
                batch = ds_train[step]
                key, key_round = jax.random.split(key)
                opt_state, batch_loss = train_step(key_round, opt_state, opt_graphdef, model_graphdef, batch, c.opt.simulate_bf16)

            # train step (gradient accumulation)
            if c.opt.grad_acc_steps > 1:
                batches = ds_train[step*c.opt.grad_acc_steps:(step+1)*c.opt.grad_acc_steps] # [grad_acc, micro_batch, T]
                opt_state, batch_loss = train_step_grad_acc(opt_state, opt_graphdef, model_graphdef, batches)
            
            # logging
            train_loss_sum += batch_loss
            train_loss_num += 1
            if train_loss_num * tokens_per_opt_step >= c.log_every_tokens:
                metrics = {}
                metrics['train_loss'] = train_loss_sum / train_loss_num
                metrics['train_tokens_seen'] = (step+1) * tokens_per_opt_step
                if jax.process_index() == 0:
                    wandb.log(metrics, step)
                    pbar.set_postfix_str(f'loss={metrics["train_loss"]:.2f}')
                train_loss_sum, train_loss_num = jnp.zeros([]), 0

        # eval at end of training
        eval_loss = eval_step(opt_state.model, model_graphdef, ds_valid, c.pad_eval)
        if jax.process_index() == 0:
            wandb.log({'eval_loss': eval_loss}, step)
            wandb.finish()
