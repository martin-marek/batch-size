import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu
from flax import nnx
from flax.nnx import filterlib
from flax.nnx.training.optimizer import OptState, _wrap_optimizer_state
import factorized, utils
from typing import Optional


class Optimizer(nnx.Optimizer):
    """Extends nnx.Optimizer with stochastic rounding."""
    def __init__(
        self,
        model,
        tx: optax.GradientTransformation,
        wrt: filterlib.Filter = nnx.Param,
        stochastic_round = False,
    ):
        self.step = nnx.training.optimizer.OptState(jnp.array(0, dtype=jnp.uint32))
        self.model = model
        self.tx = tx
        self.opt_state = nnx.training.optimizer._wrap_optimizer_state(tx.init(nnx.state(model, wrt)))
        self.wrt = wrt
        self.stochastic_round = stochastic_round

    def update(self, key, grads, **kwargs):
        params = nnx.state(self.model, self.wrt)
        opt_state = nnx.training.optimizer._opt_state_variables_to_state(self.opt_state)

        updates, new_opt_state = self.tx.update(grads, opt_state, params, **kwargs)
        new_params = apply_updates(key, params, updates, self.stochastic_round)
        assert isinstance(new_params, nnx.State)

        self.step.value += 1
        nnx.update(self.model, new_params)
        nnx.training.optimizer._update_opt_state(self.opt_state, new_opt_state)


def apply_updates(
    key: jax.Array,
    params: optax.Params,
    updates: optax.Updates,
    stochastic_round = False
) -> optax.Params:
    """Extends optax.apply_updates with stochastic rounding."""
    keys = otu.tree_split_key_like(key, params)
    def leaf_update(p, u, key):
        if p is None: return None
        param_dtype = jnp.asarray(p).dtype
        if stochastic_round:
            p = p.astype(jnp.float32) + u
            p = utils.to_bf16_stochastic(key, p)
        else:
            p += u
        return p.astype(param_dtype)
    return jax.tree.map(leaf_update, params, updates, keys, is_leaf=lambda x: x is None)


def adafactor(
    learning_rate: optax.ScalarOrSchedule,
    decay_rate: float = 0.8,
    clipping_threshold: Optional[float] = 1.0,
    min_dim_size_to_factor: int = 128,
) -> optax.GradientTransformation:
    """
    Adafactor reimplemented to use float32 state, regardless of param dtype.
    https://github.com/google-deepmind/optax/blob/8973bb3c77b07850737246815f1c028b53fffbe0/optax/_src/alias.py#L225#L327
    """
    return optax.chain(
        factorized.scale_by_factored_rms(decay_rate=decay_rate, min_dim_size_to_factor=min_dim_size_to_factor),
        optax.clip_by_block_rms(clipping_threshold) if clipping_threshold is not None else optax.identity(),
        optax.scale_by_learning_rate(learning_rate),
        optax.scale_by_param_block_rms(),
    )
