import jax
import jax.numpy as jnp
import flax
from flax import nnx
from tqdm.auto import tqdm
from functools import partial


@flax.struct.dataclass
class SamplingState:
    key: jax.Array
    step: jnp.int32
    tokens: jnp.ndarray # [B, T]
    kv_cache: dict
    done: jnp.ndarray # [B]


def _sample_top_p(key, probs, p=0.95):
    """Sample a token using top-p sampling.
    https://github.com/google/flax/blob/cca78723892c539b42c261d2625168d39b61c495/examples/gemma/sampler.py#L38"""
    probs_sorted, indices = jax.lax.top_k(probs, k=probs.shape[-1])
    cumsum_probs = jnp.cumsum(probs_sorted, axis=-1)
    mask = cumsum_probs - probs_sorted > p
    probs_sorted = jnp.where(mask, 0.0, probs_sorted)
    probs_sorted /= jnp.sum(probs_sorted, axis=-1, keepdims=True)
    next_token = jax.random.categorical(key, logits=jnp.log(probs_sorted))
    next_token = jnp.take_along_axis(indices, next_token[..., None], axis=-1)
    next_token = jnp.squeeze(next_token, axis=-1)
    return next_token


def _sample_step(state, model_graphdef, model_state, pbar, pad_id, eos_id, temperature=1):

    # we pass model_state as non-static arg, to avoid compiling it
    model = nnx.merge(model_graphdef, model_state)

    # sample next token
    key, key_sampling = jax.random.split(state.key)
    input_token = state.tokens[:, state.step, None] # [B, 1]
    logits, kv_cache = model(input_token, state.kv_cache) # [B, 1, V]
    if temperature == 0:
        sampled_token = logits[:, 0, :].argmax(1) # [B]
    else:
        probs = jax.nn.softmax(logits[:, 0, :] / temperature, axis=-1) # [B, V]
        sampled_token = _sample_top_p(key_sampling, probs)

    # update buffer
    next_token = state.tokens[:, state.step+1]
    update_token = jnp.where((~state.done) & (next_token==pad_id), sampled_token, next_token)
    tokens = state.tokens.at[:, state.step+1].set(update_token)

    # check if sampling is done
    done = state.done | ((next_token==pad_id) & (sampled_token==eos_id))
    jax.debug.callback(lambda: pbar.update(1) if jax.process_index() == 0 else None)
    
    return SamplingState(key, state.step+1, tokens, kv_cache, done)


def sample(key, model, tokens, temperature=1, pad_id=0, eos_id=1):
    B, T = tokens.shape

    # initialize state
    state = SamplingState(
        key=key,
        step=0,
        tokens=tokens,
        kv_cache=model.init_kv_cache(B, T),
        done=jnp.zeros([B], dtype=jnp.bool_),
    )

    # sample next token inside a while loop
    pbar = tqdm(total=T, desc='Sampling') if (jax.process_index() == 0) else None
    step_fn = lambda state: _sample_step(state, *nnx.split(model), pbar, pad_id, eos_id, temperature)
    cond_fn = lambda state: (state.step < T) & jnp.any(~state.done)
    state = jax.lax.while_loop(cond_fn, step_fn, state)
    jax.effects_barrier()
    if jax.process_index() == 0: pbar.close()

    # extract output sequences
    outputs = []
    for in_seq, out_seq in zip(jax.device_get(tokens), jax.device_get(state.tokens)):
        out_seq = out_seq[jnp.argmax(in_seq==pad_id):]
        if jnp.any(out_seq==pad_id): out_seq = out_seq[:jnp.argmax(out_seq==pad_id)]
        outputs += [out_seq.tolist()]

    return outputs
