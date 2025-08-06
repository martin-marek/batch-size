import jax
import jax.numpy as jnp
from flax import nnx
from collections.abc import Mapping


def flatten_dict(d, prefix=None, sep='.'):
    if isinstance(d, Mapping):
        out = {}
        for k, v in d.items():
            nested_prefix = k if prefix is None else f'{prefix}{sep}{k}'
            out |= flatten_dict(v, nested_prefix, sep)
        return out
    else:
        return {prefix: d}


def get_num_model_params(model: nnx.Module):
    graphdef, params = nnx.split(model, nnx.Param)
    n_params = jax.tree.reduce(lambda x, y: x + jnp.size(y), params, 0)
    return n_params


def halflife_to_decay(t_token, n_batch=1):
    """
    notation:
    - t_token: halflife measured in number of tokens
    - t_steps: halflife measured in number of steps
    - n_batch: number of tokens per batch
    - d: decay coefficient
    """
    t_steps = t_token / n_batch # halflife (measured in number of steps)
    d = (1/2)**(1/t_steps)
    return d


def decay_to_halflife(d, n_batch=1):
    """
    notation:
    - t_token: halflife measured in number of tokens
    - t_steps: halflife measured in number of steps
    - n_batch: number of tokens per batch
    - d: decay coefficient
    """
    # note: d**t_steps = 1/2
    t_steps = jnp.log(1/2) / jnp.log(d)
    t_token = t_steps * n_batch
    return t_token


@jax.jit
def to_bf16_stochastic(key, source):
    """
    performs (float32 -> bfloat16) stochastic rounding 
    based on https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    """
    # ensure the source array is float32, the bitwise logic depends on it
    source = source.astype(jnp.float32)

    # reinterpert float32 source as uint32 to allow bitwise operations
    source_uint32 = jax.lax.bitcast_convert_type(source, jnp.uint32)

    # randomly flip lower 16 bits of the float32 source
    # these are the bits that get truncated when converting to bf16
    random_int = jax.random.randint(
        key,
        shape=source.shape,
        minval=0,
        maxval=(1 << 16),
        dtype=jnp.uint32
    )
    result_uint32 = source_uint32 + random_int

    # mask off lower 16 bits, keep top 16 bits (corresponding to bf16 format)
    mask = jnp.uint32(0xFFFF0000)
    result_uint32 = jax.lax.bitwise_and(result_uint32, mask)

    # cast result to bf16
    result_fp32 = jax.lax.bitcast_convert_type(result_uint32, jnp.float32)
    result_bf16 = result_fp32.astype(jnp.bfloat16)

    return result_bf16
