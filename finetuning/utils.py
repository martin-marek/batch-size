import jax
import jax.numpy as jnp


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
