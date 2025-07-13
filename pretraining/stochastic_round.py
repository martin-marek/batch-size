import jax
import jax.numpy as jnp


def to_bf16(key, x):
    """Stochastically round input to bf16."""

    # round x (assumed to be in fp32) to two closest bf16 values
    # one of these values will be smaller than x, the other larger
    # one of these values will be closer to x (default rounding), the other will be farther away
    x_closer = x.astype(jnp.bfloat16)
    error = x_closer.astype(jnp.float32) - x
    x_farther = jnp.nextafter(x_closer, jnp.where(error>0, -jnp.inf, jnp.inf))

    # round x to either the closer or farther value s.t. we get the true value in expectation
    ulp = jnp.abs(x_farther.astype(jnp.float32) - x_closer.astype(jnp.float32))
    rand_unif = jax.random.uniform(key, x.shape)
    use_farther = rand_unif * ulp < jnp.abs(error)
    x_stoch = jnp.where(use_farther, x_farther, x_closer)
    
    return x_stoch
