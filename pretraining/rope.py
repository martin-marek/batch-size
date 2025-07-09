"""https://github.com/google/flax/blob/main/examples/gemma/positional_embeddings.py"""

import jax
import jax.numpy as jnp


def apply_rope(
    inputs: jax.Array, # [B, T, N, H]
    positions: jax.Array, # [B, T]
    max_wavelength: int = 10_000,
    scale_factor: float = 1.0,
) -> jax.Array:
    """Applies RoPE."""
    B, T, N, H = inputs.shape
    if scale_factor < 1.0:
        raise ValueError(f'scale_factor must be >= 1.0, got {scale_factor}')

    fraction = 2 * jnp.arange(0, H // 2) / H # [H/2]
    timescale = max_wavelength**fraction # [H/2]

    sinusoid_inp = (positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]) # [B, T, H/2]
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :] # [B, T, 1, H/2]
    sinusoid_inp /= scale_factor # [B, T, 1, H/2]

    sin = jnp.sin(sinusoid_inp) # [B, T, 1, H/2]
    cos = jnp.cos(sinusoid_inp) # [B, T, 1, H/2]

    first_half, second_half = jnp.split(inputs, 2, axis=-1) # [B, T, N, H/2]
    first_part = first_half * cos - second_half * sin # [B, T, N, H/2]
    second_part = second_half * cos + first_half * sin # [B, T, N, H/2]
    out = jnp.concatenate([first_part, second_part], axis=-1) # [B, T, N, H]
    return out.astype(inputs.dtype) # [B, T, N, H]
