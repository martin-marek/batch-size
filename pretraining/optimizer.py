import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu
from omegaconf import DictConfig
import utils
from typing import Optional, NamedTuple


def get_optimizer(c: DictConfig, params, num_opt_steps: int, tokens_per_opt_step: int):
    
    # get LR
    assert (c.peak_lr is not None) ^ ((c.peak_lr_scaled is not None) & (c.peak_lr_scaling is not None))
    if c.peak_lr is None:
        c.peak_lr = c.peak_lr_scaling * c.peak_lr_scaled

    # get schedule
    warmup_steps = int(c.warmup_frac * num_opt_steps)
    lr_schedule = optax.schedules.warmup_cosine_decay_schedule(0, c.peak_lr, warmup_steps, num_opt_steps)

    # convert (t1 <-> b1), (t2 <-> b2)
    assert (c.b1 is None) | (c.t1 is None) # at most one can be specified in config
    assert (c.b2 is None) | (c.t2 is None) # at most one can be specified in config
    assert (c.muon_b1 is None) | (c.muon_t1 is None) # at most one can be specified in config
    if c.b1 is None and c.t1 is not None: c.b1 = float(utils.halflife_to_decay(c.t1, tokens_per_opt_step))
    if c.b2 is None and c.t2 is not None: c.b2 = float(utils.halflife_to_decay(c.t2, tokens_per_opt_step))
    if c.t1 is None and c.b1 is not None: c.t1 = float(utils.decay_to_halflife(c.b1, tokens_per_opt_step))
    if c.t2 is None and c.b2 is not None: c.t2 = float(utils.decay_to_halflife(c.b2, tokens_per_opt_step))
    if c.muon_b1 is None and c.muon_t1 is not None: c.muon_b1 = float(utils.halflife_to_decay(c.muon_t1, tokens_per_opt_step))
    if c.muon_t1 is None and c.muon_b1 is not None: c.muon_t1 = float(utils.decay_to_halflife(c.muon_b1, tokens_per_opt_step))
    if c.b2_min is not None: c.b2 = max(c.b2, c.b2_min)

    if c.optimizer in ('sgd', 'signum'):
        assert c.b2 is None
        assert c.t2 is None
        assert c.weight_decay == 0
        signed = c.optimizer == 'signum'
        optimizer = optax.inject_hyperparams(sgd)(lr_schedule, c.b1, signed)

    if c.optimizer == 'adamw':
        assert c.b1 is not None
        assert c.b2 is not None
        optimizer = optax.inject_hyperparams(optax.adamw)(lr_schedule, c.b1, c.b2, weight_decay=c.weight_decay)
    
    if c.optimizer == 'adafactor':
        assert c.b1 is None
        assert c.b2 is not None
        optimizer = optax.inject_hyperparams(optax.adafactor, static_args='min_dim_size_to_factor')(lr_schedule, min_dim_size_to_factor=128, decay_rate=c.b2)
    

    if c.optimizer == 'muon':
        assert c.b1 is not None
        assert c.b2 is not None
        assert c.muon_lr is not None
        assert c.muon_b1 is not None
        muon_lr = optax.schedules.warmup_cosine_decay_schedule(0, c.muon_lr, warmup_steps, num_opt_steps)
        optimizer = optax.inject_hyperparams(muon)(muon_lr, c.muon_b1, lr_schedule, c.b1, c.b2)

    return optimizer


def sgd(
    learning_rate: float,
    b1: Optional[float] = None,
    signed = False,
) -> optax.GradientTransformation:
    return optax.chain(
        optax.trace(decay=b1) if b1 is not None else optax.identity(),
        optax.scale_by_sign() if signed else optax.identity(),
        optax.scale_by_learning_rate(learning_rate),
    )


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int = 5,
    eps: float = 1e-8,
) -> jax.Array:
    # https://github.com/google-deepmind/optax/blob/main/optax/contrib/_muon.py 
    if x.ndim < 2:
        raise ValueError(f'Input must have >= 2 dims, got {x.shape}')
    if ns_coeffs.shape != (3,):
        raise ValueError(f'ns_coeffs must have shape (3,), got {ns_coeffs}')
    def newton_schulz_iterator(x: jax.Array, coeffs: jax.Array) -> jax.Array:
        x_mT = jnp.swapaxes(x, -2, -1) # <-- changed (matrix transpose last 2 dims)
        a = x @ x_mT # <-- changed (use matrix transpose)
        b = coeffs[1] * a + coeffs[2] * a @ a
        return coeffs[0] * x + b @ x
    transposed = False
    if x.shape[-2] > x.shape[-1]: # <-- changed (check last 2 dims)
        x = jnp.swapaxes(x, -2, -1) # <-- changed (transpose last 2 dims)
        transposed = True
    x /= (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps) # <-- changed (normalize each matrix slice)
    ns_coeffs = ns_coeffs.astype(x.dtype)
    x = jax.lax.fori_loop(0, ns_steps, lambda _, x: newton_schulz_iterator(x, ns_coeffs), x)
    if transposed: x = jnp.swapaxes(x, -2, -1) # <-- changed (transpose last 2 dims)
    return x


class MuonState(NamedTuple):
    """State for the Adam algorithm."""
    count: jax.Array # shape=(), dtype=jnp.int32.
    mu: optax.Updates
    ns_coeffs: jax.Array # shape=(), dtype=jnp.int32.


def scale_by_muon(
    ns_coeffs: tuple = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    # https://github.com/google-deepmind/optax/blob/main/optax/contrib/_muon.py 

    def init_fn(params):
        mu = otu.tree_zeros_like(params) # First moment
        return MuonState(jnp.zeros([], jnp.int32), mu, jnp.asarray(ns_coeffs))

    def update_fn(updates, state, params=None):
        del params
        mu = otu.tree_update_moment(updates, state.mu, beta, 1)
        count_inc = optax.safe_increment(state.count)
        mu_hat = otu.tree_bias_correction(mu, beta, count_inc)
        # Apply Newton-schulz orthogonalization.
        updates = jax.tree.map(lambda x: orthogonalize_via_newton_schulz(x, state.ns_coeffs, ns_steps, eps), mu_hat)
        updates = jax.tree.map(lambda x: jnp.sqrt(jnp.maximum(1, x.shape[-1] / x.shape[-2])) * x, updates)
        return updates, MuonState(count_inc, mu, state.ns_coeffs)
    
    return optax.GradientTransformation(init_fn, update_fn)


def muon(
    learning_rate: float,
    muon_b1: float,
    adam_lr: float,
    adam_b1: float,
    adam_b2: float,
) -> optax.GradientTransformation:
    return optax.multi_transform(
        transforms={
            'muon': optax.chain(
                scale_by_muon(beta=muon_b1),
                optax.scale_by_learning_rate(learning_rate),
            ),
            'adam': optax.adamw(adam_lr, adam_b1, adam_b2)
        },
        param_labels=lambda params: jax.tree.map_with_path(
            lambda path, val: 'adam' if 'embed' in jax.tree_util.keystr(path) else 'muon', params
        ),
    )
