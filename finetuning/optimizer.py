import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu
from flax import nnx
import factorized, utils
from typing import Optional


class ModelAndOptimizer(nnx.Optimizer):
    """
    Extends nnx.ModelAndOptimizer (v0.12.0) with stochastic rounding.
    """
    def __init__(self, model, tx, wrt=nnx.Param, stochastic_round=False):
        super().__init__(model, tx, wrt=wrt)
        self.model = model
        self.stochastic_round = stochastic_round # <- CHANGED: added stochastic_round support

    def update(self, key, grads, **kwargs):
        param_arrays = nnx.to_arrays(nnx.pure(nnx.state(self.model, self.wrt)))
        grad_arrays = nnx.to_arrays(nnx.pure(nnx.state(grads)))
        opt_state_arrays = nnx.to_arrays(nnx.pure(self.opt_state))
        kwargs_arrays = nnx.to_arrays(nnx.pure(kwargs))

        updates, new_opt_state = self.tx.update(grad_arrays, opt_state_arrays, param_arrays, **kwargs_arrays)
        new_params = apply_updates(key, param_arrays, updates, self.stochastic_round) # <- CHANGED: added stochastic_round support

        nnx.update(self.model, new_params)
        nnx.update(self.opt_state, nnx.state(new_opt_state))
        self.step[...] += 1


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
