"""based on https://github.com/google/flax/tree/main/examples/gemma"""

import os
import dataclasses
from itertools import cycle
from contextlib import redirect_stderr

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P, NamedSharding
from rope import apply_rope


@dataclasses.dataclass(frozen=True)
class GemmaConfig:
    num_layers: int
    embed_dim: int
    hidden_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    query_pre_attn_scalar: float
    vocab_size: int = 262_144
    local_base_frequency: int = 10_000
    global_base_frequency: int = 1_000_000
    local_scale_factor: float = 1.0
    global_scale_factor: float = 1.0
    sliding_window_size: int | None = None
    attention_pattern: tuple[str] = (*(['sliding']*5), 'global')
    activ_dtype: str = 'float32'
    param_dtype: str = 'float32'
    remat: bool = False


    @classmethod
    def gemma3_1b(cls, param_dtype='float32', remat=False):
        return cls(
            num_layers=26,
            embed_dim=1152,
            hidden_dim=6912,
            num_heads=4,
            head_dim=256,
            num_kv_heads=1,
            query_pre_attn_scalar=256**-0.5, # 1/sqrt(head_dim)
            sliding_window_size=512,
            global_scale_factor=1.0,
            param_dtype=param_dtype,
            remat=remat,
        )

    @classmethod
    def gemma3_4b(cls, param_dtype='float32', remat=False):
        return cls(
            num_layers=34,
            embed_dim=2560,
            hidden_dim=10240,
            num_heads=8,
            head_dim=256,
            num_kv_heads=4,
            query_pre_attn_scalar=256**-0.5, # 1/sqrt(head_dim)
            sliding_window_size=1024,
            global_scale_factor=8.0,
            param_dtype=param_dtype,
            remat=remat,
        )

    @classmethod
    def gemma3_12b(cls, param_dtype='float32', remat=False):
        return cls(
            num_layers=48,
            embed_dim=3840,
            hidden_dim=15360,
            num_heads=16,
            head_dim=256,
            num_kv_heads=8,
            query_pre_attn_scalar=256**-0.5, # 1/sqrt(head_dim)
            sliding_window_size=1024,
            global_scale_factor=8.0,
            param_dtype=param_dtype,
            remat=remat,
        )

    @classmethod
    def gemma3_27b(cls, param_dtype='float32', remat=False):
        return cls(
            num_layers=62,
            embed_dim=5376,
            hidden_dim=21504,
            num_heads=32,
            head_dim=128,
            num_kv_heads=16,
            query_pre_attn_scalar=(5376/32)**-0.5, # 1/sqrt(embed_dim / num_heads)
            sliding_window_size=1024,
            global_scale_factor=8.0,
            param_dtype=param_dtype,
            remat=remat,
        )


class Gemma(nnx.Module):
    def __init__(self, c: GemmaConfig, rngs: nnx.Rngs):
        self.in_embed = nnx.Embed(c.vocab_size, c.embed_dim, dtype=c.activ_dtype, param_dtype=c.param_dtype, rngs=rngs)
        self.layers = nnx.List(
            TransformerBlock(
                num_heads=c.num_heads,
                num_kv_heads=c.num_kv_heads,
                embed_dim=c.embed_dim,
                head_dim=c.head_dim,
                hidden_dim=c.hidden_dim,
                query_pre_attn_scalar=c.query_pre_attn_scalar,
                sliding_window_size=c.sliding_window_size if attn_type == 'sliding' else None,
                rope_base_frequency=c.local_base_frequency if attn_type == 'sliding' else c.global_base_frequency,
                rope_scale_factor=c.local_scale_factor if attn_type == 'sliding' else c.global_scale_factor,
                activ_dtype = c.activ_dtype,
                param_dtype = c.param_dtype,
                rngs=rngs,
            ) for _, attn_type in zip(range(c.num_layers), cycle(c.attention_pattern))
        )
        self.final_norm = nnx.RMSNorm(c.embed_dim, dtype=c.activ_dtype, param_dtype=c.param_dtype, rngs=rngs)
        self.remat = c.remat

    def __call__(
        self,
        tokens, # [B, T]
        kv_cache = {}, # [B, S]
        positions = None, # [B, S]
        attn_mask = None, # [B, S, S]
    ):
        V, D = self.in_embed.embedding.value.shape
        x = self.in_embed(tokens) * jnp.sqrt(D) # [B, T, D]

        for i, layer in enumerate(self.layers):
            layer_fn = jax.remat(layer) if self.remat else layer
            x, kv_cache[i] = layer_fn(x, kv_cache.get(i), positions, attn_mask) # [B, T, D]
        
        x = self.final_norm(x)
        logits = jnp.dot(x, self.in_embed.embedding.value.T) # [B, T, V]

        return logits, kv_cache


    def init_kv_cache(self, batch_size, max_seq_len):
        kv_cache = {
            i: layer.attn.init_kv_cache(batch_size, max_seq_len)
            for i, layer in enumerate(self.layers)
        }
        return kv_cache


class Attention(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        embed_dim: int,
        head_dim: int,
        query_pre_attn_scalar: float,
        rope_base_frequency: int,
        rope_scale_factor: float,
        sliding_window_size: int | None = None,
        activ_dtype = 'float32',
        param_dtype = 'float32',
        *, rngs: nnx.Rngs,
    ):
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.sliding_window_size = sliding_window_size
        self.rope_base_frequency = rope_base_frequency
        self.rope_scale_factor = rope_scale_factor
        self.attn_vec_einsum = nnx.Einsum(einsum_str='BTNH,NHD->BTD', kernel_shape=(num_heads, head_dim, embed_dim), dtype=activ_dtype, param_dtype=param_dtype, rngs=rngs)
        self.q_einsum = nnx.Einsum(einsum_str='BTD,NDH->BTNH', kernel_shape=(num_heads, embed_dim, head_dim), dtype=activ_dtype, param_dtype=param_dtype, rngs=rngs)
        self.kv_einsum = nnx.Einsum(einsum_str='BSD,CKDH->CBSKH', kernel_shape=(2, num_kv_heads, embed_dim, head_dim), dtype=activ_dtype, param_dtype=param_dtype, rngs=rngs)
        self._query_norm = nnx.RMSNorm(head_dim, dtype=activ_dtype, param_dtype=param_dtype, rngs=rngs)
        self._key_norm = nnx.RMSNorm(head_dim, dtype=activ_dtype, param_dtype=param_dtype, rngs=rngs)

    def __call__(self,
        x, # [B, T, D]
        kv_cache = None, # [B, S]
        positions = None, # [B, S]
        attn_mask = None, # [B, T, S]
    ):
        B, T, D = x.shape
        N, D, H = self.q_einsum.kernel.value.shape
        S = T if kv_cache is None else kv_cache['v'].shape[1]

        # qkv projection
        query = self.q_einsum(x) # [B, T, N, H]
        key, value = self.kv_einsum(x) # [B, T, N, H]

        # qk norm
        query = self._query_norm(query)
        key = self._key_norm(key)

        # get token indices
        if positions is None:
            # training
            if kv_cache is None:
                positions = jnp.broadcast_to(jnp.arange(T)[None, :], [B, S])
            # sampling
            else:
                positions = jnp.full([B, 1], kv_cache['end_idx']) # [B, 1]

        # apply positional embeddings
        query = apply_rope(query, positions, self.rope_base_frequency, self.rope_scale_factor)
        key = apply_rope(key, positions, self.rope_base_frequency, self.rope_scale_factor)

        # load kv cache
        if kv_cache is not None:
            cache_dtype = kv_cache['k'].dtype
            key = kv_cache['k'].at[:, kv_cache['end_idx'], :, :].set(key[:, 0].astype(cache_dtype)) # [B, S, N, H]
            value = kv_cache['v'].at[:, kv_cache['end_idx'], :, :].set(value[:, 0].astype(cache_dtype)) # [B, S, N, H]
            query = query.astype(cache_dtype)

        # compute attention mask [B, T, S]
        if attn_mask is None:
            # if training, use lower triangular mask
            if kv_cache is None:
                attn_mask = jnp.tri(T, dtype=jnp.bool_)[None] # [B, T, S]
            # if sampling, all cached tokens are visible
            else:
                attn_mask = (jnp.arange(S) <= kv_cache['end_idx'])[None, None] # [B, 1, S]

        # add window to attention mask (TODO)
        if self.sliding_window_size is not None:
            offset = 0 if kv_cache is None else kv_cache['end_idx']
            t, s = jnp.mgrid[0:T, 0:S]
            sliding_mask = (t - self.sliding_window_size + 1 + offset) <= s
            attn_mask &= sliding_mask[None]

        # gqa attention
        attn_mask = jnp.broadcast_to(attn_mask[:, None, :, :], [B, N, T, S])
        encoded = jax.nn.dot_product_attention(query, key, value, mask=attn_mask, scale=self.query_pre_attn_scalar)

        # output projection
        attn_output = self.attn_vec_einsum(encoded)

        # update kv cache
        if kv_cache is not None:
            kv_cache = {'k': key, 'v': value, 'end_idx': kv_cache['end_idx'] + T}

        return attn_output, kv_cache

    def init_kv_cache(self, batch_size, max_seq_len):
        w = self.kv_einsum.kernel.value
        _, num_kv_heads, _, head_dim = w.shape
        sharding = NamedSharding(w.sharding.mesh, P('data', None, 'model', None)) if hasattr(w, 'sharding') else None
        kv_cache = {
            'k': jnp.zeros((batch_size, max_seq_len, num_kv_heads, head_dim), dtype=jnp.bfloat16, device=sharding),
            'v': jnp.zeros((batch_size, max_seq_len, num_kv_heads, head_dim), dtype=jnp.bfloat16, device=sharding),
            'end_idx': jnp.array(0, dtype=jnp.int32),
        }
        return kv_cache


class MLP(nnx.Module):
    def __init__(self, embed_dim, hidden_dim, activ_dtype='float32', param_dtype='float32', *, rngs):
        self.gate_proj = nnx.Linear(embed_dim, hidden_dim, use_bias=False, dtype=activ_dtype, param_dtype=param_dtype, kernel_init=jax.nn.initializers.normal(), rngs=rngs)
        self.up_proj = nnx.Linear(embed_dim, hidden_dim, use_bias=False, dtype=activ_dtype, param_dtype=param_dtype, kernel_init=jax.nn.initializers.normal(), rngs=rngs)
        self.down_proj = nnx.Linear(hidden_dim, embed_dim, use_bias=False, dtype=activ_dtype, param_dtype=param_dtype, kernel_init=jax.nn.initializers.normal(), rngs=rngs)

    def __call__(self, x):
        activations = nnx.gelu(self.gate_proj(x)) * self.up_proj(x)
        outputs = self.down_proj(activations)
        return outputs


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        embed_dim: int,
        head_dim: int,
        hidden_dim: int,
        query_pre_attn_scalar: float,
        rope_base_frequency: int,
        rope_scale_factor: float,
        sliding_window_size: int | None = None,
        activ_dtype = 'float32',
        param_dtype = 'float32',
        *, rngs: nnx.Rngs,
    ):
        self.attn = Attention(
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            embed_dim=embed_dim, head_dim=head_dim, query_pre_attn_scalar=query_pre_attn_scalar,
            rope_base_frequency=rope_base_frequency, rope_scale_factor=rope_scale_factor,
            sliding_window_size=sliding_window_size, param_dtype=param_dtype, activ_dtype=activ_dtype, rngs=rngs,
        )
        self.mlp = MLP(embed_dim, hidden_dim, activ_dtype=activ_dtype, param_dtype=param_dtype, rngs=rngs)
        self.pre_attention_norm = nnx.RMSNorm(embed_dim, dtype=activ_dtype, param_dtype=param_dtype, rngs=rngs)
        self.post_attention_norm = nnx.RMSNorm(embed_dim, dtype=activ_dtype, param_dtype=param_dtype, rngs=rngs)
        self.pre_ffw_norm = nnx.RMSNorm(embed_dim, dtype=activ_dtype, param_dtype=param_dtype, rngs=rngs)
        self.post_ffw_norm = nnx.RMSNorm(embed_dim, dtype=activ_dtype, param_dtype=param_dtype, rngs=rngs)

    def __call__(self,
        x, # [B, T, D]
        kv_cache = None, # [B, S]
        positions = None, # [B, S]
        attn_mask = None, # [B, S, S]
    ):

        # attention
        attn_inputs = self.pre_attention_norm(x)
        attn_output, kv_cache = self.attn(attn_inputs, kv_cache, positions, attn_mask)
        attn_output = self.post_attention_norm(attn_output)
        x += attn_output

        # MLP
        ffw_inputs = self.pre_ffw_norm(x)
        ffw_outputs = self.mlp(ffw_inputs)
        ffw_outputs = self.post_ffw_norm(ffw_outputs)
        x += ffw_outputs

        return x, kv_cache


def load_pretrained(model_variant, mesh=None, param_dtype='float32', remat=False):
    import kagglehub
    import sentencepiece as spm
    import orbax.checkpoint as ocp

    # helpers
    flatten_path = lambda path: jax.tree_util.keystr(path, simple=True, separator='/')
    flatten_tree = lambda tree: {flatten_path(path):v for path, v in jax.tree.leaves_with_path(tree)}

    # download weights
    with open(os.devnull, 'w') as f, redirect_stderr(f): # supress progress bar
        weights_dir = kagglehub.model_download(f'google/gemma-3/flax/{model_variant}')
    ckpt_path = f'{weights_dir}/{model_variant}'
    vocab_path = f'{weights_dir}/tokenizer.model'

    # load tokenizer
    vocab = spm.SentencePieceProcessor()
    vocab.Load(vocab_path)

    # load abstract model
    model_architecture = '_'.join(model_variant.split('-')[:2])
    model_config = getattr(GemmaConfig, model_architecture)(param_dtype, remat)
    model = nnx.eval_shape(lambda: Gemma(model_config, rngs=nnx.Rngs(0)))
    model_state = nnx.state(model)

    # load checkpoint metadata
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    checkpoint = checkpointer.metadata(ckpt_path).item_metadata

    # add checkpoint sharding annotations
    def add_sharding(path, v):
        key = flatten_path(path)
        pspec = None
        if 'input_embedding' in key: pspec = P('data', 'model') # [V, D]
        if '_norm' in key: pspec = P('data') # [D | H]
        if 'attn_vec_einsum' in key: pspec = P('model', None, 'data') # [N, H, D]
        if 'kv_einsum' in key: pspec = P(None, 'model', 'data', None) # [2, n, D, H]
        if 'q_einsum' in key: pspec = P('model', 'data', None) # [N, D, H]
        if 'mlp/linear' in key: pspec = P('model', 'data') # [F, D]
        if 'mlp/gating_einsum' in key: pspec = P(None, 'model', 'data') # [2, F, D]
        # if pspec is None: print(f'WARNING: {key} has no sharding!')
        sharding = None if (pspec is None or mesh is None) else NamedSharding(mesh, pspec)
        return jax.ShapeDtypeStruct(v.shape, param_dtype, sharding=sharding)
    checkpoint = jax.tree.map_with_path(add_sharding, checkpoint)

    # load checkpoint weights
    checkpoint = checkpointer.restore(ckpt_path, checkpoint)

    # flatten the checkpoint keys
    checkpoint = flatten_tree(checkpoint)

    # adjust checkpoint weights to match NNX format
    for key in list(checkpoint.keys()):
        if 'scale' in key:
            checkpoint[key] += 1
        if 'gating_einsum' in key:
            checkpoint[key.replace('gating_einsum', 'gate_proj')] = checkpoint[key][0].T
            checkpoint[key.replace('gating_einsum', 'up_proj')] = checkpoint[key][1].T
            del checkpoint[key]

    # transfer weights to model, mapping model layer keys to checkpoint keys
    def get_weights(path, v):
        key = flatten_path(path)
        key = 'transformer/' + key
        key = key.replace('/value', '')
        key = key.replace('layers/', 'layer_')
        key = key.replace('kernel', 'w')
        key = key.replace('in_embed/embedding', 'embedder/input_embedding')
        key = key.replace('mlp/down_proj', 'mlp/linear')
        return checkpoint[key]
    model_state = jax.tree.map_with_path(get_weights, model_state)
    nnx.update(model, model_state)

    return model, vocab
