import os
import jax
import numpy as np
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh, NamedSharding


def load_ds(key, mesh, ds_path, seq_len, batch_size, n_tokens_valid, n_tokens_train=None):

    # get dataset size
    print('getting dataset size...')
    ds_path = os.path.expanduser(ds_path)
    data = np.memmap(ds_path, dtype=np.uint16, mode='r')
    n_tokens_dataset = len(data)
    n_seq_dataset = n_tokens_dataset // seq_len

    # if n_tokens_train is None, use full dataset
    if n_tokens_train is not None: assert n_tokens_train + n_tokens_valid <= n_tokens_dataset
    if n_tokens_train is None: n_tokens_train = n_tokens_dataset - n_tokens_valid

    # get num. of train. and valid. batches
    n_batch_train = n_tokens_train // (batch_size * seq_len)
    n_batch_valid = n_tokens_valid // (batch_size * seq_len)
    n_batch = n_batch_train + n_batch_valid

    # memmap data
    print('reading data...')
    data = np.memmap(ds_path, dtype=np.uint16, shape=[n_batch, batch_size, seq_len], mode='r')
    
    # load data onto jax devices, sharded across batch dimension
    sharding = jax.sharding.NamedSharding(mesh, P(None, 'data', 'model'))
    callback = lambda index: data[index]
    data = jax.make_array_from_callback(data.shape, sharding, callback)

    # shuffle batches
    print('shuffling data...')
    data = jax.random.permutation(key, data, axis=0)

    # split data
    print('splitting data...')
    data_train = data[:n_batch_train]
    data_valid = data[n_batch_train:]
    
    return data_train, data_valid


def pad_mask(batch, eos_token_id=1):
    B, L = batch.shape

    # get idx of last EOS token
    # if there is no EOS token, equals L-1
    idx_last_eos_token = (L - 1) - jnp.argmax(batch[:, ::-1] == eos_token_id, axis=1)
    
    # only use tokens before the last EOS token
    mask = jnp.arange(L)[None, :] <= idx_last_eos_token[:, None]

    return mask # [B, L]
