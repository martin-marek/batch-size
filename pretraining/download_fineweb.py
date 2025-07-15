import os
import fire
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars; disable_progress_bars()
from typing import Optional, Literal


def load_data_shard(file):
    # https://github.com/KellerJordan/modded-nanogpt/blob/a202a3a0ca99d69bb7f847e5337c7c6e0890fd92/train_gpt.py#L411
    header = np.fromfile(file, dtype=np.int32, count=256) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with Path(file).open("rb", buffering=0) as f:
        tokens = np.empty(num_tokens, dtype=np.uint16) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def download_dataset(
        dataset: Literal['fineweb', 'finewebedu'] = 'fineweb',
        num_chunks: Optional[int] = None,
    ):
    """download dataset, save it as a np.memmap binary file"""

    # get num. chunks
    # by default, download all chunnks (10B tokens)
    # each chunk is 100M tokens
    if num_chunks is None:
        if dataset == 'fineweb': num_chunks = 103
        if dataset == 'finewebedu': num_chunks = 99

    # load chunks into memory
    print('downloading...')
    shards = []
    for i in tqdm(range(1, num_chunks+1)):
        shard_path = hf_hub_download(repo_id=f'kjj0/{dataset}10B-gpt2', filename=f'{dataset}_train_{i:06}.bin', repo_type="dataset")
        shards += [load_data_shard(shard_path)]

    # save to disk
    print('saving...')
    out_dir = os.path.expanduser('~/datasets')
    out_path = f'{out_dir}/{dataset}_gpt2.bin'
    os.makedirs(out_dir, exist_ok=True)
    n_tokens = sum(map(len, shards))
    out = np.memmap(out_path, dtype=np.uint16, mode='w+', shape=[n_tokens])
    i = 0
    for shard in tqdm(shards):
        out[i:i+len(shard)] = shard
        i += len(shard)
    out.flush()


if __name__ == '__main__':
    fire.Fire(download_dataset)
