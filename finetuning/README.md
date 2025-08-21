# Fine-tuning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/martin-marek/batch-size/blob/main/finetuning/finetune.ipynb)

For our fine-tuning experiments, we implemented Gemma 3 mostly from scratch, including sampling. Our code is loosely based on the [Gemma NNX example](https://github.com/google/flax/tree/main/examples/gemma).

## Manual training

For running experiments, we recommend interfacing with Python, although Bash is also supported thanks to [Fire](https://github.com/google/python-fire).

```python
# Python
from finetune import finetune
finetune(model_variant='gemma3-1b')
```

```bash
# Bash
python finetune.py --model_variant='gemma3-1b'
```

To quickly get started, we provide a [Colab Notebook](https://colab.research.google.com/github/martin-marek/batch-size/blob/main/finetuning/finetune.ipynb) to fine-tune Gemma 3 (12B) using a TPU v6e-1 with just 32 GB of memory.

## Sweeps

We performed our main experiment using Weights & Biases sweeps. The config files for each sweep can be found in the [runs](runs) directory.
