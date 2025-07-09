# Fine-tuning

For our fine-tuning experiments, we implemented Gemma 3 mostly from scratch, including sampling. Our code is loosely based on the [Gemma NNX example](https://github.com/google/flax/tree/main/examples/gemma).

## Manual training

For runnign experiments, we recommend interfacing with Python, although Bash is also supported thanks to [Fire](https://github.com/google/python-fire).

```python
# Python
from finetune import finetune
finetune(model_variant='gemma3-1b')
```

```bash
# Bash
python finetune.py --model_variant='gemma3-1b'
```

## Sweeps

We performed our main experiments using Weights & Biases sweeps. The config files for each sweep can be found in the [runs](runs) directory.