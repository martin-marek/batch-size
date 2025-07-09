# Small Batch Size Training for Language Models

Official repository for the paper *[Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful](https://arxiv.org/abs/TODO)*

[![](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg)](https://arxiv.org/abs/TODO)

## Key results

We show that when a small batch size is used, vanilla SGD without momentum is almost as fast as AdamW for LLM pretraining on a per-FLOP basis. In general, we find that as the batch size is reduced, the performance gaps between optimizers shrinks.

<img src="https://github.com/martin-marek/batch-size/blob/main/plots/gpt3xl_sgd.png" width="450">

Additionally, small batch sizes are much more robust to hyperparameter mispecification, meaning that when the tuning budget is limited, small batch sizes perform much better in expecation.

<img src="https://github.com/martin-marek/batch-size/blob/main/plots/adam_2d.png" width="650">

We hope that our results can be useful for memory-constrained practitioners, since small batch sizes allow the use of simple optimizers. For example, instead of using LoRA for fine-tuning, it might be preferable to do full fine-tuning with a small batch size and a memory-efficient optimizer like Adafactor, matching the performance of Adam while maintaining a similar memory footprint to LoRA.

<img src="https://github.com/martin-marek/batch-size/blob/main/plots/finetune_bar.png" width="400">

## Code structure

We implemented all of our experiments in JAX from scratch, using a mix of data and tensor parallelism. We used two independent codebases for [pretraining](pretraining) and [fine-tuning](finetuning). Please refer to either codebase for more details on running experiments.

All of our visualizations were done using Jupyter Notebooks found in the (utils)[utils] directory.

## Citation

```bibtex
@misc{smallbatch,
  title={Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful}, 
  author={Martin Marek and Sanae Lotfi and Aditya Somasundaram and Andrew Gordon Wilson and Micah Goldblum},
  year={2025},
  eprint={TODO},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
