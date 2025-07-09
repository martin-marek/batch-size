# Pretraining


## Configs

We used Hydra for config management. All model and dataset configs can be found in the [configs](configs) directory.

## Manual training

Here's an example of setting up a manual training run:
```bash
python main.py +model=gpt3xl +dataset=fw_gpt2
```
## Sweeps

We performed almost all of our experiments using Weights & Biases sweeps. The config files for each sweep can be found in the [runs](runs) directory.