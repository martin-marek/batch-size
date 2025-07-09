import wandb
import pandas as pd
import numpy as np
wandb_api = wandb.Api()


def load_sweeps(sweep_names, entity='martin-nyu', project='picodo-bs'):
    sweeps = wandb_api.project(project, entity).sweeps()
    sweeps = [sweep for sweep in sweeps if sweep.name in sweep_names]
    print(f'{len(sweeps)=}')
    runs = []
    for sweep in sweeps:
        sweep.runs.per_page = len(sweep.runs) # required to load all runs: https://github.com/wandb/wandb/issues/7666
        for run in sweep.runs:
            run_data = {'id': run.id} | run.config | dict(run.summary)
            runs += [run_data]
    df = pd.DataFrame(runs)
    return df


def halflife_to_decay(t_token, n_batch=1):
    """
    notation:
    - t_token: halflife measured in number of tokens
    - t_steps: halflife measured in number of steps
    - n_batch: number of tokens per batch
    - d: decay coefficient
    """
    t_steps = t_token / n_batch # halflife (measured in number of steps)
    d = (1/2)**(1/t_steps)
    return d


def decay_to_halflife(d, n_batch=1):
    """
    notation:
    - t_token: halflife measured in number of tokens
    - t_steps: halflife measured in number of steps
    - n_batch: number of tokens per batch
    - d: decay coefficient
    """
    # note: d**t_steps = 1/2
    t_steps = np.log(1/2) / np.log(d)
    t_token = t_steps * n_batch
    return t_token
