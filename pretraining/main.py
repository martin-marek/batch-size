import hydra
from train import train_and_evaluate
from configs import resolver_setup
from omegaconf import OmegaConf, DictConfig
from utils import flatten_dict


# load default config
@hydra.main(version_base=None, config_path='configs', config_name='base')
def main(c: DictConfig):

    # optionally load batch size config
    if 'bs_configs' in c:
        bs_config = c.bs_configs[f'bs{c.opt.batch_size}']
        c = OmegaConf.merge(c, bs_config)
        del c.bs_configs

    # optionally overwrite any values
    if 'overwrite' in c:
        c = OmegaConf.merge(c, c.overwrite)
        del c.overwrite

    # optionally translate 1d scaling to generalized scaling config
    if 'scaling_1d' in c:
        OmegaConf.update(c, f'scaling.{c.scaling_1d.key}', c.scaling_1d.value, force_add=True)
        del c.scaling_1d
    
    # optionally apply generalized scaling config
    if 'scaling' in c:
        for k, scaling in flatten_dict(c.scaling).items():
            orig_val = OmegaConf.select(c, k)
            OmegaConf.update(c, k, scaling * orig_val)
        del c.scaling

    # run training job
    OmegaConf.resolve(c)
    print(OmegaConf.to_yaml(c))
    train_and_evaluate(c)


if __name__ == '__main__':
    main()
