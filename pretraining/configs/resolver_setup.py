import operator as op
from omegaconf import OmegaConf

OmegaConf.register_new_resolver('floordiv', op.floordiv)
OmegaConf.register_new_resolver('mul', op.mul)
OmegaConf.register_new_resolver('min', min)
OmegaConf.register_new_resolver('pow', pow)