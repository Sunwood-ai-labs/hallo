import os
from omegaconf import OmegaConf

def load_config(config_path, args):
    config = OmegaConf.load(config_path)
    config = OmegaConf.merge(config, vars(args))
    return config