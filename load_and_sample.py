import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from dataset import MyCelebA
from pytorch_lightning.plugins import DDPPlugin


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['model_params']['name'])


checkpoint_path = ''.join([config['logging_params']['save_dir'], r"VAE_REP/version_3/checkpoints/last.ckpt"])
checkpoint = torch.load(checkpoint_path)

model = vae_models[config['model_params']['name']](**config['model_params'])

experiment = VAEXperiment(model,
                          config['exp_params'])
experiment.load_state_dict(checkpoint["state_dict"])
experiment.sample(latent_var=1, path=config['logging_params']['results_dir'])

for i in range(0.0, 1.0, 0.25):
    experiment.sample(latent_var=i, path=config['logging_params']['results_dir'])
