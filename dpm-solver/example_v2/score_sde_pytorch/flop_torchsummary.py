import gc
import io
import os
import time
import sys
sys.path.append("../")
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
from models import ddpm,ncsnpp,ncsnpp_multistage
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
from configs.vp import cifar10_ncsnpp_multistage_deep_continuous_v2 as configs
from sde_lib import VESDE, VPSDE, subVPSDE
from torchsummary import summary
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
config=configs.get_config()
config.eval.batch_size = 1
sampling_eps = 1e-3
# Build data pipeline
train_ds, eval_ds = datasets.get_dataset(config,
                                            uniform_dequantization=config.data.uniform_dequantization,
                                            evaluation=True)

# Create data normalizer and its inverse
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)
optimizer = losses.get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

sampling_eps = 1e-3
sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)

batch = torch.rand(sampling_shape).to(config.device)
batch = scaler(batch)
lst_steps=torch.tensor([0.7])*1000
t = lst_steps.to('cuda')#.repeat(batch.shape[0])
class mfn(torch.nn.Module):
    def __init__(self,m):
        super().__init__()
        self.m = m
    def forward(self,x):
        print(x.shape)
        return self.m(x,t)
summary(mfn(score_model).to('cuda'),sampling_shape[1:])
