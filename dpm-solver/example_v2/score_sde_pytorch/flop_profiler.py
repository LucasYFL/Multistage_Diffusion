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
from configs.vp import cifar10_ddpmpp_deep_continuous as configs
from sde_lib import VESDE, VPSDE, subVPSDE
from torch.profiler import profile, record_function, ProfilerActivity
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
config=configs.get_config()
config.eval.batch_size = 1
sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
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


sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
sampling_eps = 1e-3
sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
# batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()
# batch = batch.permute(0, 3, 1, 2)
batch = torch.rand(sampling_shape).to(config.device)
batch = scaler(batch)
#lst_steps=torch.tensor([0.050950001925230026])
lst_steps=torch.tensor([0.7])
t = lst_steps.to(batch.device).repeat(batch.shape[0])
z = torch.randn_like(batch)
mean, std = sde.marginal_prob(batch, t)
perturbed_data = mean + std[:, None, None, None] * z
with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], record_shapes=True,with_flops=True) as prof:
    with record_function("model_inference"):
        score_model(perturbed_data,t)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))