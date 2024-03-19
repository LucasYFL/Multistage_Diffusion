import gc
import io
import os
import time

import numpy as np
# import tensorflow as tf
# import tensorflow_gan as tfgan
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp, ncsnpp_multistage
import sampling
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from models import utils as mutils
import datasets
import sde_lib
from absl import flags
import torch
import losses
from models.ema import ExponentialMovingAverage
from torch.utils import tensorboard
from utils import save_checkpoint, restore_checkpoint
import pytorch_fid.fid_score as FID_score
from cluster import interval_cluster
from torchmetrics.functional import pairwise_cosine_similarity
import logging
FLAGS = flags.FLAGS


config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")

flags.mark_flags_as_required(["workdir", "config"])
def get_loss_fn(sde, train,reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5):
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
  def loss_fn(model,batch,timestep):
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    t = torch.ones(batch.shape[0], device=batch.device)*timestep
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t)
    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2
    loss = torch.mean(losses)
    return loss

  return loss_fn

def get_gradients_aff(config, workdir,timesteps):
  # return gradients in shape (ckpt, timestep, param)
  # Initialize model.
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
  
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  

  # Build data iterators
  train_ds, _ = datasets.get_dataset(config)

  train_iter = iter(train_ds)
  logging.info("prepare data")
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  train=True
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  loss_fn = get_loss_fn(sde, train, reduce_mean, continuous, likelihood_weighting, sampling_eps)
  try:
    batch, _ = next(train_iter)
  except StopIteration:
    train_iter = datasets.distributed_dataset(train_ds, config)
    batch, _ = next(train_iter)
  if config.device == torch.device('cuda'):
    batch = batch.to(f"{config.device}")  
  else:
    batch = batch.to(config.device)
  batch = scaler(batch)
  logging.info("one batch")
  grads = []
  begin_ckpt = config.eval.begin_ckpt
  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth') 
    state = restore_checkpoint(ckpt_path, state, device=f"{config.device}")
    score_model.train()
    grad_ckpt = []
    for t in timesteps:
        score_model.zero_grad()
        loss = loss_fn(score_model, batch,t)
        loss.backward()
        grad_ckpt.append(torch.cat([param.grad.flatten().to('cpu') for param in score_model.parameters()]))
    grad_ckpt = torch.stack(grad_ckpt)
    logging.info("end ckpt:{ckpt}")
    grads.append(pairwise_cosine_similarity(grad_ckpt))
  return torch.mean(torch.stack(grads),dim=0)

def main(argv):
  config = FLAGS.config
  device = config.device
  timesteps =(torch.arange(0,100,device = device)+1)/100
  affinity = get_gradients_aff(config, FLAGS.workdir,timesteps) 
  
  def TAS_cost(arr):
    center = arr[(len(arr))//2]
    return -1*torch.sum(affinity[center,arr])
  cl = interval_cluster(np.arange(100),3)
  logging.info("start clustering")
  ind = cl.calculate(TAS_cost)[1]
  logging.info(timesteps[ind])
if __name__ == "__main__":
  app.run(main)