from dataset import get_dataset
import sde_lib 

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

import torch
import logging
import os

import time
import torch.multiprocessing as mp

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

def normal_distribution(x, y_batch, s, std, bias = 0):
  bs = y_batch.shape[0]
  prob = torch.exp(-(((x - s * y_batch)**2).view(bs, -1).sum(dim=1).to(torch.float64)/std**2)/2 - bias)
  prob_y = prob.clone().view(-1, 1, 1, 1) * y_batch
  return prob.sum(dim=0, keepdim=True), prob_y.sum(dim=0, keepdim=True)

def get_exp_bias(x, y_batch, s, std):
  ## because exp() might return a very small number, we need a bias
  bs = y_batch.shape[0]
  return (-(((x - s * y_batch)**2).view(bs, -1).sum(dim=1).to(torch.float64)/std**2)/2).max()

def get_optimal_sol(batch, z, sde, t, dataloader, config):
    s, sigma = sde.transform_prob(torch.tensor([t]))
    std = s * sigma
    x = s * batch + s * sigma[:, None, None, None] * z
    prob_sum = 0.
    prob_y_sum = torch.zeros_like(x).to(torch.float64)
    y_sum = torch.zeros_like(x)
    exp_bias = -(torch.inf)
    for y_batch, _ in dataloader:
      exp_bias = max(exp_bias, get_exp_bias(x, y_batch, s, std))
    for y_batch, _ in dataloader:
      prob, prob_y = normal_distribution(x, y_batch, s, std, exp_bias)
      prob_sum += prob
      prob_y_sum += prob_y
      y_sum += y_batch.sum(dim=0, keepdim=True)
    assert config.exp.loss_func in ["epsilon", "x0"]
    if config.exp.loss_func == "x0":
      optimal_solution = prob_y_sum/prob_sum
    elif config.exp.loss_func == "epsilon":
      optimal_solution = prob_y_sum/prob_sum
    return optimal_solution

def generate_sample(dataset, config, t1, t2, dataloader, sde):
    optimal_solutiont1s = []
    optimal_solutiont2s = []
    t1s = []
    t2s = []
    for i in range(config.exp.sampling_num):
      t = time.time()
      randn_idx = torch.randint(len(dataset), (1, ))
      batch = dataset[randn_idx][0]
      z = torch.randn_like(batch)
      optimal_solutiont1 = get_optimal_sol(batch, z, sde, t1, dataloader, config)
      optimal_solutiont2 = get_optimal_sol(batch, z, sde, t2, dataloader, config)
      if (optimal_solutiont1.isnan().sum() + optimal_solutiont2.isnan().sum()) == 0:
        optimal_solutiont1s.append(optimal_solutiont1)
        optimal_solutiont2s.append(optimal_solutiont2)
        t1s.append(t1.view(1, -1))
        t2s.append(t2.view(1, -1))
      t_end = time.time() - t
      logging.info(f"The {config.host_id} host finished {i + 1}th sampling generation, with {t_end}s") 
      if (i + 1) % config.exp.num_save == 0:
        if not os.path.isdir(config.exp.save_dir):
          os.mkdir(config.exp.save_dir)
        host_pkg = os.path.join(config.exp.save_dir, str(config.host_id))
        if not os.path.isdir(host_pkg):
          os.mkdir(host_pkg)
        torch.save({
              'optimal_solutiont1s': torch.concat(tuple(optimal_solutiont1s)),
              'optimal_solutiont2s': torch.concat(tuple(optimal_solutiont2s)),
              't1s': torch.concat(tuple(t1s)),
              't2s': torch.concat(tuple(t2s)),
              }, os.path.join(host_pkg, f"{i//config.exp.num_save}_{t1:.4f}_{t2:.4f}.pth"))
        optimal_solutiont1s = []
        optimal_solutiont2s = []
        t1s = []
        t2s = []

def main(argv):
  config = FLAGS.config
  dataset, _ = get_dataset(config)
  if config.sde.type.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.sde.beta_min, 
                        beta_max=config.sde.beta_max, 
                        N=config.sde.num_scales)
    eps = 1e-3
  elif config.sde.type.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.sde.beta_min, 
                           beta_max=config.sde.beta_max, 
                           N=config.sde.num_scales)
    eps = 1e-3
  elif config.sde.type.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.sde.sigma_min, 
                        sigma_max=config.sde.sigma_max, 
                        N=config.sde.num_scales)
    eps = 1e-5
  dataloader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=config.data.batch_size, 
                                           shuffle=True, 
                                           num_workers=4)
  l = torch.cat((torch.range(eps, 1, 0.05), torch.tensor([1])))
  for t1 in l:
    for t2 in l:
      if t1 != t2:
        generate_sample(dataset, config, t1, t2, dataloader, sde)


if __name__ == "__main__":
  app.run(main)