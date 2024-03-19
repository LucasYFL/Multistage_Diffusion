from dataset import get_dataset
import sde_lib 

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

import torch
import logging
import os

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.mark_flags_as_required(["config"])

def normal_distribution(x, y_batch, s, std, bias = 0):
  bs = y_batch.shape[0]
  prob = torch.exp(-(((x - s * y_batch)**2).view(bs, -1).sum(dim=1).to(torch.float64)/std**2)/2 - bias)
  prob_y = prob.clone().view(-1, 1, 1, 1) * y_batch
  return prob.sum(dim=0, keepdim=True), prob_y.sum(dim=0, keepdim=True)

def get_exp_bias(x, y_batch, std):
  ## because exp() might return a very small number, we need a bias
  bs = y_batch.shape[0]
  return (-(((x - y_batch)**2).view(bs, -1).sum(dim=1)/std**2)/2).max()

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
                                           num_workers=8)
  optimal_solutions = []
  optimal_solution0s = []
  optimal_solution1s = []
  ts = []
  ss = []
  stds = []
  for i in range(config.exp.sampling_num):
    randn_idx = torch.randint(len(dataset), (1, ))
    batch = dataset[randn_idx][0]
    t = torch.rand((1, )) * (sde.T - eps) + eps
    s, sigma = sde.transform_prob(t)
    z = torch.randn_like(batch)
    x = s * batch + s * sigma[:, None, None, None] * z
    std = s * sigma
    prob_sum = 0.
    prob_y_sum = torch.zeros_like(x)
    y_sum = torch.zeros_like(x)
    exp_bias = -(torch.inf)
    for y_batch, _ in dataloader:
      exp_bias = max(exp_bias, get_exp_bias(x, y_batch, std))
    for y_batch, _ in dataloader:
      prob, prob_y = normal_distribution(x, y_batch, s, std, exp_bias)
      prob_sum += prob
      prob_y_sum += prob_y
      y_sum += y_batch.sum(dim=0, keepdim=True)
    assert config.exp.loss_func in ["epsilon", "x0"]
    if config.exp.loss_func == "x0":
      optimal_solution = prob_y_sum/prob_sum
    elif config.exp.loss_func == "epsilon":
      optimal_solution = x/std - prob_y_sum/(prob_sum*sigma)
    y_mean = y_sum/len(dataset)
    image_shape = z.shape
    optimal_solution_0 = z.view(1, *image_shape)
    optimal_solution_1 = (x/std - y_mean/sigma).view(1, *image_shape)
    
    optimal_solutions.append(optimal_solution)
    optimal_solution0s.append(optimal_solution_0)
    optimal_solution1s.append(optimal_solution_1)
    ts.append(t.view(1, -1))
    ss.append(s.view(1, -1))
    stds.append(std.view(1, -1))
    logging.info(f"The {config.host_id} host finished {i + 1}th sampling generation")  
    if (i + 1) % config.exp.num_save == 0:
      if not os.path.isdir(config.exp.save_dir):
        os.mkdir(config.exp.save_dir)
      host_pkg = os.path.join(config.exp.save_dir, str(config.host_id))
      if not os.path.isdir(host_pkg):
        os.mkdir(host_pkg)
      torch.save({
            'optimal_solutions': torch.concat(tuple(optimal_solutions)),
            'optimal_solution0s': torch.concat(tuple(optimal_solution0s)),
            'optimal_solution1s': torch.concat(tuple(optimal_solution1s)),
            'ts': torch.concat(tuple(ts)),
            'ss': torch.concat(tuple(ss)),
            'stds': torch.concat(tuple(stds)),
            }, os.path.join(host_pkg, f"{i//config.exp.num_save}.pth"))
      optimal_solutions = []
      optimal_solution0s = []
      optimal_solution1s = []
      ts = []
      ss = []
      stds = []
      # 


if __name__ == "__main__":
  app.run(main)