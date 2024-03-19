import ml_collections
import torch


def get_config():
  config = ml_collections.ConfigDict()

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'CIFAR10'
  data.image_size = 16
  data.batch_size = 1024
  data.random_flip = True
  data.centered = False
  data.uniform_dequantization = False
  data.num_channels = 3
  data.dataset_root = "./dataset/"
  
  # sde
  config.sde = sde = ml_collections.ConfigDict()
  sde.type = "VESDE"
  sde.sigma_min = 0.01
  sde.sigma_max = 50
  sde.num_scales = 1000
  sde.beta_min = 0.1
  sde.beta_max = 20

  # exp
  config.exp = exp = ml_collections.ConfigDict()
  exp.sampling_num = 5000
  exp.loss_func = "epsilon"
  exp.num_save = 2
  exp.save_dir = "./exp"
  
  #host
  config.host_id = 1
  
  return config