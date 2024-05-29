# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training NCSNv3 on CIFAR-10 with continuous sigmas."""

from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = True
  training.reduce_mean = True
  training.fewer_step = False
  # sampling
  sampling = config.sampling
  sampling.method = 'dpm_solver'
  sampling.dpm_solver_method = "singlestep"
  sampling.steps = 20
  sampling.dpm_solver_order = 3
  sampling.algorithm_type = "dpmsolver"
  sampling.skip_type = "time_uniform"
  sampling.noise_removal = False
  sampling.eps=1e-3
  sampling.thresholding = False
  sampling.rtol = 0.05
  # data
  data = config.data
  data.centered = True

  # model
  model = config.model
  model.name = 'DiT'
  model.input_size=32
  model.patch_size=4
  model.in_channels=3
  model.hidden_size=768
  model.depth=12
  model.num_heads=12
  model.mlp_ratio=4.0
  model.class_dropout_prob=0.1
  model.num_classes=1000
  model.learn_sigma=False
  model.ema_rate = 0.9999
  model.treat_t_as_y = False
  return config

