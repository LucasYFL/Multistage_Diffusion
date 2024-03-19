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
import torch


def get_config():
    config = get_default_configs()
    # training
    training = config.training
    training.sde = 'vpsde'
    training.continuous = True
    training.reduce_mean = True
    training.fewer_step = False
    training.n_iters = 500000
    training.batch_size = 128
    # sampling
    sampling = config.sampling
    sampling.method = 'dpm_solver'
    sampling.dpm_solver_method = "singlestep"
    sampling.steps = 20
    sampling.dpm_solver_order = 3
    sampling.algorithm_type = "dpmsolver"
    sampling.skip_type = "time_uniform"
    sampling.noise_removal = False
    sampling.eps = 1e-3
    sampling.thresholding = False
    sampling.rtol = 0.05
    # data
    data = config.data
    data.centered = False
    data.normalize = True
    optim = config.optim
    optim.weight_decay = 0.03
    optim.beta1 = 0.99
    optim.warmup = 2500
    optim.optimizer = 'adamw'
    optim.grad_clip = -1
    # model
    model = config.model
    model.name = 'UViT_multimodel'
    model.embed_dim = 512
    model.num_classes = -1
    model.in_chans = 3
    model.patch_size = 2
    model.depth = 12
    model.num_heads = 8
    model.mlp_ratio = 4.0
    model.mlp_time_embed = False
    model.qkv_bias = False
    model.qk_scale = None
    model.use_checkpoint = False
    model.skip = True
    model.conv = True
    model.norm_layer = torch.nn.LayerNorm
    model.ema_rate = 0.9999
    model.de_nfs = [768, 512, 128]
    model.stage_num = 3
    model.stage_interval = [
        [[0, 0.4420]], [[0.4420, 0.6308]], [[0.6308, 1]]
    ]
    return config
