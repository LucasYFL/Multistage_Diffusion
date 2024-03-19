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

"""Training and evaluation"""
from models import ddpm, ncsnv2, ncsnpp, ncsnpp_multistage, UViT
from utils import save_checkpoint, restore_checkpoint
from torchvision.utils import make_grid, save_image
from torch.utils import tensorboard
import torch
import sde_lib
import likelihood
import evaluation
import datasets
from models.ema import ExponentialMovingAverage
from models import utils as mutils
import sampling
import losses
import logging
import tensorflow_gan as tfgan
import numpy as np
import time
import io
import gc
import tensorflow as tf
from ml_collections.config_flags import config_flags
from absl import flags
from absl import app
import os

local_rank = int(os.environ["LOCAL_RANK"])
total_rank = int(os.environ['LOCAL_WORLD_SIZE'])


# Keep the import below for registering all model definitions
FLAGS = flags.FLAGS


config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
config_flags.DEFINE_config_file(
    "config1", None, "Training configuration.", lock_config=True)
config_flags.DEFINE_config_file(
    "config2", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string("m1", None, "Model 1 directory.")
flags.DEFINE_string("m2", None, "Model 2  directory.")
flags.DEFINE_string("m3", None, "Model 2  directory.")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config", 'm1'])
tf.config.experimental.set_visible_devices([], "GPU")
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
local_rank = int(os.environ["LOCAL_RANK"])
total_rank = int(os.environ['LOCAL_WORLD_SIZE'])
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend='nccl')


def evaluate(config,
             workdir, m1, m2=None, m3=None,config1=None,config2=None,
             eval_folder="eval"):
    """Evaluate trained models.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints.
      eval_folder: The subfolder for storing evaluation results. Default to
        "eval".
    """
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    os.makedirs(eval_dir, exist_ok=True)

    # Build data pipeline
    train_ds, _ = datasets.get_dataset(config)

    train_iter = datasets.distributed_dataset(train_ds, config)

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Use inceptionV3 for images with resolution higher than 256.
    # inceptionv3 = config.data.image_size >= 256
    # inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)
    # Initialize model
    score_models = []
    optimizers = []
    emas = []
    states = []
    mdir = (m1, m2,m3)
    checkpoint_dirs = []
    logging.info(config.eval.t_tuples)
    configs = (config,)*3 if config1 is None else (config, config1,config2)

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min,
                            beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min,
                               beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min,
                            sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    temp_model = [mutils.create_model(con, local_rank) for con in configs if con]
    for i in range(len(config.eval.t_tuples)+1):
        # conv = config.eval.t_converge[i]
        conv = i
        s = temp_model[conv]
        score_models.append(s)
        opt = losses.get_optimizer(config, s.parameters())
        optimizers.append(opt)
        ema = ExponentialMovingAverage(
            s.parameters(), decay=config.model.ema_rate)
        emas.append(ema)
        states.append(dict(optimizer=opt, model=s, ema=ema, step=0))
        checkpoint_dirs.append(os.path.join(
            workdir, mdir[conv], "checkpoints"))

    # Build the sampling function when sampling is enabled
    if config.eval.enable_sampling:
        sampling_shape = (config.eval.batch_size // total_rank,
                          config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(
            config, sde, sampling_shape, inverse_scaler, sampling_eps, local_rank)

    begin_ckpt = config.eval.begin_ckpt
    if local_rank == 0:
        logging.info("begin checkpoint: %d" % (begin_ckpt,))
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1, config.eval.ckpt_freq):
        for i in range(len(checkpoint_dirs)):
            if config.eval.t_converge[i]:
                logging.info("{} is converged model".format(i))
                ckpt_path = os.path.join(
                    checkpoint_dirs[i], "checkpoint_{}.pth".format(config.eval.converge_epoch))
            else:
                ckpt_path = os.path.join(
                    checkpoint_dirs[i], "checkpoint_{}.pth".format(ckpt))
            logging.info(ckpt_path)
            states[i] = restore_checkpoint(
                ckpt_path, states[i], device=f"{config.device}:{local_rank}")
            emas[i].copy_to(score_models[i].parameters())

        # Generate samples and compute IS/FID/KID when enabled
        if config.eval.enable_sampling:

            # inceptionv3 = FID_score.get_inceptionV3(device = f"{config.device}:{local_rank}", dims = 2048)

            if local_rank == 0:
                logging.info(eval_dir)
            num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
            for r in range(num_sampling_rounds):
                if local_rank == 0:
                    logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))

                # Directory to save samples. Different for each host to avoid writing conflicts
                this_sample_dir = os.path.join(
                    eval_dir, f"ckpt_{ckpt}_host_{local_rank}")
                os.makedirs(this_sample_dir, exist_ok=True)
                samples_raw, n = sampling_fn(
                    score_models, config.eval.t_tuples)

                samples = torch.clip(samples_raw.permute(
                    0, 2, 3, 1) * 255., 0, 255).to(torch.uint8)
                # center the sample when calculating fid
                samples_fid = (torch.clone(samples).permute(
                    0, 3, 1, 2).to(torch.float32) / 255) * 2 - 1
                samples = samples.reshape(
                    (-1, config.data.image_size, config.data.image_size, config.data.num_channels)).cpu().numpy()
                # Write samples to disk or Google Cloud Storage
                with open(
                        os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, samples=samples)
                    fout.write(io_buffer.getvalue())

                if r == 0:
                    nrow = int(np.sqrt(samples_raw.shape[0]))
                    image_grid = make_grid(samples_raw, nrow, padding=2)
                    with open(
                            os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
                        save_image(image_grid, fout)


def main(argv):
    global config_fewer
    config_fewer = FLAGS.config
    config1 = FLAGS.config1
    # Run the evaluation pipeline
    evaluate(FLAGS.config, FLAGS.workdir, FLAGS.m1,
             FLAGS.m2, FLAGS.m3,FLAGS.config1, FLAGS.config2,FLAGS.eval_folder)


if __name__ == "__main__":
    app.run(main)
