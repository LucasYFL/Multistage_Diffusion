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

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time

import numpy as np
# import tensorflow as tf
# import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp, ncsnpp_multistage, UViT
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
import pytorch_fid.fid_score as FID_score
import glob

FLAGS = flags.FLAGS
local_rank = int(os.environ["LOCAL_RANK"])
total_rank = int(os.environ['LOCAL_WORLD_SIZE'])


def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    tb_dir = os.path.join(workdir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    score_model = mutils.create_model(config, local_rank)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(
        workdir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Build data iterators
    train_ds, eval_ds = datasets.get_dataset(config)

    train_iter = datasets.distributed_dataset(train_ds, config)
    eval_iter = datasets.distributed_dataset(eval_ds, config)

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

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

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting, t1=float(config.training.t0), t2=float(config.training.t1),
                                       cond=config.model.num_classes > 0)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting, t1=float(config.training.t0), t2=float(config.training.t1),
                                      cond=config.model.num_classes > 0)
    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (config.training.batch_size, config.data.num_channels,
                          config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(
            config, sde, sampling_shape, inverse_scaler, sampling_eps, local_rank)

    num_train_steps = config.training.n_iters

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))

    for step in range(initial_step, num_train_steps + 1):
        try:
            batch, labels = next(train_iter)
        except StopIteration:
            train_iter = datasets.distributed_dataset(train_ds, config)
            batch, labels = next(train_iter)
        if config.device == torch.device('cuda'):
            batch = batch.to(f"{config.device}:{local_rank}")
            labels = labels.to(f"{config.device}:{local_rank}")
        else:
            batch = batch.to(config.device)
            labels = labels.to(config.device)
        batch = scaler(batch)
        # Execute one training step
        loss = train_step_fn(state, batch, labels)
        if step % config.training.log_freq == 0 and local_rank == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
            writer.add_scalar("training_loss", loss, step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0 and local_rank == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            try:
                batch, labels = next(eval_iter)
            except StopIteration:
                eval_iter = datasets.distributed_dataset(eval_ds, config)
                batch, labels = next(eval_iter)
            if config.device == torch.device('cuda'):
                eval_batch = batch.to(f"{config.device}:{local_rank}")
                labels = labels.to(f"{config.device}:{local_rank}")
            else:
                eval_batch = batch.to(config.device)
                labels = labels.to(config.device)
            eval_batch = scaler(eval_batch)
            eval_loss = eval_step_fn(state, eval_batch,labels)
            if local_rank == 0:
                logging.info("step: %d, eval_loss: %.5e" %
                             (step, eval_loss.item()))
                writer.add_scalar("eval_loss", eval_loss.item(), step)

        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and (step % config.training.snapshot_freq == 0 or step == num_train_steps) and local_rank == 0:
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(
                checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

            # Generate and save samples
            if config.training.snapshot_sampling:
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                sample, n = sampling_fn(score_model)
                ema.restore(score_model.parameters())
                this_sample_dir = os.path.join(
                    sample_dir, "iter_{}".format(step))
                os.makedirs(this_sample_dir, exist_ok=True)
                nrow = int(np.sqrt(sample.shape[0]))
                image_grid = make_grid(sample, nrow, padding=2)
                sample = np.clip(sample.permute(0, 2, 3, 1).cpu(
                ).numpy() * 255, 0, 255).astype(np.uint8)
                with open(
                        os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
                    np.save(fout, sample)

                with open(
                        os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
                    save_image(image_grid, fout)


def evaluate(config,
             workdir,
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
    score_model = mutils.create_model(config, local_rank)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    checkpoint_dir = os.path.join(workdir, "checkpoints")

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
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        # Wait if the target checkpoint doesn't exist yet
        waiting_message_printed = False
        ckpt_filename = os.path.join(
            checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        while not os.path.exists(ckpt_filename):
            if not waiting_message_printed:
                logging.warning(
                    "Waiting for the arrival of checkpoint_%d" % (ckpt,))
                waiting_message_printed = True
            time.sleep(60)

        # Wait for 2 additional mins in case the file exists but is not ready for reading
        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
        try:
            state = restore_checkpoint(
                ckpt_path, state, device=f"{config.device}:{local_rank}")
        except:
            time.sleep(60)
            try:
                state = restore_checkpoint(
                    ckpt_path, state, device=f"{config.device}:{local_rank}")
            except:
                time.sleep(120)
                state = restore_checkpoint(
                    ckpt_path, state, device=f"{config.device}:{local_rank}")
        ema.copy_to(score_model.parameters())

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
                samples_raw, n = sampling_fn(score_model)

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

                # Force garbage collection before calling TensorFlow code for Inception network

            #   gc.collect()

            #   act = FID_score.get_activations(samples_fid, inceptionv3)
            #   # Force garbage collection again before returning to JAX code
            #   gc.collect()
            #   # Save latent represents of the Inception network to disk or Google Cloud Storage
            #   with open(
            #       os.path.join(this_sample_dir, f"statistics_{r}.npz"), "wb") as fout:
            #     io_buffer = io.BytesIO()
            #     np.savez_compressed(
            #       io_buffer, stats=act)
            #     fout.write(io_buffer.getvalue())

            # # Calculate dataset statistic if needed
            # try:
            #   stats_gt = evaluation.load_dataset_stats(config)['stats_gt']
            # except:
            #   stats_gt_rank = []
            #   while True:
            #     try:
            #       batch, _ = next(train_iter)
            #       # centered to [-1, 1]
            #       batch = batch * 2 - 1
            #       if config.device == torch.device('cuda'):
            #         batch = batch.to(f"{config.device}:{local_rank}")
            #       else:
            #         batch = batch.to(config.device)
            #       act = FID_score.get_activations(batch, inceptionv3)
            #       stats_gt_rank.append(act)
            #       logging.info(f"Generating the ground truth statistics in {len(stats_gt_rank)} iterations.")
            #     except StopIteration:
            #       stats_gt_rank = np.concatenate(stats_gt_rank, axis=0)
            #       if not os.path.isdir('assets'):
            #         os.mkdir('assets')
            #       if not os.path.isdir('assets/stats/'):
            #         os.mkdir('assets/stats/')
            #       np.savez(f'assets/stats/{config.data.dataset.lower()}_stats_{local_rank}.npz', stats_gt=stats_gt_rank)
            #       break
            #   if local_rank == 0:
            #     stats_gt = []
            #     files = glob.glob(os.path.join('assets/stats/', f'{config.data.dataset.lower()}_stats_*.npz'))
            #     for f in files:
            #       stats_gt.append(np.load(f)['stats_gt'])
            #     stats_gt = np.concatenate(stats_gt, axis=0)
            #     np.savez(f'assets/stats/{config.data.dataset.lower()}_stats.npz', stats_gt=stats_gt)

            # # Compute inception scores, FIDs and KIDs.
            # # Load all statistics that have been previously computed and saved for each host
            # if local_rank == 0:
            #   all_stats = []

            #   for rank_id in range(total_rank):
            #     this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}_host_{rank_id}")
            #     stats = glob.glob(os.path.join(this_sample_dir, "statistics_*.npz"))
            #     wait_message = False
            #     while len(stats) < num_sampling_rounds:
            #       if not wait_message:
            #         logging.warning(f"Waiting for statistics on host {rank_id}")
            #         wait_message = True
            #       stats = glob.glob(
            #         os.path.join(this_sample_dir, "statistics_*.npz"))
            #       time.sleep(30)

            #     for stat_file in stats:
            #       with open(stat_file, "rb") as fin:
            #         stat = np.load(fin)
            #         all_stats.append(stat["stats"])

            #   all_stats = np.concatenate(all_stats, axis=0)[:config.eval.num_samples]

            #   mu_gt = np.mean(stats_gt, axis=0)
            #   sigma_gt = np.cov(stats_gt, rowvar=False)
            #   mu_pred = np.mean(all_stats, axis=0)
            #   sigma_pred = np.cov(all_stats, rowvar=False)
            #   fid = FID_score.calculate_frechet_distance(
            #         mu_gt, sigma_gt, mu_pred, sigma_pred
            #         )

            #   logging.info(
            #     "ckpt-%d --- FID: %.6e" % (
            #       ckpt, fid))

            #   with open(os.path.join(eval_dir, f"report_{ckpt}.npz"),
            #                         "wb") as f:
            #     io_buffer = io.BytesIO()
            #     np.savez_compressed(io_buffer, fid=fid)
            #     f.write(io_buffer.getvalue())
