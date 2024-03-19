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
import os

local_rank = int(os.environ["LOCAL_RANK"])
total_rank = int(os.environ['LOCAL_WORLD_SIZE'])

import json
import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import torch
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend='nccl')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
FLAGS = flags.FLAGS


config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
  if FLAGS.mode == "train":
    # Create the working directory
    # Set logger so that it outputs to both console and file
    # Make logging work for both disk and Google Cloud Storage
    
    FLAGS.config.training.batch_size = int(FLAGS.config.training.batch_size / total_rank)
    # print(FLAGS.config.training.batch_size)
    if local_rank == 0:
      os.makedirs(FLAGS.workdir, exist_ok=True)
      gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
      handler = logging.StreamHandler(gfile_stream)
      formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
      handler.setFormatter(formatter)
      logger = logging.getLogger()
      logger.addHandler(handler)
      logger.setLevel('INFO')
      out_file = open(os.path.join(FLAGS.workdir,"train_config.json"), "w")
  
      json.dump(FLAGS.config.to_json_best_effort(), out_file, indent = 6,separators=(',\n', ': '))
        
      out_file.close()
      logger.info(FLAGS.config)
    # Run the training pipeline
    run_lib.train(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == "eval":
    # Run the evaluation pipeline
    run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
