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
"""Return training and evaluation/test datasets from config files."""
# import jax
# import tensorflow as tf
import torchvision
import torchvision.transforms as TF
import torch
import os


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered or config.data.normalize:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x


# def crop_resize(image, resolution):
#   """Crop and resize an image to the given resolution."""
#   crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
#   h, w = tf.shape(image)[0], tf.shape(image)[1]
#   image = image[(h - crop) // 2:(h + crop) // 2,
#           (w - crop) // 2:(w + crop) // 2]
#   image = tf.image.resize(
#     image,
#     size=(resolution, resolution),
#     antialias=True,
#     method=tf.image.ResizeMethod.BICUBIC)
#   return tf.cast(image, tf.uint8)


# def resize_small(image, resolution):
#   """Shrink an image to the given resolution."""
#   h, w = image.shape[0], image.shape[1]
#   ratio = resolution / min(h, w)
#   h = tf.round(h * ratio, tf.int32)
#   w = tf.round(w * ratio, tf.int32)
#   return tf.image.resize(image, [h, w], antialias=True)


# def central_crop(image, size):
#   """Crop the center of an image to the given size."""
#   top = (image.shape[0] - size) // 2
#   left = (image.shape[1] - size) // 2
#   return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_dataset(config, uniform_dequantization=False, evaluation=False):
    """Create data loaders for training and evaluation.

    Args:
      config: A ml_collection.ConfigDict parsed from config files.
      uniform_dequantization: If `True`, add uniform dequantization to images.
      evaluation: If `True`, fix number of epochs to 1.

    Returns:
      train_ds, eval_ds, dataset_builder.
    """
    # Compute batch size for this worker.
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    # if batch_size % jax.device_count() != 0:
    #   raise ValueError(f'Batch sizes ({batch_size} must be divided by'
    #                    f'the number of devices ({jax.device_count()})')

    # Reduce this when image resolution is too large and data pointer is stored
    # shuffle_buffer_size = 10000
    # prefetch_size = tf.data.experimental.AUTOTUNE
    # num_epochs = None if not evaluation else 1
    root = "./dataset"
    if not os.path.isdir(root):
        os.mkdir(root)
    # Create dataset builders for each dataset.
    if config.data.dataset == 'CIFAR10':
        if not os.path.isdir(os.path.join(root, config.data.dataset)):
            os.mkdir(os.path.join(root, config.data.dataset))
        download = 'cifar-10-batches-py' not in os.listdir(
            os.path.join(root, config.data.dataset))

        train_transforms = [
            TF.PILToTensor(),
            TF.ConvertImageDtype(torch.float),
            TF.Resize((config.data.image_size, config.data.image_size)),
            # TF.Normalize(0.5, 0.5)
        ]
        if config.data.normalize:
            train_transforms.append(TF.Normalize(0.5, 0.5))
        eval_trainforms = train_transforms.copy()
        if config.data.random_flip:
            train_transforms.append(TF.RandomHorizontalFlip())
        train_transforms = TF.Compose(train_transforms)
        eval_trainforms = TF.Compose(eval_trainforms)
        train_ds = torchvision.datasets.CIFAR10(root=os.path.join(root, config.data.dataset),
                                                transform=train_transforms,
                                                train=True,
                                                download=download)
        eval_ds = torchvision.datasets.CIFAR10(root=os.path.join(root, config.data.dataset),
                                               transform=eval_trainforms,
                                               train=False,
                                               download=download)

    #   def resize_op(img):
    #     img = tf.image.convert_image_dtype(img, tf.float32)
    #     return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

    # def preprocess_fn(d):
    #   ## TODO change to pytorch
    #   """Basic preprocessing function scales data to [0, 1) and randomly flips."""
    #   img = resize_op(d['image'])
    #   if config.data.random_flip and not evaluation:
    #     img = tf.image.random_flip_left_right(img)
    #   if uniform_dequantization:
    #     img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.

    #   return dict(image=img, label=d.get('label', None))

    return train_ds, eval_ds


def distributed_dataset(ds, config, drop_last=False):
    sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True)

    batch_sampler = torch.utils.data.BatchSampler(
        sampler, config.training.batch_size, drop_last=drop_last)

    dataloader = torch.utils.data.DataLoader(
        ds, batch_sampler=batch_sampler, pin_memory=True, num_workers=1)
    dataset_iter = iter(dataloader)

    return dataset_iter
