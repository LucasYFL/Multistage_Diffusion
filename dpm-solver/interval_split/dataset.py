import torchvision
import torchvision.transforms as TF
import torch
import os


def get_dataset(config):
  """Create data loaders for training and evaluation.
  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.
  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  # Compute batch size for this worker.
  batch_size = config.data.batch_size
  root = config.data.dataset_root
  if not os.path.isdir(root):
    os.mkdir(root)
  # Create dataset builders for each dataset.
  if config.data.dataset == 'CIFAR10':
    if not os.path.isdir(os.path.join(root, config.data.dataset)):
      os.mkdir(os.path.join(root, config.data.dataset))
    download = 'cifar-10-batches-py' not in os.listdir(os.path.join(root, config.data.dataset))
    
    train_transforms = [
      TF.PILToTensor(),
      TF.ConvertImageDtype(torch.float),
      TF.Resize((config.data.image_size, config.data.image_size)),
    ]
    eval_trainforms = train_transforms.copy()
    # if config.data.random_flip:
    #   train_transforms.append(TF.RandomHorizontalFlip())
    train_transforms = TF.Compose(train_transforms)
    eval_trainforms = TF.Compose(eval_trainforms)
    train_ds = torchvision.datasets.CIFAR10(root = os.path.join(root, config.data.dataset),
                                            transform=train_transforms, 
                                            train = True, 
                                            download = download)
    eval_ds = torchvision.datasets.CIFAR10(root = os.path.join(root, config.data.dataset), 
                                           transform=eval_trainforms, 
                                           train = False, 
                                           download = download)

  return train_ds, eval_ds

def distributed_dataset(ds, config, drop_last = False):
  sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle = True)
  
  batch_sampler = torch.utils.data.BatchSampler(sampler, config.training.batch_size, drop_last = drop_last)

  dataloader = torch.utils.data.DataLoader(ds, batch_sampler=batch_sampler, pin_memory = True, num_workers=1)
  dataset_iter = iter(dataloader)
  
  return dataset_iter