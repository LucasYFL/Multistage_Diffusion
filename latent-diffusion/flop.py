from torchstat import stat
import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange

from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    # model.load_state_dict(sd,strict=False)
    # model.cuda()
    model.eval()
    return model

def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step

parser = get_parser()
opt, unknown = parser.parse_known_args()
ckpt = None

if not os.path.exists(opt.resume):
    raise ValueError("Cannot find {}".format(opt.resume))
if os.path.isfile(opt.resume):
    # paths = opt.resume.split("/")
    try:
        logdir = '/'.join(opt.resume.split('/')[:-1])
        # idx = len(paths)-paths[::-1].index("logs")+1
        print(f'Logdir is {logdir}')
    except ValueError:
        paths = opt.resume.split("/")
        idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
        logdir = "/".join(paths[:idx])
    ckpt = opt.resume
else:
    assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
    logdir = opt.resume.rstrip("/")
    ckpt = os.path.join(logdir, "model.ckpt")

base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
opt.base = base_configs

configs = [OmegaConf.load(cfg) for cfg in opt.base]
cli = OmegaConf.from_dotlist(unknown)
config = OmegaConf.merge(*configs, cli)
print(config)
model, global_step = load_model(config, None, False, True)
print(f"global step: {global_step}")
model = model.model
sampling_shape = (3,64,64)
model.to('cpu')
t = torch.tensor([999])
class mfn(torch.nn.Module):
    def __init__(self,m):
        super().__init__()
        self.m = m
    def forward(self,x):
        return self.m(x,t)
stat(mfn(model).to('cpu'),sampling_shape)
