from cluster import interval_cluster
from configs.vp import cifar10_ddpmpp_continuous as configs
from sde_lib import VESDE, VPSDE, subVPSDE
from dpm_solver import NoiseScheduleVP
import numpy as np
import torch as ch
import time
config=configs.get_config()
config.eval.batch_size = 16
sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
sampling_eps = 1e-3
ns = NoiseScheduleVP('linear', continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)
def snr_cost(arr):
    arr = ch.Tensor(arr)
    center = arr[(len(arr))//2]
    return ch.sum(ch.abs((ns.marginal_lambda(center)-ns.marginal_lambda(arr))*2))
timesteps =( np.arange(0,1000)+1) / 1000
print(timesteps)
start_time = time.time()
cl = interval_cluster(timesteps,3)
print(cl.calculate(snr_cost))
print(time.time()-start_time)