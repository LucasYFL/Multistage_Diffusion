# training
torchrun --nproc_per_node=2 --master_port=29502 main.py --config ./configs/vp/cifar10_ncsnpp_multistage_deep_continuous_v2.py --workdir exp/cifar_multistage --mode train --config.training.batch_size=128

# eval

# 1. sampling
torchrun --nproc_per_node=1 --master_port=29600 main_interval.py --config "configs/vp/cifar10_ncsnpp_multistage_deep_continuous_v2.py" --m1 cifar_multistage --workdir exp --eval_folder cifar_multistage/eval --config.eval.t_tuples="()" --config.eval.t_converge="(0,)" --config.eval.begin_ckpt=1 --config.eval.end_ckpt=12 --config.eval.batch_size=1024 --config.sampling.steps=20  --config.sampling.eps=1e-4 
# 2. calculate fid from samples
python evaluation_fromsample.py --config "configs/vp/cifar10_ncsnpp_multistage_deep_continuous_v2.py" --workdir exp --eval_folder cifar_multistage/eval  --config.eval.begin_ckpt=1 --config.eval.end_ckpt=12 --config.eval.batch_size=1024

