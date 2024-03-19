module purge
module load cuda/11.7.1 cudnn/11.7-v8.7.0
eval "$(conda shell.bash hook)"
conda activate score_sde_pytorch
cd /scratch/qingqu_root/qingqu1/huijiezh/dpm-solver/example_v2/score_sde_pytorch
torchrun --nproc_per_node=1 --master_port=29600 main.py --config ./configs/vp/cifar10_ncsnpp_multistage_deep_continuous_test.py --mode train --workdir /scratch/qingqu_root/qingqu1/shared_data/dpm_experiments/test_multistage --config.training.batch_size 32

