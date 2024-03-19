#!/bin/bash
# The interpreter used to execute the script
#“#SBATCH” directives that convey submission options:
#SBATCH --job-name=interval_split_exp
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --time=60:00:00
#SBATCH --account=qingqu1
#SBATCH --partition=standard
#SBATCH --output=/scratch/qingqu_root/qingqu1/huijiezh/dpm-solver/interval_split/out/1.out
#SBATCH --cpus-per-task=4

module purge
module load cuda/11.7.1 cudnn/11.7-v8.7.0
eval "$(conda shell.bash hook)"
conda activate score_sde_pytorch
cd /scratch/qingqu_root/qingqu1/huijiezh/dpm-solver/interval_split
python interval_split.py --config ./configs/cifar10_vp_config.py --config.host_id 1
