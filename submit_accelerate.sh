#!/bin/bash
#SBATCH --job-name=timebrains_v15
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --output=./log/accelerate_%j.out

module load PrgEnv-gnu
module load gcc-native/13.2
module load cuda/12.6
module load cudatoolkit/24.11_12.6
module load cray-python/3.11.7
module load brics/aws-ofi-nccl/1.8.1

source /home/u6dm/jindong.u6dm/my_torch_env/bin/activate
# pip install monai scikit-image

cd /home/u6dm/jindong.u6dm/Code/TimeBrainsGen
mkdir -p log checkpoints

echo "Starting Accelerate Multi-GPU Training (v15, 200 epochs, ROI+FocalLoss)..."
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes 4 --num_machines 1 train.py --epochs 200
echo "Training Script Completed."
