#!/bin/bash
#SBATCH --job-name=eval_v15_ep200
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --output=./log/evaluate_v2_%j.out

module load PrgEnv-gnu
module load gcc-native/13.2
module load cuda/12.6
module load cudatoolkit/24.11_12.6
module load cray-python/3.11.7
module load brics/aws-ofi-nccl/1.8.1

source /home/u6dm/jindong.u6dm/my_torch_env/bin/activate


cd /home/u6dm/jindong.u6dm/Code/TimeBrainsGen
mkdir -p log predictions

echo "Starting Epoch 200 Evaluation (v15, 5 samples)..."
python evaluate.py --epoch 200 --num_samples 5
echo "Evaluation Script Completed."
