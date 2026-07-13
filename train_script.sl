#!/bin/bash

# GPUs architecture and number
# ----------------------------
# Partition (submission class)
#SBATCH --partition gpu_all

# GPUs per compute node
#   a100:8 (maximum) for gpu
#   a100:8 (maximum) for hpda
#   h200:8 (maximum) for gpu_h200
#SBATCH --gres=gpu:a100:1


# ----------------------------
# processes / tasks
#SBATCH -n 1

# ----------------------------
# CPUs per task
# Set the number of cpu in proportion to the number of GPU's devices :
#   gpu: until 16 cores / device
#   hpda: until 16 cores / device
#   gpu_h200: until 24 cores / device
#SBATCH --cpus-per-task 16

# ------------------------
# Job time (hh:mm:ss)
#SBATCH --time 05:00:00
# ------------------------

#SBATCH --output=/home/2017018/jaffol01/lung_map/slurm_out/slurm-%j.out


# environments
# ---------------------------------

module purge
module load aidl/pytorch/2.2.0-cuda12.1
source .venv/bin/activate

#export WANDB_MODE=offline
export HF_HOME="/home/2017018/jaffol01/lung_map/src/segmentation/models/pretrain_models/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONPATH="/home/2017018/jaffol01/lung_map:$PYTHONPATH"


# script execution
# ---------------------------------

srun python src/segmentation/models/dev/train_lightning.py --save_dir /dlocal/run/$SLURM_JOB_ID
