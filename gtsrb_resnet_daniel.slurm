#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --partition=gpuv100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniel-alfonso.montoyavasquez@cea.fr

echo "Starting commands!"
pwd;hostname
printf "\n"
printf "load modules: "
module load anaconda
module load cuda/11.6
printf "\n"
printf "Check loaded modules: "
module list


# With Slurm jobs, your ~/.bashrc is not sourced, so it doesn’t initialize the shell for Conda environments
eval "$(conda shell.bash hook)"
conda activate gtsrb_pip_env

echo "list conda-envs explicit paths:"
conda info --envs
printf "\n"
ls /home/users/dmontoya/.conda/envs
printf "\n"

printf "\n"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
printf "\n"

python --version
printf "\n"
# nvidia-smi
# printf "\n"
printf "Start training:"
srun --kill-on-bad-exit=1 python train_gtsrb_classifier.py trainer.epochs=500 datamodule.batch_size=64 slurm=True trainer.gpus=2 model.lr=1e-4

# dmesg -T
printf "\n"
echo "Done!"
