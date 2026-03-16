#!/bin/bash

#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=nova
#SBATCH --account=jannesar-lab
#SBATCH --job-name="ic"
#SBATCH --mail-user=msamani@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# === Step 1: Load Required Modules ===
module load python/3.10  # Or use your cluster's Python module version
# Or: module load anaconda  (if you're using conda)

# === Step 2: Activate Your Environment ===
# If using venv:
source /lustre/hdd/LAS/jannesar-lab/msamani/pythonenv/bin/activate

# If using conda:
# source activate your_env_name

# === Step 3: Add Your Module (optional) ===
# If your Python module is in a custom directory:
#export PYTHONPATH=$PYTHONPATH:/path/to/your/module
# === Step 4: Run Your Script ===
#python /lustre/hdd/LAS/jannesar-lab/msamani/OSF/scripts/random_subnet_generator_main.py
#python /lustre/hdd/LAS/jannesar-lab/msamani/OSF/scripts/train_img_classification.py
#python /lustre/hdd/LAS/jannesar-lab/msamani/SuperSAM/IC_nas.py
#python /lustre/hdd/LAS/jannesar-lab/msamani/SuperSAM/IC_nas_opentunner.py
#python /lustre/hdd/LAS/jannesar-lab/msamani/SuperSAM/generate_random_model_for_training.py
torchrun --nproc_per_node=1 --master_port=23355 train2.py     --model "output/pruned/deit_4.2G.pth"     --teacher-model regnety_160.deit_in1k     --epochs 300     --batch-size 64     --opt adamw     --lr 0.0005     --wd 0.05     --lr-scheduler cosineannealinglr     --lr-warmup-method linear     --lr-warmup-epochs 0     --lr-warmup-decay 0.033     --amp     --label-smoothing 0.1     --mixup-alpha 0.8     --auto-augment ra     --ra-sampler     --random-erase 0.25     --cutmix-alpha 1.0     --data-path "data/imagenet"     --output-dir "output/finetuned/deit_4.2G"     --interpolation bicubic
