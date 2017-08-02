#!/bin/bash

# Slurm submission script, 
# GPU job 
# CRIHAN v 1.00 - Jan 2017 
# support@criann.fr

# Not shared resources
#SBATCH --share

# Job name
#SBATCH -J "lenet"

# Batch output file
#SBATCH --output ./outputjobs/lenet.o%J

# Batch error file
#SBATCH --error ./outputjobs/lenet.e%J

# GPUs architecture and number
# ----------------------------
# Partition (submission class)
#SBATCH --partition gpu_k80 

# GPUs per compute node
#   gpu:4 (maximum) for gpu_k80 
#   gpu:2 (maximum) for gpu_p100 
#SBATCH --gres gpu:1
# ----------------------------

# Job time (hh:mm:ss)
#SBATCH --time 24:00:00

# MPI task maximum memory (MB)
#SBATCH --mem-per-cpu 3000 
# ----------------------------

#SBATCH --mail-type ALL
# User e-mail address
#SBATCH --mail-user soufiane.belharbi@insa-rouen.fr

# environments
# ---------------------------------
module load cuda/8.0
module load python/2.7.12
# ---------------------------------

cd $LOCAL_WORK_DIR/workspace/code/class-invariance-hint/

# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python trainLenet.py lenet_0_1000_3_0_0_1_0_0_True_False_False_False_False.yaml 

