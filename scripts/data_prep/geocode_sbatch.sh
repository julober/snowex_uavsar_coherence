#!/bin/bash

#SBATCH --job-name=geocode
#SBATCH --output=logs/geocode_%A_%a.out
#SBATCH --error=logs/geocode_%A_%a.err
#SBATCH --account=ucb403_alpine2     # Alpine allocation
#SBATCH --partition=amilan    # Alpine partition
#SBATCH --qos=long            # Alpine qos
#SBATCH --time=72:00:00           # Max wall time
#SBATCH --nodes=1          # Number of Nodes
#SBATCH --ntasks=16          # Number of tasks per job
#SBATCH --job-name=geocode     # Job submission name
#SBATCH --mail-type=END            # Email user when job finishes
#SBATCH --mail-user=julia.lober@colorado.edu # Email address of user

ml purge
ml anaconda
conda activate coherence

SLC_LIST=`ls /projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/raw/*23205*.slc`
ANN_LIST=`ls /projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/raw/*23205*.ann`

# SLC_LIST=(/path/to/slc1 /path/to/slc2 ...)  # Fill in your SLC file paths
# ANN_LIST=(/path/to/ann1 /path/to/ann2 ...)
LLH_FP=/projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/raw/lowman_23205_01_BC_s2_2x8.llh
OUT_DIR=/projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/geocoded/
WEIGHTS_FP=/projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/lowman_23205_weights.npz

python geocode_batch.py \
    --slc_list ${SLC_LIST[$SLURM_ARRAY_TASK_ID]} \
    --ann_list ${ANN_LIST[$SLURM_ARRAY_TASK_ID]} \
    --llh_fp $LLH_FP \
    --out_dir $OUT_DIR \
    --weights_fp $WEIGHTS_FP \
    --n_jobs 1

conda deactivate
ml purge
