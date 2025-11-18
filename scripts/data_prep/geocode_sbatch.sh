#!/bin/bash

#SBATCH --job-name=geocode
#SBATCH --output=geocode_%A_%a.out
#SBATCH --array=0-22
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00


SLC_LIST=`ls /projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/slcs_raw/05208*.slc`
ANN_LIST=`ls /projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/slcs_raw/05208*.ann`

# SLC_LIST=(/path/to/slc1 /path/to/slc2 ...)  # Fill in your SLC file paths
# ANN_LIST=(/path/to/ann1 /path/to/ann2 ...)
LLH_FP=/projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/slcs_raw/
OUT_DIR=/projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/slcs_geocoded/
WEIGHTS_FP=/projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/lowman_05208_weights.npz

python geocode_batch.py \
    --slc_list ${SLC_LIST[$SLURM_ARRAY_TASK_ID]} \
    --ann_list ${ANN_LIST[$SLURM_ARRAY_TASK_ID]} \
    --llh_fp $LLH_FP \
    --out_dir $OUT_DIR \
    --weights_fp $WEIGHTS_FP \
    --n_jobs 1