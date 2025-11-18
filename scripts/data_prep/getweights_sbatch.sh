#!/bin/bash

#SBATCH --job-name=getweights
#SBATCH --output=getweights_%A_%a.out
#SBATCH --array=0-22
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

FLIGHT_NUMS=05208
LLH_DIR=/projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/slcs_raw/
SLC_DIR=/projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/slcs_raw/
ANN_DIR=/projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/slcs_raw/
OUT_FP=/projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/

python getweights_batch.py \
  --flight_nums 05208 23205 \
  --llh_dir ${LLH_FP} \
  --ann_dir ${ANN} \
  --out_dir ${OUT_FP}