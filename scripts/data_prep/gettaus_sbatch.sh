#!/bin/bash

#SBATCH --job-name=gettaus
#SBATCH --output=logs/gettaus_%A_%a.out
#SBATCH --error=logs/gettaus_%A_%a.err
#SBATCH --account=ucb403_alpine2
#SBATCH --partition=amilan
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --array=0-1
#SBATCH --mail-type=END
#SBATCH --mail-user=julia.lober@colorado.edu

ml purge
ml anaconda
conda activate coherence

# Define arguments for each flight
FLIGHT_IDS=("23205" "05208")
COH_DIR=("/projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/coherences")
DAY_FILES=("/projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/lowman_23205_day_intervals.txt"
           "/projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/lowman_05208_day_intervals.txt")
OUT_FPS=("/projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/taus/lowman_23205_taus_2020_2021.tif"
         "/projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/taus/lowman_05208_taus_2020_2021.tif")

COH_FILES=(${COH_DIR}/*${FLIGHT_IDS[$SLURM_ARRAY_TASK_ID]}*.tif)
DAY_FILE=${DAY_FILES[$SLURM_ARRAY_TASK_ID]}
OUT_FP=${OUT_FPS[$SLURM_ARRAY_TASK_ID]}

python gettaus_batch.py \
    --coh_tifs "${COH_FILES[@]}"  \
    --days $DAY_FILE \
    --out_fp $OUT_FP

conda deactivate
ml purge
