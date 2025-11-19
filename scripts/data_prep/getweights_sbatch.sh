#!/bin/bash

#SBATCH --job-name=getweights
#SBATCH --output=logs/getweights_%A_%a.out
#SBATCH --error=logs/getweights_%A_%a.err
#SBATCH --account=ucb403_alpine2     # Alpine allocation
#SBATCH --partition=amilan    # Alpine partition
#SBATCH --qos=long            # Alpine qos
#SBATCH --time=72:00:00           # Max wall time
#SBATCH --nodes=1          # Number of Nodes
#SBATCH --ntasks=16          # Number of tasks per job
#SBATCH --job-name=get_weights     # Job submission name
#SBATCH --mail-type=END            # Email user when job finishes
#SBATCH --mail-user=julia.lober@colorado.edu # Email address of user

ml purge
ml anaconda
conda activate coherence

#FLIGHT_NUMS=05208
LLH_DIR="/projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/raw/"
SLC_DIR="/projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/raw/"
ANN_DIR="/projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/raw/"
OUT_FP="/projects/julo9057/snowex_uavsar_coherence/data/snowex_lowman/"

python getweights_batch.py \
  --flight_nums 05208 23205 \
  --llh_dir ${LLH_DIR} \
  --ann_dir ${ANN_DIR} \
  --out_dir ${OUT_FP}

conda deactivate 
ml purge
