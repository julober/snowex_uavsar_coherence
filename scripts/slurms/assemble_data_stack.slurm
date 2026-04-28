#!/bin/bash
#SBATCH -J assemble_campaigns
#SBATCH -o /bsuhome/julialober/logs/%x-%A_%a.log
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p bsudfq
#SBATCH -t 12:00:00
#SBATCH --array=0-2   # <-- Adjust based on number of campaigns (0 to N-1)

# 1. Define your array of campaigns
# (Make sure the array length matches your #SBATCH --array limit above!)
CAMPAIGNS=("lowman" "mores_creek" "grand_mesa")

# Get the specific campaign for this array task
CURRENT_CAMPAIGN=${CAMPAIGNS[$SLURM_ARRAY_TASK_ID]}

# 2. Define standard directories
OUT_DIR="/bsuhome/julialober/scratch/coherence_data/stacks"
COH_DIR="/bsuhome/julialober/scratch/coherence_data/uavsar_cohs"
INC_DIR="/bsuhome/julialober/scratch/coherence_data/inc_angle"
SC_DIR="/bsuhome/julialober/scratch/coherence_data/NSIDC-0768"
PARQUET_IDX="/bsuhome/julialober/scratch/coherence_data/campaign_summary_index.parquet"

# Optional overrides (Leave completely blank to let Python deduce them)
AOI_WKT="" 
FLIGHT_IDS=""
DATE_PAIRS=""

echo "Starting job for campaign: $CURRENT_CAMPAIGN"

# 3. Build the Python command
CMD="/bsuhome/julialober/miniforge3/envs/coherence_clean/bin/python run_campaign.py \
    --campaign $CURRENT_CAMPAIGN \
    --out-dir $OUT_DIR \
    --coh-dir $COH_DIR \
    --inc-dir $INC_DIR \
    --sc-dir $SC_DIR \
    --parquet-index $PARQUET_IDX"

# Add optional arguments only if they are not empty
if [ -n "$AOI_WKT" ]; then CMD="$CMD --aoi-wkt \"$AOI_WKT\""; fi
if [ -n "$FLIGHT_IDS" ]; then CMD="$CMD --flight-ids $FLIGHT_IDS"; fi
if [ -n "$DATE_PAIRS" ]; then CMD="$CMD --date-pairs $DATE_PAIRS"; fi

# Execute
eval $CMD

echo "Job complete for $CURRENT_CAMPAIGN."