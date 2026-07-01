#!/bin/bash
#SBATCH -J geocode_uavsar
#SBATCH -o /bsuhome/julialober/logs/%x-%A_%a.log
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p bsudfq
#SBATCH -t 48:00:00 
#SBATCH --mail-user=julialober@u.boisestate.edu
#SBATCH --mail-type=END
#SBATCH --array=1-2           # Adjust to match the lines in campaigns_retry.txt

echo "Starting."
echo "Making log dir..."
mkdir -p /bsuhome/julialober/logs

CAMPAIGN_FILE="campaigns_snowaccum.txt"
CAMPAIGN=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $CAMPAIGN_FILE)

echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Starting geocoding for campaign: $CAMPAIGN"

# Use the direct path to your coherence environment's Python (where rasterio is installed)
/bsuhome/julialober/miniforge3/envs/coherence/bin/python -B geocode_uavsar/geocode_campaign.py $CAMPAIGN

echo "Done"
