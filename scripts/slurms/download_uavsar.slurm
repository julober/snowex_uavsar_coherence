#!/bin/bash
#SBATCH -J download_uavsar # job name
#SBATCH -o /bsuhome/julialober/logs/%x-%A_%a.log # output and error file name (%j expands to jobID)
#SBATCH -n 1 # total number of tasks requested
#SBATCH -N 1 # number of nodes you want to run on
#SBATCH -p bsudfq # queue (partition)
#SBATCH -t 12:00:00 # run time (hh:mm:ss) - 1 minute in this example.
#SBATCH --mail-user=julialober@u.boisestate.edu
#SBATCH --mail-type=END
#SBATCH --array=1-2

echo "Starting."

#eval "$(conda shell.bash hook)"
#mamba activate uavsar_pytools

echo "Making log dir..."

# 2. Make sure our log directory exists
mkdir -p /bsuhome/julialober/logs

# 3. Read the Nth li:ne from campaigns.txt (where N is the Slurm Array Task ID)
CAMPAIGN_FILE="campaigns_snowaccum.txt"
CAMPAIGN=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $CAMPAIGN_FILE)

# 4. Execute the Python script for that specific campaign
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Starting download for campaign: $CAMPAIGN"

/bsuhome/julialober/miniforge3/envs/uavsar_pytools/bin/python download_uavsar/download_campaign.py $CAMPAIGN

echo "Done"
