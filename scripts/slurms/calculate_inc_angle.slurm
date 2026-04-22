#!/bin/bash
#SBATCH -J calculate_inc_angle # job name
#SBATCH -o /bsuhome/julialober/logs/%x-%j.log # output and error file name (%j expands to jobID)
#SBATCH -n 1 # total number of tasks requested
#SBATCH -N 1 # number of nodes you want to run on
#SBATCH -p bsudfq # queue (partition)
#SBATCH -t 12:00:00 # run time (hh:mm:ss) - 1 minute in this example.
#SBATCH --mail-user=julialober@u.boisestate.edu
#SBATCH --mail-type=END

# 1. Load your environment
# (Update this to match however you initialize your conda/mamba environment on your cluster)
source ~/.bashrc

# 2. Define your directories
# Change these paths to point to your actual data locations
LKV_DIRECTORY="/bsuhome/julialober/scratch/coherence_data/uavsar_geoslcs/"
OUTPUT_DIRECTORY="/bsuhome/julialober/scratch/coherence_data/inc_angle/"

# 3. Run the script
echo "Starting incidence angle calculations..."
echo "LKV Dir: ${LKV_DIRECTORY}"
echo "Out Dir: ${OUTPUT_DIRECTORY}"

/bsuhome/julialober/miniforge3/envs/uavsar_pytools/bin/python calc_inc_angle/calculate_inc_angle.py \
    --lkv_dir "${LKV_DIRECTORY}" \
    --out_dir "${OUTPUT_DIRECTORY}"

echo "Incidence angle calculations complete!"
