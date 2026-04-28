#!/bin/bash
#SBATCH -J assemble_data # job name
#SBATCH -o /bsuhome/julialober/logs/%x-%j.log # output and error file name (%j expands to jobID)
#SBATCH -n 1 # total number of tasks requested
#SBATCH -N 1 # number of nodes you want to run on
#SBATCH -p bsudfq # queue (partition)
#SBATCH -t 12:00:00 # run time (hh:mm:ss)
#SBATCH --mail-user=julialober@u.boisestate.edu
#SBATCH --mail-type=END

echo "Starting assemble_data job."

echo "Making log dir..."
# Make sure our log directory exists
mkdir -p /bsuhome/julialober/logs

echo "Starting Python execution..."
# Execute the Python runner script
/bsuhome/julialober/miniforge3/envs/coherence_v2/bin/python run_assemble/run_assemble_mcs.py

echo "Job complete."