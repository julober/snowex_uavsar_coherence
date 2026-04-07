#!/bin/bash
#SBATCH -J calc_coherence                          # job name
#SBATCH -o /bsuhome/julialober/logs/%x-%j.log      # output and error file name (%j expands to jobID)
#SBATCH -n 1                                        # total number of tasks requested
#SBATCH -N 1                                        # number of nodes you want to run on
#SBATCH -p bsudfq                                   # queue (partition)
#SBATCH -t 12:00:00                                 # run time (hh:mm:ss)
#SBATCH --mail-user=julialober@u.boisestate.edu
#SBATCH --mail-type=END

echo "Starting coherence calculation..."
echo "Making log dir..."
mkdir -p /bsuhome/julialober/logs

# 1. Load your environment
source ~/.bashrc

# 2. Define your directories
# Change these paths to point to your actual data locations
INPUT_DIRECTORY="/bsuhome/julialober/scratch/coherence_data/uavsar_geoslcs"
OUTPUT_DIRECTORY="/bsuhome/julialober/scratch/coherence_data/coherence"

# 3. Choose a mode:
#    - interferometric  (nearest-neighbor temporal pairs, same flight/pol/segment)
#    - copol            (HV vs. VH cross-polarization pairs, same flight/date/segment)
MODE="interferometric"

# 4. Set the square window size for the coherence averaging filter (default: 5)
WINDOW_SIZE=5

echo "Input Dir:   ${INPUT_DIRECTORY}"
echo "Output Dir:  ${OUTPUT_DIRECTORY}"
echo "Mode:        ${MODE}"
echo "Window Size: ${WINDOW_SIZE}"

# 5. Run the script
/bsuhome/julialober/miniforge3/envs/coherence/bin/python calc_coherence/calc_coherence.py \
    --input_dir "${INPUT_DIRECTORY}" \
    --out_dir   "${OUTPUT_DIRECTORY}" \
    --mode      "${MODE}" \
    --window_size "${WINDOW_SIZE}"

echo "Coherence calculation complete!"
