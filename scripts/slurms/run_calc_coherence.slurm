#!/bin/bash
#SBATCH -J calc_coherence                          # job name
#SBATCH --output=logs/%x-%A_%a.log                 # per-array-task log (%A = job ID, %a = task ID)
#SBATCH -n 1                                        # total number of tasks requested
#SBATCH -N 1                                        # number of nodes you want to run on
#SBATCH -p bsudfq                                   # queue (partition)
#SBATCH -t 12:00:00                                 # run time (hh:mm:ss)
#SBATCH --mail-user=julialober@u.boisestate.edu
#SBATCH --mail-type=END
#SBATCH --array=1-999%10                            # upper bound auto-trimmed; adjust %N to set max concurrent tasks

echo "Starting coherence calculation (array task ${SLURM_ARRAY_TASK_ID})..."
echo "Making log dir..."
mkdir -p logs

# 1. Load your environment
source ~/.bashrc

# 2. Base data directories
# Change these paths to point to your actual data locations
BASE_INPUT="/bsuhome/julialober/scratch/coherence_data/uavsar_geoslcs"
OUTPUT_DIRECTORY="/bsuhome/julialober/scratch/coherence_data/coherence"

# 3. Build an array of all lowest-level flight directories
#    Expected layout: BASE_INPUT/<site>/<flight>/
#    e.g. uavsar_geoslcs/lowman/05208/  uavsar_geoslcs/lowman/23205/ ...
mapfile -t FLIGHT_DIRS < <(find "${BASE_INPUT}" -mindepth 2 -maxdepth 2 -type d | sort)

NUM_FLIGHTS=${#FLIGHT_DIRS[@]}
echo "Total flight directories found: ${NUM_FLIGHTS}"

if [ "${NUM_FLIGHTS}" -eq 0 ]; then
    echo "ERROR: No flight directories found under ${BASE_INPUT}. Exiting."
    exit 1
fi

# 4. Select the flight directory for this task (Slurm arrays are 1-indexed)
FLIGHT_DIR="${FLIGHT_DIRS[$((SLURM_ARRAY_TASK_ID - 1))]}"

if [ -z "${FLIGHT_DIR}" ]; then
    echo "ERROR: No directory for array task ID ${SLURM_ARRAY_TASK_ID}. Exiting."
    exit 1
fi

echo "Array Task ID:  ${SLURM_ARRAY_TASK_ID}"
echo "Flight Dir:     ${FLIGHT_DIR}"
echo "Output Dir:     ${OUTPUT_DIRECTORY}"

# 5. Choose a mode:
#    - interferometric  (nearest-neighbor temporal pairs, same flight/pol/segment)
#    - copol            (HV vs. VH cross-polarization pairs, same flight/date/segment)
MODE="interferometric"

# 6. Set the square window size for the coherence averaging filter (default: 5)
WINDOW_SIZE=5

# 7. Optional polarization filter for interferometric mode (HH, HV, VH, VV).
#    Leave empty ("") to process all polarizations.
POLARIZATION=""

echo "Mode:           ${MODE}"
echo "Window Size:    ${WINDOW_SIZE}"
echo "Polarization:   ${POLARIZATION:-all}"

# 8. Build the optional --polarization flag
POL_FLAG=""
if [ -n "${POLARIZATION}" ]; then
    POL_FLAG="--polarization ${POLARIZATION}"
fi

# 9. Run the script for this specific flight directory
/bsuhome/julialober/miniforge3/envs/coherence/bin/python calc_coherence/calc_coherence.py \
    --input_dir   "${FLIGHT_DIR}" \
    --out_dir     "${OUTPUT_DIRECTORY}" \
    --mode        "${MODE}" \
    --window_size "${WINDOW_SIZE}" \
    ${POL_FLAG}

echo "Coherence calculation complete for ${FLIGHT_DIR}!"
