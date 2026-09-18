#!/bin/bash
#SBATCH -J calc_coherence                          # job name
#SBATCH -o /bsuhome/julialober/logs/%x-%A_%a.log                # per-array-task log (%A = job ID, %a = task ID)
#SBATCH -n 1                                        # total number of tasks requested
#SBATCH -N 1                                        # number of nodes you want to run on
#SBATCH -p bsudfq                                   # queue (partition)
#SBATCH -t 12:00:00                                 # run time (hh:mm:ss)
#SBATCH --mail-user=julialober@u.boisestate.edu
#SBATCH --mail-type=END
#SBATCH --array=1

echo "Starting coherence calculation (array task ${SLURM_ARRAY_TASK_ID})..."
echo "Making log dir..."
mkdir -p logs

# 1. Load your environment
source ~/.bashrc

# 2. Base data directories
# Change these paths to point to your actual data locations
BASE_INPUT="/bsuhome/julialober/scratch/coherence_data/uavsar_geoslcs"
OUTPUT_DIRECTORY="/bsuhome/julialober/scratch/coherence_data/uavsar_cohs"

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
    echo "${FLIGHT_DIR}"
    echo "ERROR: No directory for array task ID ${SLURM_ARRAY_TASK_ID}. Exiting."
    exit 1
fi

echo "Array Task ID:  ${SLURM_ARRAY_TASK_ID}"
# echo "Flight Dir:     ${FLIGHT_DIR}"
echo "Output Dir:     ${OUTPUT_DIRECTORY}"

# 5. Choose a mode:
#    - interferometric  (nearest-neighbor temporal pairs, same flight/pol/segment)
#    - crosspol            (HV vs. VH cross-polarization pairs, same flight/date/segment)
MODE="crosspol"

# 5a. Optional: path to a text file listing specific .tif files to process
#     (one absolute path per line; blank lines and # comments are ignored).
#     When set, --input_dir is not passed and the SLURM array flight-dir
#     selection above is bypassed — set --array=1 in the SBATCH header.
#     Leave empty ("") to use directory-based discovery (default behaviour).
FILE_LIST="calc_coherence/peeler_31619_files.txt"

# 5b. Which temporal pairs to compute in interferometric mode.
#     "nearest_neighbors" – consecutive pairs only (default, faster)
#     "all"              – every unique pair within each group
PAIRS="nearest_neighbors"

# 6. Set the rectangular window size for the coherence averaging filter (row col)
# WINDOW_SIZE=(3 5)
WINDOW_SIZE=(7 10)
# WINDOW_SIZE=(14 20)
# WINDOW_SIZE=(40 40)
# WINDOW_SIZE=(60 60)
# WINDOW_SIZE=(80 80)
# WINDOW_SIZE=(110 110)
# WINDOW_SIZE=(133 133)

# 7. Optional polarization filter for interferometric mode (HH, HV, VH, VV).
#    Leave empty ("") to process all polarizations.
POLARIZATION=""

echo "Mode:           ${MODE}"
echo "Window Size:    ${WINDOW_SIZE[*]}"
echo "Polarization:   ${POLARIZATION:-all}"

# 8. Build the optional --polarization flag
POL_FLAG=""
if [ -n "${POLARIZATION}" ]; then
    POL_FLAG="--polarization ${POLARIZATION}"
fi

# 9. Build the input flag: use file list if provided, otherwise the flight directory
if [ -n "${FILE_LIST}" ]; then
    INPUT_FLAG="--file_list ${FILE_LIST}"
else
    INPUT_FLAG="--input_dir ${FLIGHT_DIR}"
fi

echo "Input flag: ${INPUT_FLAG}"

# 10. Run the script
/bsuhome/julialober/miniforge3/envs/coherence_v2/bin/python calc_coherence/calc_coherence.py \
    ${INPUT_FLAG} \
    --out_dir     "${OUTPUT_DIRECTORY}" \
    --mode        "${MODE}" \
    --window_size "${WINDOW_SIZE[@]}" \
    --pairs       "${PAIRS}" \
    ${POL_FLAG}

echo "Coherence calculation complete!"
