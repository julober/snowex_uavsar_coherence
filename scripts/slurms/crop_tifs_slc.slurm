#!/bin/bash
#SBATCH -J crop_tifs # job name
#SBATCH -o /bsuhome/julialober/logs/%x-%A_%a.log # output and error file name (%j expands to jobID)
#SBATCH -n 1 # total number of tasks requested
#SBATCH -N 1 # number of nodes you want to run on
#SBATCH -p bsudfq # queue (partition)
#SBATCH -t 12:00:00 # run time (hh:mm:ss) - 1 minute in this example.
#SBATCH --mail-user=julialober@u.boisestate.edu
#SBATCH --mail-type=END

# ----------------- CONFIGURATION -----------------
# Path to the directory containing your geolocated tifs
DATA_DIR="/bsuhome/julialober/scratch/coherence_data/uavsar_geoslcs/grmesa/27416/"

# Path to the text file containing the list of files to crop
# for f in `ls /bsuhome/julialober/scratch/coherence_data/uavsar_geoslcs/stlake/27129/*1x1*`; do basename "$f"; done >> stlake_1x1_list.txt
LIST_FILE="crop_tif/grmesa_27416_1x1_list.txt"

# Bounding Box coordinates: minx miny maxx maxy
# Example layout (Lon_Min Lat_Min Lon_Max Lat_Max)
# BBOX="-111.7332140938517 40.58585096349459 -111.6857755957641 40.60845482790766" # SLC north LCC
BBOX="-108.2579339076524 38.99561257820081 -108.1334585623525 39.09310506271517" # Gr Mesa

# coords = Polygon([
#     (-111.7332140938517, 40.6079909188825),
#     (-111.7307874732357, 40.58585096349459),
#     (-111.6857755957641, 40.58703464363237),
#     (-111.6873712400506, 40.60845482790766),
#     (-111.7332140938517, 40.6079909188825)
# ])

# Set to "--delete" to purge the original uncropped tifs, or leave empty "" to keep them
DELETE_FLAG=""
# -------------------------------------------------

# Echo job tracking details to log
echo "Starting cropping batch job at $(date)"
echo "Target Directory: $DATA_DIR"
echo "BBox constraints: $BBOX"

# Load required cluster modules or activate conda environment
# Example:
# module load anaconda
# source activate uavsar_env

# Run the Python compilation tool
/bsuhome/julialober/miniforge3/envs/coherence/bin/python crop_tif/crop_tifs.py \
    --file-list "$LIST_FILE" \
    --input-dir "$DATA_DIR" \
    --bbox $BBOX \
    $DELETE_FLAG

echo "Job finished tracking at $(date)"