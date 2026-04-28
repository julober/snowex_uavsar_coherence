import os
import re
import rasterio
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box
import logging

########################################################################################
# Run with (example): 
# python scripts/slurms/generate_campaign_index_file.py \
#  --input /bsuhome/julialober/scratch/coherence_data/uavsar_cohs \
#  --output /bsuhome/julialober/scratch/coherence_data/campaign_summary_index.parquet
########################################################################################

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Regex to parse the coherence filenames
# Example: donner_03904_200131_200219_HH_s1_w13_int_coh.tif
COH_RE = re.compile(
    r"^(?P<flight_path>[^_]+)_"
    r"(?P<flight_num>\d{5})_"
    r"(?P<date1>\d{6})_"
    r"(?P<date2>\d{6})_"
    r"(?P<pol>[^_]+)_s(?P<seg>\d+)_"
)

def get_bounding_box(tid_path):
    """Extracts the bounding box from a GeoTIFF as a Shapely box."""
    with rasterio.open(tid_path) as src:
        bounds = src.bounds
        # box(minx, miny, maxx, maxy)
        return box(bounds.left, bounds.bottom, bounds.right, bounds.top), src.crs

def generate_index(root_dir, output_parquet):
    root_path = Path(root_dir)
    data = []

    # 1. Crawl directory for all .tif files
    logger.info(f"Scanning {root_dir}...")
    tif_files = list(root_path.rglob("*.tif"))
    
    for tif in tif_files:
        # Only process interferometric (int_coh) files for date pairs
        if "int_coh" not in tif.name:
            continue
            
        match = COH_RE.match(tif.name)
        if match:
            meta = match.groupdict()
            data.append({
                'flight_path': meta['flight_path'],
                'flight_num': meta['flight_num'],
                'date_pair': (meta['date1'], meta['date2']),
                'full_path': str(tif)
            })

    if not data:
        logger.error("No valid coherence TIFs found.")
        return

    df = pd.DataFrame(data)

    # 2. Aggregate unique date pairs per flight number
    # We group by flight_num but keep flight_path as well
    summary = df.groupby(['flight_path', 'flight_num']).agg({
        'date_pair': lambda x: sorted(list(set(x))),
        'full_path': 'first' # Use one file to get the bounding box
    }).reset_index()

    # Rename columns for clarity
    summary = summary.rename(columns={'date_pair': 'date_pairs'})

    # 3. Extract Geometries
    logger.info("Extracting spatial bounding boxes...")
    geometries = []
    crs_list = []
    
    for path in summary['full_path']:
        geom, crs = get_bounding_box(path)
        geometries.append(geom)
        crs_list.append(crs.to_string() if crs else "EPSG:4326")

    summary['geometry'] = geometries
    summary['crs'] = crs_list
    
    # Drop the temporary path used for bounds extraction
    summary = summary.drop(columns=['full_path'])

    # 4. Convert to GeoDataFrame and Save
    # We use the CRS from the first file (usually they are all consistent per flight)
    gdf = gpd.GeoDataFrame(summary, geometry='geometry', crs=crs_list[0])
    
    logger.info(f"Saving index with {len(gdf)} flight numbers to {output_parquet}")
    gdf.to_parquet(output_parquet)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Top level directory of coherence TIFs")
    parser.add_argument("--output", default="campaign_summary.parquet", help="Path to save geoparquet")
    args = parser.parse_args()

    generate_index(args.input, args.output)