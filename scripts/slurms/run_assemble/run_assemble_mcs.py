import pandas as pd
from shapely.wkt import loads
import sys 
from pathlib import Path
import logging

# Configure logging to show timestamps, log level, and the message
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.layers import assemble_data # Ensure this matches your actual import

# 1. Parse the Polygon using Well-Known Text (WKT)
aoi_wkt = "POLYGON ((-115.63548033331044 43.90718161502215, -115.63548033331044 43.98397534443853, -115.7352319093607 43.98397534443853, -115.7352319093607 43.90718161502215, -115.63548033331044 43.90718161502215))"
mcs_aoi = loads(aoi_wkt)

# 2. Define the date pairs
dates_pd = [
    (pd.Timestamp('2020-01-31 00:00:00'), pd.Timestamp('2020-02-13 00:00:00')), 
    (pd.Timestamp('2020-01-31 00:00:00'), pd.Timestamp('2020-02-21 00:00:00')), 
    (pd.Timestamp('2020-01-31 00:00:00'), pd.Timestamp('2020-03-11 00:00:00')), 
    (pd.Timestamp('2020-01-31 00:00:00'), pd.Timestamp('2021-01-15 00:00:00')), 
    (pd.Timestamp('2020-01-31 00:00:00'), pd.Timestamp('2021-01-20 00:00:00')), 
    (pd.Timestamp('2020-01-31 00:00:00'), pd.Timestamp('2021-01-27 00:00:00')), 
    (pd.Timestamp('2020-01-31 00:00:00'), pd.Timestamp('2021-02-03 00:00:00')), 
    (pd.Timestamp('2020-01-31 00:00:00'), pd.Timestamp('2021-02-10 00:00:00')), 
    (pd.Timestamp('2020-01-31 00:00:00'), pd.Timestamp('2021-03-03 00:00:00')), 
    (pd.Timestamp('2020-01-31 00:00:00'), pd.Timestamp('2021-03-10 00:00:00')), 
    (pd.Timestamp('2020-01-31 00:00:00'), pd.Timestamp('2021-03-16 00:00:00')), 
    (pd.Timestamp('2020-01-31 00:00:00'), pd.Timestamp('2021-03-22 00:00:00')), 
    (pd.Timestamp('2020-02-13 00:00:00'), pd.Timestamp('2020-02-21 00:00:00')), 
    (pd.Timestamp('2020-02-13 00:00:00'), pd.Timestamp('2020-03-11 00:00:00')), 
    (pd.Timestamp('2020-02-13 00:00:00'), pd.Timestamp('2021-01-15 00:00:00')), 
    (pd.Timestamp('2020-02-13 00:00:00'), pd.Timestamp('2021-01-20 00:00:00')), 
    (pd.Timestamp('2020-02-13 00:00:00'), pd.Timestamp('2021-01-27 00:00:00')), 
    (pd.Timestamp('2020-02-13 00:00:00'), pd.Timestamp('2021-02-03 00:00:00')), 
    (pd.Timestamp('2020-02-13 00:00:00'), pd.Timestamp('2021-02-10 00:00:00')), 
    (pd.Timestamp('2020-02-13 00:00:00'), pd.Timestamp('2021-03-03 00:00:00')), 
    (pd.Timestamp('2020-02-13 00:00:00'), pd.Timestamp('2021-03-10 00:00:00')), 
    (pd.Timestamp('2020-02-13 00:00:00'), pd.Timestamp('2021-03-16 00:00:00')), 
    (pd.Timestamp('2020-02-13 00:00:00'), pd.Timestamp('2021-03-22 00:00:00')), 
    (pd.Timestamp('2020-02-21 00:00:00'), pd.Timestamp('2020-03-11 00:00:00')), 
    (pd.Timestamp('2020-02-21 00:00:00'), pd.Timestamp('2021-01-15 00:00:00')), 
    (pd.Timestamp('2020-02-21 00:00:00'), pd.Timestamp('2021-01-20 00:00:00')), 
    (pd.Timestamp('2020-02-21 00:00:00'), pd.Timestamp('2021-01-27 00:00:00')), 
    (pd.Timestamp('2020-02-21 00:00:00'), pd.Timestamp('2021-02-03 00:00:00')), 
    (pd.Timestamp('2020-02-21 00:00:00'), pd.Timestamp('2021-02-10 00:00:00')), 
    (pd.Timestamp('2020-02-21 00:00:00'), pd.Timestamp('2021-03-03 00:00:00')), 
    (pd.Timestamp('2020-02-21 00:00:00'), pd.Timestamp('2021-03-10 00:00:00')), 
    (pd.Timestamp('2020-02-21 00:00:00'), pd.Timestamp('2021-03-16 00:00:00')), 
    (pd.Timestamp('2020-02-21 00:00:00'), pd.Timestamp('2021-03-22 00:00:00')), 
    (pd.Timestamp('2020-03-11 00:00:00'), pd.Timestamp('2021-01-15 00:00:00')), 
    (pd.Timestamp('2020-03-11 00:00:00'), pd.Timestamp('2021-01-20 00:00:00')), 
    (pd.Timestamp('2020-03-11 00:00:00'), pd.Timestamp('2021-01-27 00:00:00')), 
    (pd.Timestamp('2020-03-11 00:00:00'), pd.Timestamp('2021-02-03 00:00:00')), 
    (pd.Timestamp('2020-03-11 00:00:00'), pd.Timestamp('2021-02-10 00:00:00')), 
    (pd.Timestamp('2020-03-11 00:00:00'), pd.Timestamp('2021-03-03 00:00:00')), 
    (pd.Timestamp('2020-03-11 00:00:00'), pd.Timestamp('2021-03-10 00:00:00')), 
    (pd.Timestamp('2020-03-11 00:00:00'), pd.Timestamp('2021-03-16 00:00:00')), 
    (pd.Timestamp('2020-03-11 00:00:00'), pd.Timestamp('2021-03-22 00:00:00')), 
    (pd.Timestamp('2021-01-15 00:00:00'), pd.Timestamp('2021-01-20 00:00:00')), 
    (pd.Timestamp('2021-01-15 00:00:00'), pd.Timestamp('2021-01-27 00:00:00')), 
    (pd.Timestamp('2021-01-15 00:00:00'), pd.Timestamp('2021-02-03 00:00:00')), 
    (pd.Timestamp('2021-01-15 00:00:00'), pd.Timestamp('2021-02-10 00:00:00')), 
    (pd.Timestamp('2021-01-15 00:00:00'), pd.Timestamp('2021-03-03 00:00:00')), 
    (pd.Timestamp('2021-01-15 00:00:00'), pd.Timestamp('2021-03-10 00:00:00')), 
    (pd.Timestamp('2021-01-15 00:00:00'), pd.Timestamp('2021-03-16 00:00:00')), 
    (pd.Timestamp('2021-01-15 00:00:00'), pd.Timestamp('2021-03-22 00:00:00')), 
    (pd.Timestamp('2021-01-20 00:00:00'), pd.Timestamp('2021-01-27 00:00:00')), 
    (pd.Timestamp('2021-01-20 00:00:00'), pd.Timestamp('2021-02-03 00:00:00')), 
    (pd.Timestamp('2021-01-20 00:00:00'), pd.Timestamp('2021-02-10 00:00:00')), 
    (pd.Timestamp('2021-01-20 00:00:00'), pd.Timestamp('2021-03-03 00:00:00')), 
    (pd.Timestamp('2021-01-20 00:00:00'), pd.Timestamp('2021-03-10 00:00:00')), 
    (pd.Timestamp('2021-01-20 00:00:00'), pd.Timestamp('2021-03-16 00:00:00')), 
    (pd.Timestamp('2021-01-20 00:00:00'), pd.Timestamp('2021-03-22 00:00:00')), 
    (pd.Timestamp('2021-01-27 00:00:00'), pd.Timestamp('2021-02-03 00:00:00')), 
    (pd.Timestamp('2021-01-27 00:00:00'), pd.Timestamp('2021-02-10 00:00:00')), 
    (pd.Timestamp('2021-01-27 00:00:00'), pd.Timestamp('2021-03-03 00:00:00')), 
    (pd.Timestamp('2021-01-27 00:00:00'), pd.Timestamp('2021-03-10 00:00:00')), 
    (pd.Timestamp('2021-01-27 00:00:00'), pd.Timestamp('2021-03-16 00:00:00')), 
    (pd.Timestamp('2021-01-27 00:00:00'), pd.Timestamp('2021-03-22 00:00:00')), 
    (pd.Timestamp('2021-02-03 00:00:00'), pd.Timestamp('2021-02-10 00:00:00')), 
    (pd.Timestamp('2021-02-03 00:00:00'), pd.Timestamp('2021-03-03 00:00:00')), 
    (pd.Timestamp('2021-02-03 00:00:00'), pd.Timestamp('2021-03-10 00:00:00')), 
    (pd.Timestamp('2021-02-03 00:00:00'), pd.Timestamp('2021-03-16 00:00:00')), 
    (pd.Timestamp('2021-02-03 00:00:00'), pd.Timestamp('2021-03-22 00:00:00')), 
    (pd.Timestamp('2021-02-10 00:00:00'), pd.Timestamp('2021-03-03 00:00:00')), 
    (pd.Timestamp('2021-02-10 00:00:00'), pd.Timestamp('2021-03-10 00:00:00')), 
    (pd.Timestamp('2021-02-10 00:00:00'), pd.Timestamp('2021-03-16 00:00:00')), 
    (pd.Timestamp('2021-02-10 00:00:00'), pd.Timestamp('2021-03-22 00:00:00')), 
    (pd.Timestamp('2021-03-03 00:00:00'), pd.Timestamp('2021-03-10 00:00:00')), 
    (pd.Timestamp('2021-03-03 00:00:00'), pd.Timestamp('2021-03-16 00:00:00')), 
    (pd.Timestamp('2021-03-03 00:00:00'), pd.Timestamp('2021-03-22 00:00:00')), 
    (pd.Timestamp('2021-03-10 00:00:00'), pd.Timestamp('2021-03-16 00:00:00')), 
    (pd.Timestamp('2021-03-10 00:00:00'), pd.Timestamp('2021-03-22 00:00:00')), 
    (pd.Timestamp('2021-03-16 00:00:00'), pd.Timestamp('2021-03-22 00:00:00'))
]

# 3. Define paths and lists
flight_ids = ['05208', '23205']
coh_dir = '/bsuhome/julialober/scratch/coherence_data/uavsar_cohs/lowman'
inc_dir = '/bsuhome/julialober/scratch/coherence_data/inc_angle'
sc_dir = '/bsuhome/julialober/scratch/coherence_data/NSIDC-0768'

if __name__ == "__main__":
    logger.info("Executing assemble_data...")
    
    data = assemble_data(
        tile_aoi=mcs_aoi,
        date_pairs=dates_pd,
        flight_ids=flight_ids,
        fp_coh=coh_dir,
        fp_inc=inc_dir,
        fp_snowclimate=sc_dir
    )
    
    logger.info("Finished assembly.")
    
    # Save the output to disk as a Zarr store
    zarr_out_path = '/bsuhome/julialober/scratch/coherence_data/stacks/mores_creek2/'
    logger.info(f"Saving to Zarr store at: {zarr_out_path}")
    
    data.to_zarr(zarr_out_path, mode='w')
    
    logger.info("Save complete.")
