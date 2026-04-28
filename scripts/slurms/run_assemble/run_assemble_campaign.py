import argparse
import sys
import logging
import pandas as pd
from pathlib import Path
from shapely.wkt import loads

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.layers import assemble_data_largeaoi

def parse_args():
    parser = argparse.ArgumentParser(description="Run data assembly for a campaign.")
    
    # Required Arguments (Provided by Slurm)
    parser.add_argument("--campaign", required=True, help="Campaign/Flight path name")
    parser.add_argument("--out-dir", required=True, help="Base output directory")
    parser.add_argument("--coh-dir", required=True, help="Coherence directory path")
    parser.add_argument("--inc-dir", required=True, help="Incidence angle directory path")
    parser.add_argument("--sc-dir", required=True, help="Snowclimate directory path")
    parser.add_argument("--parquet-index", required=True, help="Path to your metadata parquet file")
    
    # Optional Arguments (If not provided, Python will figure them out)
    parser.add_argument("--aoi-wkt", default=None, help="WKT string of bounding box")
    parser.add_argument("--flight-ids", nargs='*', default=None, help="List of flight IDs")
    parser.add_argument("--date-pairs", default=None, help="Optional specific date pairs")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Starting assembly for campaign: {args.campaign}")

    # 1. Deduce missing info from Parquet if needed
    if not args.aoi_wkt or not args.flight_ids:
        logger.info("AOI or Flight IDs not provided. Looking up in Parquet index...")
        # Assuming you made the summary table mentioned above
        df_index = pd.read_parquet(args.parquet_index)
        
        # Filter to just this campaign
        campaign_df = df_index[df_index['flight_path'] == args.campaign]
        
        if campaign_df.empty:
            logger.error(f"Campaign {args.campaign} not found in {args.parquet_index}")
            sys.exit(1)
            
        # Extract flight IDs and AOI
        flight_ids = args.flight_ids if args.flight_ids else campaign_df['flight_num'].unique().tolist()
        
        # Grab the bounding box for this campaign (assuming it's stored as WKT)
        aoi_wkt = args.aoi_wkt if args.aoi_wkt else campaign_df['aoi_geometry'].iloc[0]
        master_aoi = loads(aoi_wkt)
    else:
        flight_ids = args.flight_ids
        master_aoi = loads(args.aoi_wkt)

    # Note: If date_pairs is None, we let assemble_data_largeaoi figure it out 
    # automatically via your existing filepath-parsing logic!
    date_pairs = None 

    # 2. Setup the output directory
    # Result: /out/dir/<campaign>/
    campaign_out_dir = Path(args.out_dir) / args.campaign
    campaign_out_dir.mkdir(parents=True, exist_ok=True)
    zarr_out_path = str(campaign_out_dir / "assembled_data.zarr")

    # 3. Run the pipeline
    assemble_data_largeaoi(
        master_aoi=master_aoi,
        flight_ids=flight_ids,
        date_pairs=date_pairs, 
        fp_dest=zarr_out_path,
        fp_coh=args.coh_dir,
        fp_inc=args.inc_dir,
        fp_snowclimate=args.sc_dir
    )
    
    logger.info(f"Finished {args.campaign}. Data saved to {zarr_out_path}")