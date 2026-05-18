import argparse
import sys
import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.wkt import loads
from shapely.ops import unary_union  # <-- Added for geometry merging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.layers import assemble_data_largeaoi

def parse_args():
    parser = argparse.ArgumentParser(description="Run data assembly for a campaign.")
    
    # Required Arguments
    parser.add_argument("--campaign", required=True, help="Campaign/Flight path name")
    parser.add_argument("--out-dir", required=True, help="Base output directory")
    parser.add_argument("--coh-dir", required=True, help="Coherence directory path")
    parser.add_argument("--inc-dir", required=True, help="Incidence angle directory path")
    parser.add_argument("--sc-dir", required=True, help="Snowclimate directory path")
    parser.add_argument("--parquet-index", required=True, help="Path to your metadata parquet file")
    
    # Optional Arguments
    parser.add_argument("--aoi-wkt", default=None, help="WKT string of bounding box")
    parser.add_argument("--flight-ids", nargs='*', default=None, help="List of flight IDs")
    parser.add_argument("--date-pairs", default=None, help="Optional specific date pairs")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Starting assembly for campaign: {args.campaign}")

    # 1. Deduce missing info from Parquet if needed
    if not args.aoi_wkt or not args.flight_ids or not args.date_pairs:
        logger.info("Missing arguments. Looking up in Parquet index...")
        df_index = gpd.read_parquet(args.parquet_index)
        
        # Filter to just this campaign
        campaign_df = df_index[df_index['flight_path'] == args.campaign]
        
        if campaign_df.empty:
            logger.error(f"Campaign {args.campaign} not found in {args.parquet_index}")
            sys.exit(1)
            
        # --- FLIGHT IDS ---
        flight_ids = args.flight_ids if args.flight_ids else campaign_df['flight_num'].unique().tolist()
        
        # --- AOI UNION ---
        if args.aoi_wkt:
            master_aoi = loads(args.aoi_wkt)
        else:
            # Load all WKT geometries for this campaign into Shapely Polygons
            geometries = campaign_df['geometry'].tolist()
            # Merge them into a single shape, then get the rectangular bounding envelope
            merged_aoi = unary_union(geometries).envelope
            master_aoi = merged_aoi
            logger.info(f"Unioned AOI created for flight(s): {flight_ids}")

        # --- DATE PAIRS UNION ---
        if args.date_pairs:
            # If passed explicitly, you'd likely want to evaluate/parse it here
            date_pairs = args.date_pairs
        else:
            # Assume Parquet has a 'date_pairs' column (containing lists of date tuples/strings)
            if 'date_pairs' in campaign_df.columns:
                all_pairs = set()
                for pairs_list in campaign_df['date_pairs']:
                    # Add to set to automatically drop duplicates across flight nums
                    all_pairs.update(tuple(p) for p in pairs_list)
                
                # Convert back to list of pandas Timestamps
                date_pairs = [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in all_pairs]
                # Sort chronologically by start date, then end date
                date_pairs = sorted(date_pairs, key=lambda x: (x[0], x[1]))
                logger.info(f"Unioned {len(date_pairs)} unique date pairs from parquet.")
            else:
                logger.info("No 'date_pairs' column found in parquet. Leaving as None for auto-discovery.")
                date_pairs = None
    else:
        flight_ids = args.flight_ids
        master_aoi = loads(args.aoi_wkt)
        date_pairs = args.date_pairs

    # 2. Setup the output directory
    campaign_out_dir = Path(args.out_dir) / args.campaign
    campaign_out_dir.mkdir(parents=True, exist_ok=True)
    zarr_out_path = str(campaign_out_dir)

    # 3. Run the pipeline
    logger.info(f"Target Zarr store: {zarr_out_path}")
    assemble_data_largeaoi(
        master_aoi=master_aoi,
        flight_ids=flight_ids,
        date_pairs=date_pairs, 
        fp_dest=zarr_out_path,
        fp_coh=args.coh_dir,
        fp_inc=args.inc_dir,
        fp_snowclimate=args.sc_dir
    )
    
    logger.info(f"Finished {args.campaign}. Data saved successfully.")