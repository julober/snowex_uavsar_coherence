import argparse
import re
import logging
import rioxarray as rxr
import py3dep
from shapely.geometry import box
from pathlib import Path

# Adjust this import to point to where your calc_inc_angle function lives!
from uavsar_pytools.incidence_angle import calc_inc_angle 

def main():
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Find LKV files, reproject to UTM, and calculate incidence angles.")
    parser.add_argument('--lkv_dir', type=str, required=True, help="Path to folder containing .lkv files.")
    parser.add_argument('--out_dir', type=str, required=True, help="Path to save the output .inc.tif files.")
    args = parser.parse_args()

    lkv_dir = Path(args.lkv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use rglob to search recursively
    x_files = list(lkv_dir.rglob("*lkv.x*.tif")) 
    
    logger.info(f"Found {len(x_files)} flight segments to process in {lkv_dir}.")

    for x_file in x_files:
        match = re.search(r'_(\d{5})_.*_(s\d+)_', x_file.name)
        if not match:
            logger.warning(f"Skipping {x_file.name}: Could not parse flight_id and segment.")
            continue
            
        flight_id, seg_num = match.groups()
        out_name = out_dir / f"{flight_id}_{seg_num}.inc.tif"
        
        if out_name.exists():
            logger.info(f"Skipping {flight_id}_{seg_num}: Already exists.")
            continue

        # Locate corresponding Y and Z files
        y_file = x_file.with_name(x_file.name.replace('lkv.x', 'lkv.y'))
        z_file = x_file.with_name(x_file.name.replace('lkv.x', 'lkv.z'))
        
        if not (y_file.exists() and z_file.exists()):
            logger.warning(f"Missing Y or Z file for {flight_id}_{seg_num}. Skipping.")
            continue

        logger.info(f"Processing Flight {flight_id}, Segment {seg_num}...")
        
        try:
            # --- 1. Load Look Vectors (in EPSG:4326) ---
            logger.info("  -> Loading LKV files...")
            lkv_x_4326 = rxr.open_rasterio(x_file, masked=True).squeeze()
            lkv_y_4326 = rxr.open_rasterio(y_file, masked=True).squeeze()
            lkv_z_4326 = rxr.open_rasterio(z_file, masked=True).squeeze()

            # --- 2. Fetch the DEM using py3dep ---
            logger.info("  -> Fetching 10 m DEM via py3dep...")
            aoi_4326 = box(*lkv_x_4326.rio.bounds())
            dem_4326 = py3dep.get_dem(geometry=aoi_4326, resolution=10, crs="EPSG:4326") 

            # --- 3. Reproject to UTM Zone 11N (Meters) ---
            #utm_crs = "EPSG:32611" # Lowman, ID is in UTM Zone 11N
            res_m = 5.556          # Standard UAVSAR pixel resolution in meters
            utm_crs = lkv_x_4326.rio.estimate_utm_crs()
            logger.info(f"  -> Automatically selected local UTM CRS: {utm_crs}")

            logger.info(f"  -> Reprojecting arrays to {utm_crs}...")
            lkv_x_utm = lkv_x_4326.rio.reproject(utm_crs, resolution=res_m)
            lkv_y_utm = lkv_y_4326.rio.reproject(utm_crs, resolution=res_m)
            lkv_z_utm = lkv_z_4326.rio.reproject(utm_crs, resolution=res_m)

            # Force the DEM to perfectly align with the new LKV grid
            dem_utm = dem_4326.rio.reproject_match(lkv_x_utm)

            # --- 4. Calculate True Incidence Angle ---
            logger.info("  -> Calculating incidence angle...")
            inc_utm_arr = calc_inc_angle(
                dem=dem_utm.values,
                lkv_x=lkv_x_utm.values,
                lkv_y=lkv_y_utm.values,
                lkv_z=lkv_z_utm.values,
                pixel_size=res_m
            )

            # --- 5. Reproject Back to EPSG:4326 ---
            logger.info("  -> Reprojecting results back to EPSG:4326...")
            inc_utm_da = lkv_x_utm.copy(data=inc_utm_arr)
            inc_4326_da = inc_utm_da.rio.reproject_match(lkv_x_4326)
            
            # --- 6. Save the final output ---
            inc_4326_da.name = "incidence_angle"
            inc_4326_da.rio.to_raster(out_name)
            logger.info(f"  -> Successfully saved to {out_name.name}")
            
        except Exception as e:
            logger.error(f"  -> Error processing {flight_id}_{seg_num}: {e}")

if __name__ == "__main__":
    main()
