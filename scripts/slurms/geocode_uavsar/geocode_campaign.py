import sys
import os
import shutil
import logging
from pathlib import Path
import rasterio
import numpy as np
import time
from datetime import datetime
from uavsar_pytools.georeference import geolocate_uavsar

# Force all logs (INFO and up) to print to the Slurm .log file with timestamps
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    stream=sys.stdout,
    force=True  # Overrides any logging settings from rasterio/uavsar_pytools
)

def combine_to_complex_tif(real_fp, imag_fp, out_fp):
    """Combines real and imaginary TIFs into a single complex64 TIF."""
    with rasterio.open(real_fp) as src_real, rasterio.open(imag_fp) as src_imag:
        real_arr = src_real.read(1)
        imag_arr = src_imag.read(1)
        profile = src_real.profile

    # Combine into a single complex numpy array
    complex_arr = real_arr + 1j * imag_arr

    # Update the GeoTIFF profile to support complex numbers
    profile.update(
        dtype='complex64',
        count=1
    )

    # Write the combined array to a new file
    with rasterio.open(out_fp, 'w', **profile) as dst:
        dst.write(complex_arr.astype(np.complex64), 1)

def main():  
    logging.info(f"UAVSAR PYTOOLS LOADED FROM: {geolocate_uavsar.__code__.co_filename}")
    if len(sys.argv) != 2:
        print("Usage: python geocode_campaign.py <campaign_abbr>")
        sys.exit(1)

    campaign = sys.argv[1].strip()
    logging.info(f"Starting geocoding for campaign: {campaign}")

    # Define root paths
    in_root = Path("/bsuhome/julialober/scratch/coherence_data/uavsar_slcs") / campaign
    out_root = Path("/bsuhome/julialober/scratch/coherence_data/uavsar_geoslcs") / campaign

    if not in_root.exists():
        logging.error(f"Input directory does not exist: {in_root}")
        sys.exit(1)

    # 1. Loop through each flight path/number directory
    for flight_dir in in_root.iterdir():
        if not flight_dir.is_dir():
            continue

        flight_num = flight_dir.name
        out_dir = out_root / flight_num
        out_dir.mkdir(parents=True, exist_ok=True)

        # 2. Find all SLC segments in this folder
        slcs = list(flight_dir.glob("*.slc"))
        
        for slc_fp in slcs:
            base_name = slc_fp.name
            parts = base_name.replace('.slc', '').split('_')
            
            # Ensure we have a correctly formatted UAVSAR stack filename
            if len(parts) < 10:
                logging.warning(f"Unexpected filename format for {base_name}. Skipping.")
                continue
                
            site = parts[0]       # lowman
            line = parts[1]       # 05208
            segment_id = parts[6] # 01
            pol = parts[7]        # BU
            swath = parts[8]      # s1
            looks = parts[9]      # 2x8

            # Reconstruct the LLH and LKV filenames
            llh_name = f"{site}_{line}_{segment_id}_{pol}_{swath}_{looks}.llh"
            lkv_name = f"{site}_{line}_{segment_id}_{pol}_{swath}_{looks}.lkv"
            
            llh_fp = flight_dir / llh_name
            lkv_fp = flight_dir / lkv_name
            
            # Reconstruct the ANN filename
            ann_name = "_".join(parts[:8]) + ".ann"
            ann_fp = flight_dir / ann_name

            # Fallback in case of unexpected date/flight suffix quirk in the .ann
            if not ann_fp.exists():
                anns = list(flight_dir.glob("*.ann"))
                if anns:
                    ann_fp = anns[0]
                else:
                    logging.warning(f"Missing ANN file for {base_name}. Skipping.")
                    continue

            if not llh_fp.exists():
                logging.warning(f"Missing LLH file {llh_name} for {base_name}. Skipping.")
                continue

            # Save the original directory so we can return to it safely
            original_cwd = os.getcwd()
            tmp_path = out_dir / "tmp"

            # ==========================================
            # STEP 1: GEOCODE THE SLC
            # ==========================================
            out_complex_tif = out_dir / f"{base_name}.complex.tif"
            if out_complex_tif.exists():
                logging.info(f"  --> Skipping SLC {base_name}: {out_complex_tif.name} already exists.")
            else:
                start_time_slc = datetime.now()
                t0_slc = time.time()
                logging.info(f"==================================================")
                logging.info(f"START [{start_time_slc.strftime('%Y-%m-%d %H:%M:%S')}]: Geolocating SLC block")
                logging.info(f"  --> Target SLC: {slc_fp.name}")
                logging.info(f"  --> Using ANN : {ann_fp.name}")
                logging.info(f"  --> Using LLH : {llh_fp.name}")
                
                try:
                    # Change the working directory to the isolated output folder!
                    os.chdir(out_dir)

                    # Remove any orphaned tmp directory before processing
                    shutil.rmtree(tmp_path, ignore_errors=True)

                    geolocate_uavsar(str(slc_fp), str(ann_fp), str(out_dir), str(llh_fp))
                    
                    real_tif = out_dir / f"{base_name}.real.tif"
                    imag_tif = out_dir / f"{base_name}.imag.tif"

                    if real_tif.exists() and imag_tif.exists():
                        logging.info(f"  --> Combining real and imag into single complex TIF...")
                        combine_to_complex_tif(str(real_tif), str(imag_tif), str(out_complex_tif))
                        
                        real_tif.unlink()
                        imag_tif.unlink()
                        logging.info(f"  --> Successfully created: {out_complex_tif.name}")
                    else:
                        logging.error(f"  --> Failed to find expected .real.tif or .imag.tif for {base_name}.")
                except Exception as e:
                    logging.error(f"  --> Failed to process SLC {slc_fp.name}: {e}")
                finally:
                    # Always return to the original directory
                    os.chdir(original_cwd)
                    # Clean up tmp directory after processing
                    shutil.rmtree(tmp_path, ignore_errors=True)
                    
                    # Calculate elapsed time for SLC
                    t1_slc = time.time()
                    end_time_slc = datetime.now()
                    elapsed_slc = (t1_slc - t0_slc) / 60.0
                    logging.info(f"END [{end_time_slc.strftime('%Y-%m-%d %H:%M:%S')}]: SLC {base_name} done. Elapsed Time: {elapsed_slc:.2f} minutes.")

            # ==========================================
            # STEP 2: GEOCODE THE LKV (LOOK VECTOR)
            # ==========================================
            if lkv_fp.exists():
                expected_lkv = out_dir / f"{base_name}.lkv.x.tif"
                if expected_lkv.exists():
                    logging.warning(f"  --> Skipping LKV {lkv_name}: {expected_lkv.name} already exists.")
                else:
                    start_time_lkv = datetime.now()
                    t0_lkv = time.time()
                    logging.info(f"--------------------------------------------------")
                    logging.info(f"START [{start_time_lkv.strftime('%Y-%m-%d %H:%M:%S')}]: Geolocating LKV block")
                    logging.info(f"  --> Target LKV: {lkv_fp.name}")
                    logging.info(f"  --> Using ANN : {ann_fp.name}")
                    logging.info(f"  --> Using LLH : {llh_fp.name}")
                    
                    try:
                        # Change directory again for the LKV step
                        os.chdir(out_dir)

                        # Remove any orphaned tmp directory before processing
                        shutil.rmtree(tmp_path, ignore_errors=True)

                        geolocate_uavsar(str(lkv_fp), str(ann_fp), str(out_dir), str(llh_fp))
                        logging.info(f"  --> Successfully geolocated LKV (x, y, z tifs generated) for {base_name}")
                    except Exception as e:
                        logging.error(f"  --> Failed to process LKV {lkv_name}: {e}")
                    finally:
                        os.chdir(original_cwd)
                        # Clean up tmp directory after processing
                        shutil.rmtree(tmp_path, ignore_errors=True)
                        
                        # Calculate elapsed time for LKV
                        t1_lkv = time.time()
                        end_time_lkv = datetime.now()
                        elapsed_lkv = (t1_lkv - t0_lkv) / 60.0
                        logging.info(f"END [{end_time_lkv.strftime('%Y-%m-%d %H:%M:%S')}]: LKV {lkv_name} done. Elapsed Time: {elapsed_lkv:.2f} minutes.")
            else:
                logging.warning(f"Missing LKV file {lkv_name} for {base_name}. Skipping LKV geocoding.")

    logging.info(f"==================================================")
    logging.info(f"Finished geocoding all lines for {campaign}!")

if __name__ == "__main__":
    main()