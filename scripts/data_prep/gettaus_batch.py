# script that takes a list of coherence files and outputs tau values 
import numpy as np
from pathlib import Path
import argparse
import rioxarray as rxa
import rasterio
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')))
from taus import get_taus_spatial

# fit parameters
tau_guess = 6
bounds = (0,20)
xtol=1e-6
ftol=1e-6
gamma_inf = 0.17  # or None to fit gamma
out_dir = '../data/snowex_lowman/taus/'

# takes command line arguments: <coherence_tifs> <days_between_scenes> <output_file>
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coh_tifs', nargs='+', required=True, help='List of coherence .tif files')
    parser.add_argument('--days', nargs='+', required=True, help='File with number of days between scenes')
    parser.add_argument('--out_fp', required=True, help='Output file path for tau .npy file')
    args = parser.parse_args()

    # load coherence files into a 3D numpy array
    coh_arrays = []
    for coh_tif in args.coh_tifs:
        coh_data = rxa.open_rasterio(coh_tif)[0].values  # assume single band
        coh_arrays.append(coh_data)
    coh_arr = np.stack(coh_arrays, axis=0)

    # read days between scenes file
    days_fp = args.days[0]
    print("---- DEBUG ----")
    print("cwd:", os.getcwd())
    print("args.days:", days_fp)
    print("Is file?", os.path.isfile(days_fp))
    print("Is dir?", os.path.isdir(days_fp))
    print("sys.executable:", sys.executable)
    print("----------------")   
    days = np.loadtxt(days_fp, dtype=float)
    days = np.array(days)

    if coh_arr.shape[0] != len(days):
        print("Error: Number of coherence files does not match number of days entries.")
        print(f"Number of day intervals: {len(days)}")
        print(days)
        print(f"Number of coherence files: {len(args.coh_tifs)}")
        print(args.coh_tifs)
        sys.exit(1)

    # calculate taus
    tau_grid, success_grid = get_taus_spatial(coh_arr, days, tau_guess, bounds, xtol, ftol, gamma_inf)

    height, width = coh_arrays[0].shape
    transform = coh_arrays[0].rio.transform()
    crs = coh_arrays[0].rio.crs

    # save tau grid to output file
    with rasterio.open(
            out_dir + f'{args.out_fp}',
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='float32',
            crs=crs,
            transform=transform,
            nodata=np.nan,
        ) as dst:
            dst.write(coh.astype('float32'), 1)
