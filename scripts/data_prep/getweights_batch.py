import argparse
import os
import numpy as np
from pathlib import Path
from geocoding import interp_weights

def get_llh_paths(flight_num, llh_dir):
    """Return the .llh file path for a given flight number."""
    llh_files = sorted(Path(llh_dir).glob(f"*{flight_num}*.llh"))
    if not llh_files:
        raise FileNotFoundError(f"No .llh file found for flight {flight_num} in {llh_dir}")
    return str(llh_files[0])

def get_ann_paths(flight_num, ann_dir):
    """Return the .ann file path for a given flight number."""
    ann_files = sorted(Path(ann_dir).glob(f"*{flight_num}*.ann"))
    if not ann_files:
        raise FileNotFoundError(f"No .ann file found for flight {flight_num} in {ann_dir}")
    return str(ann_files[0])

def main():
    parser = argparse.ArgumentParser(description="Calculate interpolation weights for flight paths.")
    parser.add_argument('--flight_nums', nargs='+', required=True, help='List of flight numbers (e.g. 05208 23205)')
    parser.add_argument('--llh_dir', required=True, help='Directory containing .llh files')
    parser.add_argument('--ann_dir', required=True, help='Directory containing .ann files')
    parser.add_argument('--out_dir', required=True, help='Directory to save weights .npz files')
    parser.add_argument('--rows', type=int, default=None, help='Override number of rows')
    parser.add_argument('--cols', type=int, default=None, help='Override number of cols')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for flight_num in args.flight_nums:
        print(f"Processing flight {flight_num}")
        llh_fp = get_llh_paths(flight_num, args.llh_dir)
        ann_fp = get_ann_paths(flight_num, args.ann_dir)

        # Read annotation for rows/cols
        import pandas as pd
        from uavsar_pytools.convert.tiff_conversion import read_annotation
        metadata = read_annotation(ann_fp)
        metadata = pd.DataFrame(metadata).T
        rows = args.rows if args.rows is not None else int(metadata.loc['slc_2_2x8 rows', 'value'])
        cols = args.cols if args.cols is not None else int(metadata.loc['slc_2_2x8 columns', 'value'])

        arr = np.fromfile(llh_fp, dtype='f4')
        lats = arr[0::3].reshape(rows, cols)
        lons = arr[1::3].reshape(rows, cols)
        src_pts = np.column_stack([lons.ravel(), lats.ravel()])

        # Target grid: same as source
        lat_space = np.linspace(np.min(lats), np.max(lats), num=rows)
        lon_space = np.linspace(np.min(lons), np.max(lons), num=cols)
        tgt_lons, tgt_lats = np.meshgrid(lon_space, lat_space, indexing='xy')
        tgt_pts = np.column_stack([tgt_lons.ravel(), tgt_lats.ravel()])

        out_fp = os.path.join(args.out_dir, f"lowman_{flight_num}_weights.npz")
        print(f"Calculating weights for flight {flight_num} ...")
        vtx, wts, invalid = interp_weights(src_pts, tgt_pts, save=True, out_fp=out_fp)
        print(f"Saved weights to {out_fp}")

if __name__ == "__main__":
    main()
