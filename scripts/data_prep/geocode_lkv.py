import argparse
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')))
from geocoding import interp_weights, geolocate_uavsar

def process_slc(slc_fp, ann_fp, llh_fp, out_dir, weights_fp=None):
    # Calculate weights if not provided
    if weights_fp is None or not os.path.exists(weights_fp):
        print("No valid weights provided.") 
        exit(1)
        # ...read src_pts, tgt_pts as in your workflow...
        vtx, wts, invalid = interp_weights(src_pts, tgt_pts, save=True, out_fp=weights_fp)
    else:
        print(f"Using preloaded weights from {weights_fp}")
        data = np.load(weights_fp, allow_pickle=True)
        vtx = data['vertices']
        wts = data['weights']
        invalid = data['invalid']
    # Geocode SLC
    print("Calling geolocate_uavsar")
    geolocate_uavsar(slc_fp, ann_fp, llh_fp, out_dir, vtx=vtx, wts=wts, invalid=invalid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lkv", nargs="+", help="lkv file")
    parser.add_argument("--ann", nargs="+", help="ann file")
    parser.add_argument("--llh_fp", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--weights_fp", default=None)
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()

    for lkv, ann in zip(args.lkv, args.ann) :
        print(f"Geocoding {lkv}")
        process_slc(lkv, ann, args.llh_fp, args.out_dir, args.weights_fp)
