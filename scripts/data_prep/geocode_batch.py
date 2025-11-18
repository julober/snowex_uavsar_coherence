import argparse
import os
from joblib import Parallel, delayed
from geocoding import interp_weights, geolocate_uavsar

def process_slc(slc_fp, ann_fp, llh_fp, out_dir, weights_fp=None):
    # Calculate weights if not provided
    if weights_fp is None or not os.path.exists(weights_fp):
        # ...read src_pts, tgt_pts as in your workflow...
        vtx, wts, invalid = interp_weights(src_pts, tgt_pts, save=True, out_fp=weights_fp)
    else:
        data = np.load(weights_fp, allow_pickle=True)
        vtx = data['vertices']
        wts = data['weights']
        invalid = data['invalid']
    # Geocode SLC
    geolocate_uavsar(slc_fp, ann_fp, llh_fp, out_dir, vtx=vtx, wts=wts, invalid=invalid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slc_list", nargs="+", help="List of SLC files")
    parser.add_argument("--ann_list", nargs="+", help="List of annotation files")
    parser.add_argument("--llh_fp", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--weights_fp", default=None)
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()

    Parallel(n_jobs=args.n_jobs)(
        delayed(process_slc)(slc, ann, args.llh_fp, args.out_dir, args.weights_fp)
        for slc, ann in zip(args.slc_list, args.ann_list)
    )
