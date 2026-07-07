#!/usr/bin/env encoding
import argparse
import os
from osgeo import gdal

def crop_raster(input_path, bbox, delete_original=False):
    """
    Crops a single GeoTIFF file to the given bounding box using GDAL.
    """
    if not os.path.exists(input_path):
        print(f"Warning: File not found, skipping: {input_path}")
        return False

    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_cropped{ext}"
    
    print(f"Processing: {input_path} -> {output_path}")
    
    try:
        # outputBounds expects: [minx, miny, maxx, maxy]
        warp_options = gdal.WarpOptions(
            format="GTiff",
            outputBounds=bbox,
            dstNodata=-9999
        )
        
        # Execute the crop warp
        gdal.Warp(output_path, input_path, options=warp_options)
        
        # Verify output was created successfully before deleting old file
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            if delete_original:
                print(f"--> Successfully cropped. Deleting original: {input_path}")
                os.remove(input_path)
            else:
                print(f"--> Successfully cropped. Kept original.")
            return True
        else:
            print(f"Error: Generated output file is empty for {input_path}")
            return False
            
    except Exception as e:
        print(f"Failed to crop {input_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Crop a list of geolocated GeoTIFFs to a bounding box.")
    
    parser.add_argument("-l", "--file-list", required=True, 
                        help="Path to a text file containing names/paths of TIF files (one per line).")
    parser.add_argument("-d", "--input-dir", default=None, 
                        help="Optional directory path containing the files listed in the file-list.")
    parser.add_argument("-b", "--bbox", type=float, nargs=4, required=True,
                        help="Bounding box layout: minx miny maxx maxy (e.g., -118.5 34.0 -118.1 34.4)")
    parser.add_argument("--delete", action="store_true", 
                        help="Delete original files after a successful crop. Default: Keep them.")

    args = parser.parse_args()

    # Read target files
    if not os.path.exists(args.file_list):
        print(f"Error: File list '{args.file_list}' does not exist.")
        return

    with open(args.file_list, 'r') as f:
        files = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

    print(f"Found {len(files)} files to process. Target BBox: {args.bbox}")

    success_count = 0
    for filename in files:
        # Resolve path if an input directory was provided
        filepath = os.path.join(args.input_dir, filename) if args.input_dir else filename
        
        if crop_raster(filepath, args.bbox, delete_original=args.delete):
            success_count += 1

    print(f"\nDone! Successfully processed {success_count}/{len(files)} files.")

if __name__ == "__main__":
    main()