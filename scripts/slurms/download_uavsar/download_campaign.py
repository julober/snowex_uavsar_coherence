import sys
import os
import logging # Add this!

# Add this block right below your imports!
# This forces ALL logs (INFO and up) to print to stdout (your Slurm .out file)
logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s: %(message)s',
    stream=sys.stdout 
)

from uavsar_pytools.download.download_slcs import get_uavsar_slcs, download_uavsar_slcs

def main():
    if len(sys.argv) != 2:
        print("Usage: python download_campaign.py <campaign_abbr>")
        sys.exit(1)

    campaign = sys.argv[1].strip()
    print(f"Processing campaign: {campaign}")

    # 1. Get the dictionary of all files for this campaign
    # Format returned: {'lowman_05208': ['url1', 'url2'], ...}
    if True: 
        links_dict = get_uavsar_slcs(flight_name=campaign,
                getann=True,
                getllh=True,
                getlkv=True,
                pol=['HH', 'HV', 'VH', 'VV'],
                pxlsp=['1x1'])
                #tag=['BU','BC'])
    else: 
        links_dict = get_uavsar_slcs(flight_name=campaign,
                getann=True,
                getllh=True,
                getlkv=True,
                pol=['HH', 'HV', 'VH', 'VV'],
                pxlsp=['1x1'])

    if not links_dict:
        print(f"No files found for campaign: {campaign}")
        sys.exit(0)

    # 2. Loop through every flight line in that dictionary
    for dict_key, files_to_download in links_dict.items():

        # Split 'lowman_05208' into 'lowman' and '05208'
        try:
            abbr, flight_num = dict_key.split('_', 1)
        except ValueError:
            print(f"Unexpected key format: {dict_key}. Skipping...")
            continue

        # 3. Create the specific output directory: abbr/flightnum/
        root_dir = "/bsuhome/julialober/scratch/coherence_data/uavsar_slcs"
        out_dir = os.path.join(root_dir, abbr, flight_num)
        os.makedirs(out_dir, exist_ok=True)

        # 4. Download this flight line's files into that specific folder
        print(f"\n>>> Downloading {len(files_to_download)} files to {out_dir}/ ...")
        download_uavsar_slcs(files_to_download, out_dir)

    print(f"\nFinished processing all flight lines for {campaign}!")

if __name__ == "__main__":
    main()
