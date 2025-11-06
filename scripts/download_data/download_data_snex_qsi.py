import earthaccess
import os

out_dir = "./data/snowex_lowman/snex_qsi"

earthaccess.login()

# 2. Search
results = earthaccess.search_data(
    short_name='SNEX20_QSI_DEM',  # ATLAS/ICESat-2 L3A Land Ice Height
    bounding_box=(-116.61035, 43.1676, -114.99345, 45.12889),  # Only include files in area of interest...
)

print(results)

print(f'\nCurrent working directory is {os.getcwd()}')
yesno = input(f"Do you want to download this data to {out_dir}? (y/n): ")

if (yesno == 'y') : 
    yesno = input(f"Are you sure? (y/n): ")
    if (yesno == 'y') :
        files = earthaccess.download(results, out_dir)
else : 
    exit

