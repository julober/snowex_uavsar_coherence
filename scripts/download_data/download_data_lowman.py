# This script is mostly from the SnowEx Hackweek UAVSAR Tutorial 

from uavsar_pytools import UavsarCollection
import os
## Collection name, the SnowEx Collection names are listed above. These are case 
## and space sensitive.
collection_name = 'Lowman, CO'

## Directory to save collection into. This will be filled with directory with 
## scene names and tiffs inside of them.
out_dir = './data/snowex_lowman'

## This is optional, but you will generally want to at least limit the date
## range between 2019 and today.
date_range = ('2019-11-01', '2023-12-31')

# Keywords: to download incidence angles with each image use `inc = True`
# For only certain pols use `pols = ['VV','HV']`

collection = UavsarCollection(collection = collection_name, work_dir = out_dir, dates = date_range, pols = ['HH'])
collection.find_urls()

print(f'\nCurrent working directory is {os.getcwd()}')
yesno = input(f"Do you want to download this data to {out_dir}? (y/n): ")

if (yesno == 'y') : 
    print(f'This will take a long time and a lot of space, ~1-5 gB and 10 minutes per\n' \
           'image pair depending on which scene, so run it if you have the space and time.')
    yesno = input(f"Are you sure you want to continue? (y/n): ")
    if (yesno == 'y') :
        # this downloads all the data
        collection.collection_to_tiffs()
else : 
    exit