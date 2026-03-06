import shapely
from shapely.geometry import Polygon
import geopandas as gpd
import py3dep

import numpy as np
import xarray as xr
import rioxarray 
import os
import time

# i don't think this will work - figure it out
from validation import validate_aoi, validate_date_pairs, make_reference_grid, validate_alignment
# from nlcd import get_nlcd_layers

def assemble_data(
    aoi, 
    date_pairs, 
    fp_dest,
    fp_coherence = '../data/coherence/',
    fp_snowclimate = '../data/NSIDC-0768/',
    crs = 'EPSG:4326', 
    res = 30,
    overwrite = False
) -> xr.Dataset: 
    
    """
    Assembles all the data layers needed for my model. 
    Can handle all flights from a particular flight path. User will be responsible
    for making sure that the dates passed in correspond to the dates of the 
    UAVSAR flights. 

    Should create a .zarr file that saves the data cube like: 
    location_name/
        static/
        dynamic/
            date1_date2/
            date1_date3/
            date2_date3/
        .zmetadata

    TODO: 
    - make this adaptable in case files already exist. 
    - save all output as files  
    - figure out how to handle the two directions of flights 

    """

    # 1. Validate inputs 
    aoi = validate_aoi(aoi)
    date_pairs = validate_date_pairs(date_pairs)
    aoi_gdf = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[aoi])

    # 2. Create reference grid 
    if crs == 'EPSG:4326' and res == 30:
        res_deg = 0.00381807117 # 30 m at 45 deg latitude 
    # 
    ref = make_reference_grid(aoi=aoi, crs=crs, resolution=res_deg)

    # 2. Get UAVSAR coherence layers
    coh = get_uavsar_coherence(aoi=aoi, 
                               date_pairs=date_pairs, 
                               crs=crs, 
                               ref_grid=ref, 
                               fp=fp_coherence)
    # print(coh)
    # print(coh['pair'].values)

    # 3. Get NLCD layers 
    s = time.time()
    nlcd = get_nlcd_layers(aoi=aoi, crs=crs, ref_grid=ref)
    e = time.time()
    print(f"Got NLCD {list(nlcd.keys())} in {e-s:.3f} seconds.")

    # 4. Get DEM 
    # print(f'Making call to py3dep.get_map with resolution {resolution} in m.')
    s = time.time()
    dem = py3dep.get_dem(geometry=aoi, resolution=res, crs=crs)
    dem.rio.write_crs(crs)
    dem = dem.rio.reproject_match(ref)
    e = time.time()
    print(f"Got DEM in {e-s:.3f} seconds.")

    s = time.time()
    topo = get_topo_layers(dem=dem, ref_grid=ref)
    e = time.time()
    print(f'Got topo layers: {list(topo.keys())} in {e-s:.3f} seconds.')

    s = time.time()
    snow_class = get_snow_climatology(aoi=aoi, crs=crs, fp=fp_snowclimate, ref_grid=ref)
    e = time.time()
    print(f"Got snow climatology: {list(snow_class.keys())} in {e-s:.3f} seconds.")

    # 5a. Get AORC for each range 
    s = time.time()
    aorc = get_aorc_layers(aoi=aoi, date_pairs=date_pairs, crs=crs, ref_grid=ref)
    e = time.time()
    print(f"Got AORC: {list(aorc.keys())} in {e-s:.3f} seconds.")

    # 5b. Get UCLA for each range 

    s = time.time() 
    snow = get_snow_layers(aoi=aoi, date_pairs=date_pairs, crs=crs, ref_grid=ref)
    e = time.time()
    print(f"Got SWE layers: {list(snow.keys())} in {e-s:.3f} seconds.")

    ds_list = [coh, nlcd, dem, topo, snow_class, snow, aorc]
    # ds_list = [nlcd, dem, topo, snow_class, snow]
    validate_alignment(ds_list)

    ds = xr.merge(ds_list, join='exact', compat='minimal')

    # When you load data, xarray remembers its original chunking in the 'encoding' dictionary.
    # If you don't delete this, to_zarr() will try to use the old chunks and crash!
    for var in ds.variables:
        ds[var].encoding.pop('chunks', None)
        ds[var].encoding.pop('preferred_chunks', None)

    # 2. Define the new Spatial-Only chunking strategy
    # Using -1 tells Dask to NOT chunk that dimension (keep it as one single block).
    # Replace 'time', 'y', and 'x' with your actual dimension names.
    chunk_dict = {
        'pair': -1,   # Keep the entire time series together
        'y': 25,     # Chunk spatially into 512x512 pixel grids
        'x': 25
    }

    # 3. Apply the unified chunks to the entire dataset
    ds_chunked = ds.chunk(chunk_dict)

    # 4. Save the cube to Zarr
    # consolidated=True puts all the metadata in one file, making it much faster to read later.
    # ds_chunked.to_zarr('my_unified_cube.zarr', mode='w', consolidated=True)

    if fp_dest is not None: 
        if overwrite: 
            ds_chunked.to_zarr(fp_dest, 
                               mode='w', 
                               consolidated=True)
        else:
            ds_chunked.to_zarr(fp_dest, 
                            mode='w-', 
                            consolidated=True)
    
    return ds
    

def get_uavsar_coherence(
    aoi, 
    date_pairs, 
    crs,
    fp,
    ref_grid = None,
) -> xr.Dataset: 

    # find files
    files = os.listdir(fp)
    # filter files by date pairs
    fname_pairs = {}
    for f in files:
        # print(f)
        for start_date, end_date in date_pairs:
            sd_str = start_date.strftime('%y%m%d')
            ed_str = end_date.strftime('%y%m%d')
            if sd_str in f and ed_str in f:
                # print(f'Found file {f} for pair: {sd_str}_{ed_str}')
                fname_pairs[f'{sd_str}_{ed_str}'] = f
    
    # open files as xarray dataset with pair dimension
    ds_list = []
    for k, f in fname_pairs.items():
        ds = xr.open_dataset(fp + f, chunks={})
        ds.rio.write_crs(crs, inplace=True)
        ds = ds.rename({list(ds.data_vars.keys())[0]: 'coherence'})
        ds['pair'] = k
        ds = ds.set_coords('pair')
        ds_list.append(ds)

    ds = xr.concat(ds_list, dim='pair')
    ds = ds.sortby('pair')

    # check that geometry is a GeoDataFrame and has a CRS 
    # if not isinstance(aoi, gpd.GeoDataFrame):
    #     raise ValueError("Input geometry must be a GeoDataFrame.")
    # if aoi.crs != 'EPSG:4326':
    #     raise ValueError("Input geometry must be in EPSG:4326 (WGS84).")
    
    

    if ref_grid is not None : 
        ds = ds.rio.reproject_match(ref_grid)

    return ds


# =============================================================================
# FUNCTIONS FOR DOWNLOADING ENVIRONMENTAL DATA
# =============================================================================

def get_nlcd_layers(
    aoi, 
    crs,
    # out_fp,
    years={'cover': [2019], 'canopy': [2019]},
    ref_grid = None
)-> xr.Dataset:
    """
    Get NLCD Layers. Saves the layers as tifs to the specified output filepath
    if they don't already exists, returns the reprojected layers as a dict. 
    
    Parameters 
    ----------
    geometry: geopandas.GeoDataFrame
        A GeoDataFrame geometry in EPSG:4326 (WGS84) 
    out_fp: str
        A filepath to save the output tifs to
    years: dict, optional
        The years of NLCD data to download. Should be a dictionary. Default 
        is {'cover': [2019], 'canopy': [2019]}.
    crs_dest: str, optional
        The target CRS for reprojecting the NLCD layers. Default is 
        'EPSG:26911' (UTM 11N).
    res_dest: int, optional
        The target resolution for reprojecting the NLCD layers. Default is 
        30m.

    Returns
    -------
    dict
        A dictionary of reprojected NLCD layers, with keys corresponding to the 
        raster layers.
    

    """
    # get nlcd layers for the given geometry and years
    import pygeohydro as gh
    import geopandas as gpd
    import os 

    g = gpd.GeoSeries([aoi], crs=crs)

    # check that geometry is a GeoDataFrame and has a CRS 
    # if not isinstance(aoi, gpd.GeoDataFrame):
    #     raise ValueError("Input geometry must be a GeoDataFrame.")
    # if not out_fp:
    #     raise ValueError("Output file path must be provided.")
    # if aoi.crs != 'EPSG:4326':
    #     raise ValueError("Input geometry must be in EPSG:4326 (WGS84).")
    
    # get nlcd
    ds = gh.nlcd_bygeom(geometry=g, years=years)[0]

    # ds = ds.rename({'cover_2019': 'cover', 'canopy_2019': 'canopy'})

    # nlcd_reproj = {}
    if ref_grid is not None : 
        ds = ds.rio.reproject_match(ref_grid)

    # save as tifs 
    # for key, value in nlcd_reproj.items():
    #     # check that file doesn't already exist 
    #     if os.path.exists(f"{out_fp}_{key}.tif"):
    #         raise ValueError(f"File {out_fp}_{key}.tif already exists.")
    #     else:
    #         value.rio.to_raster(f"{out_fp}_{key}.tif")

    # will download 2019 automatically - should I use 2021 instead? 
    return ds

def get_snow_climatology(
    aoi, 
    crs,
    fp, 
    ref_grid = None
):
    fname = '/SnowClass_NA_300m_10.0arcsec_2021_v01.0.nc'
    # open filepath fp
    # check if file exists
    if not os.path.exists(fp + fname):
        error_msg = f"Could not open file {fp + fname}. Please download {fname} from NSIDC at " \
                    "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0768_global_seasonal_snow_classification_v01/."
        raise ValueError(error_msg) 

    ds = xr.open_dataset(fp + fname, chunks={})
    ds = ds.rename({'lat': 'y', 'lon': 'x', 'SnowClass': 'snow_class'})
    ds = ds.sortby(['x', 'y'])

    g = gpd.GeoSeries([aoi], crs=crs).to_crs('EPSG:4326').total_bounds
    minx, miny, maxx, maxy = g
    # print(f"Clipping snow climatology to AOI bounds: {minx}, {miny}, {maxx}, {maxy}")
    # print(ds)

    ds = ds.sel(x=slice(minx, maxx), y=slice(miny, maxy))

    ds = ds.rio.write_crs('EPSG:4326')

    if ref_grid is not None :
        ds = ds.rio.reproject_match(ref_grid)
    else: 
        g = gpd.GeoSeries([aoi], crs=crs)
        ds = ds.rio.clip(g, crs=crs)

    return ds

import xrspatial as xrs

def get_topo_layers(
    dem,
    ref_grid = None
) -> xr.Dataset:
    ds = xr.Dataset()

    if dem.rio.crs == 'EPSG:4326':
        dem_reproj = dem.rio.reproject('EPSG:5070')
    else:
        dem_reproj = dem

    # pad to extend data values to the edge
    # n = 5
    # dem_padded = dem_reproj.pad(y=n, x=n, mode='edge')

    # ds['slope'] = xrs.slope(dem_padded)
    # ds['aspect'] = xrs.aspect(dem_padded)
    # ds['curve'] = xrs.curvature(dem_padded)

    # ds = ds.isel(y=slice(n, -n), x=slice(n, -n))

    # 1. Pad the DEM with NaNs as before
    dem_nan_padded = dem_reproj.pad(y=1, x=1, constant_values=np.nan)

    # 2. Add use_coordinate=False to bypass the monotonic error
    dem_extrapolated = dem_nan_padded.interpolate_na(
        dim='x', method='linear', fill_value="extrapolate", use_coordinate=False
    ).interpolate_na(
        dim='y', method='linear', fill_value="extrapolate", use_coordinate=False
    )

    # 3. Calculate metrics and trim the edges exactly as before
    slope_extrap = xrs.slope(dem_extrapolated)
    aspect_extrap = xrs.aspect(dem_extrapolated)
    curve_extrap = xrs.curvature(dem_extrapolated)

    # 4. Trim the results back to your original DEM dimensions
    # isel(slice(1, -1)) removes the 1-pixel artificial boundary we just created
    ds['slope'] = slope_extrap.isel(y=slice(1, -1), x=slice(1, -1))
    ds['aspect'] = aspect_extrap.isel(y=slice(1, -1), x=slice(1, -1))
    ds['curve'] = curve_extrap.isel(y=slice(1, -1), x=slice(1, -1))

    if ref_grid is not None : 
        ds = ds.rio.reproject_match(ref_grid)

    return ds

import earthaccess

def get_snow_layers(
    aoi, 
    crs,
    date_pairs,
    ref_grid = None
):
    # print(aoi)
    # turn aoi into list of points in counter-clockwise order 
    xx, yy = aoi.exterior.coords.xy
    x = xx.tolist()
    y = yy.tolist()

    g = gpd.GeoSeries([aoi], crs=crs)
    bounds = g.to_crs("EPSG:4326").total_bounds 

    # print(type(x), type(y))
    # print(list(zip(x,y)))

    years = get_years(date_pairs)

    # print(str(min(years)), str(max(years)))

    grans = earthaccess.search_data(
        short_name='WUS_UCLA_SR', 
        cloud_hosted=True, 
        temporal=(str(min(years)), str(max(years))),
        polygon=list(zip(x,y))
    )

    auth = earthaccess.login()
    if not auth.authenticated:
        # ask for credentials and persist them in a .netrc file
        auth.login(strategy="interactive", persist=True)

    # print(grans)
    fileset = earthaccess.open(grans)
    # ds = xr.open_mfdataset(fileset, chunks={})
    # Separate granules by type based on naming convention [cite: 42, 44]
    swe_files = [f for f in fileset if "SWE_SCA_POST" in (f.path if hasattr(f, 'path') else str(f))]
    sd_files = [f for f in fileset if "SD_POST" in (f.path if hasattr(f, 'path') else str(f))]

    # swe_list = [process_ucla_granule_eager(f, bounds) for f in swe_files]
    # sd_list = [process_ucla_granule_eager(f, bounds) for f in sd_files]

    # Process and concatenate each group along time
    ds_swe = xr.concat([process_ucla_granule(f) for f in swe_files], dim='time').sortby('time')
    ds_sd = xr.concat([process_ucla_granule(f) for f in sd_files], dim='time').sortby('time')

    # open_kwargs = dict(
    #     chunks={},
    #     preprocess=lambda d: process_ucla_granule_optimized(d, bounds),
    #     parallel=True,
    #     compat='override', # Silences the Future Warning about coordinate overlap
    #     coords='minimal'
    # )
    
    # ds_swe = xr.open_mfdataset(swe_files, **open_kwargs).sortby('time')
    # ds_sd = xr.open_mfdataset(sd_files, **open_kwargs).sortby('time')

    # ds_swe = xr.concat(swe_list, dim='time').sortby('time')
    # ds_sd = xr.concat(sd_list, dim='time').sortby('time')

    # Merge the different parameters into one large data cube
    # This combines SWE_Post, SCA_Post, and SD_Post [cite: 20, 32]
    ds_full = xr.merge([ds_swe, ds_sd], compat='override')
    ds_full = ds_full.isel(Stats=2)
    ds_full = ds_full.rio.write_crs("EPSG:4326")

    ds_clip = ds_full.rio.clip(g, crs=crs)

    # print(ds_clip)
    ds_clip = ds_clip.transpose('y', 'x', 'time')
    # print(ds_clip)

    ds = xr.Dataset()
    ds_list = []
    # get metrics for each date pair
    for i, (start_date, end_date) in enumerate(date_pairs):
        ds_slice = ds_clip.sel(time=slice(start_date, end_date))

        # ran into some missing data (e.g. 2021/02-2021/11)
        if (len(ds_slice['time']) == 0): 
            print(f'Could not get WUS_UCLA_SR data for {start_date} to {end_date}.')
            continue

        ds_metrics = get_snow_metrics(ds_slice,
                                      metrics=['swe_change', 'sd_change'])
        pair_name = start_date.strftime('%y%m%d') + "_" + end_date.strftime('%y%m%d')
        ds_metrics['pair'] = pair_name
        ds_metrics = ds_metrics.set_coords('pair')

        ds_list.append(ds_metrics)

        # reproject to reference grid 
        # if ref_grid is not None : 
        #     ds_metrics = ds_metrics.rio.reproject_match(ref_grid)

    
    ds = xr.concat(ds_list, dim='pair')
    ds = ds.sortby('pair')
    if ref_grid is not None : 
        ds = ds.rio.reproject_match(ref_grid)
    return ds
    # print(ds)


def get_snow_metrics(
    ds,
    metrics = 'all'
) -> xr.Dataset:
    
    out = xr.Dataset()

    valid_metrics = [
        'total_swe',
        'max_swe',
        'swe_change',
        'sd_change'
    ]

    if metrics == 'all':
        metrics = valid_metrics
    else:
        for metric in metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Valid metrics are: {valid_metrics}")

    if 'total_swe' in metrics: 
        out['total_swe'] = ds['SWE_Post'].sum(dim='time')
    if 'max_swe' in metrics: 
        out['max_swe'] = ds['SWE_Post'].max(dim='time')
    if 'swe_change' in metrics: 
        # first minus last 
        out['swe_change'] = ds['SWE_Post'].isel(time=-1) - ds['SWE_Post'].isel(time=0)
    if 'sd_change' in metrics: 
        # first minus last 
        out['sd_change'] = ds['SD_Post'].isel(time=-1) - ds['SD_Post'].isel(time=0)
    # if 'snow_cover_change' in metrics: 
    #     out['snow_cover_change'] = ds['SCA_Post']

    return out

import xarray as xr
import pandas as pd
import re

def process_ucla_granule_eager(file_obj, aoi_wgs84_bounds):
    # Open the dataset lazily
    ds = xr.open_dataset(file_obj, chunks={})
    
    # 1. Parse Water Year
    filename = getattr(file_obj, 'path', str(file_obj))
    match = re.search(r'WY(\d{4})_\d{2}', filename)
    start_year = int(match.group(1)) if match else 2000
    
    # 2. Assign Time
    num_days = ds.sizes['Day']
    time_coords = pd.date_range(start=f"{start_year}-10-01", periods=num_days, freq='D')
    
    # 3. Handle Spatial Coordinates
    ds = ds.rename({'Day': 'time', 'Latitude': 'y', 'Longitude': 'x'})
    ds = ds.assign_coords(
        time=time_coords,
        y=ds.y.values.flatten(),
        x=ds.x.values.flatten()
    )
    
    # 4. Rough Slice
    ds = ds.sortby(['x', 'y'])
    minx, miny, maxx, maxy = aoi_wgs84_bounds
    ds_slice = ds.sel(
        x=slice(minx - 0.01, maxx + 0.01), 
        y=slice(miny - 0.01, maxy + 0.01)
    )
    
    # THE CRITICAL STEP: Download this tiny slice into RAM right now
    # and return a pure, in-memory xarray dataset
    return ds_slice.compute()

def process_ucla_granule_optimized(ds, aoi_wgs84_bounds=None):
    """
    Revised to work as a 'preprocess' function for open_mfdataset.
    """
    # 1. Parse Water Year from the encoding (more reliable than filename regex in open_mfdataset)
    # If filename is available, use it; otherwise, infer from global attributes
    filename = ds.encoding.get('source', '')
    match = re.search(r'WY(\d{4})_\d{2}', filename)
    start_year = int(match.group(1)) if match else 2000 # fallback
    
    # 2. Assign Time
    num_days = ds.sizes['Day']
    time_coords = pd.date_range(start=f"{start_year}-10-01", periods=num_days, freq='D')
    
    # 3. Streamline Spatial Coordinates
    # We rename first so we can use .sel() effectively
    ds = ds.rename({'Day': 'time', 'Latitude': 'y', 'Longitude': 'x'})
    
    # Use standard 1D arrays for coords (only take the first column/row)
    ds = ds.assign_coords(
        time=time_coords,
        y=ds.y.values.flatten(),
        x=ds.x.values.flatten()
    )

    # 4. CRITICAL: Rough Spatial Slice
    # This discards 99% of the data before it ever hits your loop
    if aoi_wgs84_bounds is not None:
        minx, miny, maxx, maxy = aoi_wgs84_bounds
        ds = ds.sel(x=slice(minx, maxx), y=slice(maxy, miny))
    
    return ds

# def get_snow_layers_optimized(aoi, crs, date_pairs, ref_grid=None):
#     # 1. Setup Spatial Filter
#     g = gpd.GeoSeries([aoi], crs=crs)
#     aoi_wgs84_bounds = g.to_crs("EPSG:4326").total_bounds # [minx, miny, maxx, maxy]
    
#     years = get_years(date_pairs)
    
#     # ... [Search and Auth code remains the same] ...

#     # 2. Parallel Opening with open_mfdataset
#     # Using a lambda to pass the bounds into the preprocess function
#     open_kwargs = dict(
#         chunks={'Day': -1, 'Latitude': 100, 'Longitude': 100},
#         preprocess=lambda d: process_ucla_granule_optimized(d, aoi_wgs84_bounds),
#         parallel=True
#     )
    
#     ds_swe = xr.open_mfdataset(swe_files, **open_kwargs).sortby('time')
#     ds_sd = xr.open_mfdataset(sd_files, **open_kwargs).sortby('time')

#     # 3. Combine and Slice Stats (Median)
#     ds_full = xr.merge([ds_swe, ds_sd]).isel(Stats=2)
#     ds_full = ds_full.rio.write_crs("EPSG:4326")
    
#     # Fine clip (accurate to the polygon)
#     ds_clip = ds_full.rio.clip(g, crs=crs).compute() # Compute here to stop lazy-loading overhead
    
#     # 4. Metrics Loop (Now fast because data is small and in memory)
#     ds_list = []
#     for start_date, end_date in date_pairs:
#         ds_slice = ds_clip.sel(time=slice(start_date, end_date))
#         if ds_slice.sizes['time'] == 0:
#             continue

#         ds_metrics = get_snow_metrics(ds_slice, metrics=['swe_change', 'sd_change'])
#         pair_name = f"{start_date:%y%m%d}_{end_date:%y%m%d}"
#         ds_metrics = ds_metrics.assign_coords(pair=pair_name).set_coords('pair')
#         ds_list.append(ds_metrics)

#     ds_final = xr.concat(ds_list, dim='pair').sortby('pair')
    
#     if ref_grid is not None:
#         ds_final = ds_final.rio.reproject_match(ref_grid)
        
#     return ds_final

def process_ucla_granule(file_obj):
    # Open the dataset
    ds = xr.open_dataset(file_obj, chunks={})
    # print(f"Processing file: {file_obj}")
    # print(ds)
    
    # 1. Parse the Water Year (WY) for the time axis [cite: 30, 42]
    filename = file_obj.path if hasattr(file_obj, 'path') else str(file_obj)
    match = re.search(r'WY(\d{4})_\d{2}', filename)
    start_year = int(match.group(1))
    
    # 2. Create time coordinate starting Oct 1st [cite: 30, 101]
    num_days = ds.sizes['Day']
    time_coords = pd.date_range(start=f"{start_year}-10-01", periods=num_days, freq='D')
    
    # 3. Handle Spatial Coordinates [cite: 37, 39]
    # Squeeze (225x1) to 1D and rename to y (lat) and x (lon)
    lat_values = ds.Latitude.values.flatten()
    lon_values = ds.Longitude.values.flatten()
    
    # Drop old lat/lon to avoid conflicts during renaming
    # ds = ds.drop_vars(['Latitude', 'Longitude'])
    
    # Rename 'day' to 'time' and map statistics dimension if necessary [cite: 28]
    ds = ds.rename({'Day': 'time', 'Latitude': 'y', 'Longitude': 'x'})
    
    # Assign the new dimensional coordinates
    ds = ds.assign_coords(
        time=time_coords,
        y=lat_values,
        x=lon_values
    )
    
    return ds


import s3fs
import fsspec

def get_aorc_layers(
    aoi, 
    date_pairs, 
    crs,
    ref_grid = None
) -> xr.Dataset: 
    """
    Get AORC layers for the given AOI and date pairs. Returns a dictionary of 
    xarray DataArrays, with keys corresponding to the variable names. 

    Parameters 
    ----------
    aoi: shapely.Polygon
        A shapely Polygon geometry in EPSG:4326 (WGS84)
    date_pairs: list of tuples
        A list of (start_date, end_date) tuples, where each date is a string in 
        the format 'YYYY-MM-DD'.
    ref_grid: xarray.DataArray
        An xarray DataArray to use as a reference grid for reprojection. 

    Returns
    -------
    dict
        A dictionary of xarray DataArrays, with keys corresponding to the variable names.
    
    """
    base_url = f's3://noaa-nws-aorc-v1-1-1km'

    years = get_years(date_pairs)
    g = gpd.GeoSeries([aoi], crs=crs)

    # print(years)
    ds_full = xr.Dataset()
    # print(years)
    # for yr in years: 
    #     single_year_url = f"{base_url}/{yr}.zarr"
    #     ds_yr = xr.open_zarr(fsspec.get_mapper(single_year_url, anon=True), consolidated=True)
    #     if ds_full.data_vars == {}:
    #         ds_full = ds_yr
    #     else:
    #         ds_full = xr.concat([ds_full, ds_yr], dim='time')

    # def fix_aorc_dates(ds):
    # # Filter out any dates from 1970 or any year that doesn't match the file
    # # This keeps only the 'good' data
    #     return ds.sel(time=ds.time.dt.year > np.min(years))
    # # for some reason this stopped working on 3/3
    s3_out = s3fs.S3FileSystem(anon=True)
    fileset = [s3fs.S3Map(
                    root=f"s3://{base_url}/{yr}.zarr", s3=s3_out, check=False
                ) for yr in years]
    # print(fileset)
    ds_full = xr.open_mfdataset(fileset, engine='zarr')

    # print(ds_full)
    # clip to the aoi 
    ds_clip = ds_full.rio.clip(g.geometry.values, crs=crs)
    ds_clip = ds_clip.rename({'latitude': 'y', 'longitude': 'x'})

    ds = xr.Dataset()
    ds_list = []
    # get metrics for each date pair
    for i, (start_date, end_date) in enumerate(date_pairs):
        ds_slice = ds_clip.sel(time=slice(start_date, end_date))

        # ran into some missing data (e.g. 2021/02-2021/11)
        if (len(ds_slice['time']) == 0): 
            print(f'Could not get AORC data for {start_date} to {end_date}.')
            continue

        ds_metrics = get_aorc_metrics(ds_slice)
        pair_name = start_date.strftime('%y%m%d') + "_" + end_date.strftime('%y%m%d')
        ds_metrics['pair'] = pair_name
        ds_metrics = ds_metrics.set_coords('pair')

        ds_list.append(ds_metrics)

        # reproject to reference grid 
        # if ref_grid is not None : 
        #     ds_metrics = ds_metrics.rio.reproject_match(ref_grid)

    ds = xr.concat(ds_list, dim='pair')
    ds = ds.sortby('pair')

    if ref_grid is not None : 
        ds = ds.rio.reproject_match(ref_grid)
    return ds

def get_aorc_metrics(
    aorc_ds,
    # date_pair,
    metrics = 'all'
) -> xr.Dataset:
    """
    Get a list of available AORC metrics. If list_metrics is 'all', returns a 
    list of all available metrics. If list_metrics is a list of metric names, 
    returns a list of the specified metrics. 

    Parameters 
    ----------
    list_metrics: str or list of str
        If 'all', returns a list of all available metrics. If a list of metric 
        names, returns a list of the specified metrics. Default is 'all'.

    Returns
    -------
    list of str
        A list of available AORC metrics, or the specified metrics if 
        list_metrics is a list of metric names.
    
    """

    ds = xr.Dataset()

    valid_metrics = [
        'mean_temp',
        'max_temp',
        'total_precip'
    ]

    if metrics == 'all':
        metrics = valid_metrics
    else:
        for metric in metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Valid metrics are: {valid_metrics}")
    
    if 'mean_temp' in metrics: 
        ds['mean_temp'] = aorc_ds['APCP_surface'].mean(dim='time')
    if 'max_temp' in metrics: 
        ds['max_temp'] = aorc_ds['APCP_surface'].max(dim='time')
    if 'total_precip' in metrics: 
        ds['total_precip'] = aorc_ds['APCP_surface'].sum(dim='time')

    return ds


def get_years(
    date_pairs
): 
    years = set()
    for start_date, end_date in date_pairs:
        years.update([start_date.year, end_date.year])
    return years