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
    metrics = [ # not currently using! 
        'mean_temp',
        'max_temp',
        'total_posdeg',
        'temp_diff',
        'temp_diff_acq',
        'total_precip',
        'total_rain',
        'total_snow',
        'acq_day_precip',
        'mean_wind',
        'max_wind',
        'hours_blowing_snow'
    ]
    aorc = get_aorc_layers(aoi=aoi, 
                           date_pairs=date_pairs, 
                           crs=crs, 
                           ref_grid=ref)
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
    metrics = 'all',
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
                                      metrics=metrics)
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
        'swe_accum',
        'swe_ablate',
        'density_change',
        # 'snow_status_change'
    ]

    if metrics == 'all':
        metrics = valid_metrics
    else:
        for metric in metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Valid metrics are: {valid_metrics}")

    # Calculate day-to-day changes
    swe_diff = ds['SWE_Post'].diff(dim='time')
    sd_diff = ds['SD_Post'].diff(dim='time')

    if 'swe_accum' in metrics:
        # Sum of only the positive daily SWE changes (new snow mass)
        out['swe_accum'] = swe_diff.where(swe_diff > 0, other=0).sum(dim='time')
        
    if 'swe_ablate' in metrics:
        # Sum of only the negative daily SWE changes (melt/sublimation mass)
        # Note: Taking the absolute value makes it easier for the ML model to interpret
        out['swe_ablate'] = abs(swe_diff.where(swe_diff < 0, other=0).sum(dim='time'))

    if 'density_change' in metrics:
        # Bulk Density = SWE / Snow Depth (handle division by zero where snow is absent)
        # Assuming SWE is in meters and SD is in meters, this yields a ratio.
        # If units differ (e.g., mm and meters), the ratio is still a valid ML feature.
        
        # Reference day density
        dens_ref = ds['SWE_Post'].isel(time=0) / ds['SD_Post'].isel(time=0).where(ds['SD_Post'].isel(time=0) > 0)
        
        # Secondary day density
        dens_sec = ds['SWE_Post'].isel(time=-1) / ds['SD_Post'].isel(time=-1).where(ds['SD_Post'].isel(time=-1) > 0)
        
        # Change in density over the 12 days. Fill NaNs with 0 (no snow = no density change)
        out['density_change'] = (dens_sec - dens_ref).fillna(0)

    if 'snow_status_change' in metrics:
        # 1 if snow appeared, -1 if snow melted completely, 0 if no change in presence/absence
        # Convert to boolean (has snow > 0), then cast to integer to subtract
        has_snow_ref = (ds['SWE_Post'].isel(time=0) > 0).astype(int)
        has_snow_sec = (ds['SWE_Post'].isel(time=-1) > 0).astype(int)
        
        out['snow_status_change'] = has_snow_sec - has_snow_ref

    # if 'swe_change' in metrics: 
    #     # first minus last 
    #     out['swe_change'] = ds['SWE_Post'].isel(time=-1) - ds['SWE_Post'].isel(time=0)
    # if 'sd_change' in metrics: 
    #     # first minus last 
    #     out['sd_change'] = ds['SD_Post'].isel(time=-1) - ds['SD_Post'].isel(time=0)
    # # if 'snow_cover_change' in metrics: 
    # #     out['snow_cover_change'] = ds['SCA_Post']

    return out

import re 
import pandas as pd

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
    ref_grid = None,
    metrics = 'all'
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

        ds_metrics = get_aorc_metrics(ds_slice, metrics=metrics)
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
        'total_posdeg',
        'temp_diff',
        'temp_diff_acq',
        'freeze_thaw_cycles',
        'diurnal_temp_range',
        'total_precip',
        'total_rain',
        'total_snow',
        'acq_day_precip',
        'mean_wind',
        'max_wind',
        'hours_blowing_snow'
    ]

    if metrics == 'all':
        metrics = valid_metrics
    else:
        for metric in metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Valid metrics are: {valid_metrics}")
    
    # temperature metrics 
    temp = aorc_ds['TMP_2maboveground'] - 273.15 # convert from K to C
    if 'mean_temp' in metrics: 
        ds['mean_temp'] = temp.mean(dim='time')
    if 'max_temp' in metrics: 
        ds['max_temp'] = temp.max(dim='time')
    if 'total_posdeg' in metrics: 
        ds['total_posdeg'] = (
        temp
        .where(temp > 0, other=0)
        .sum(dim='time')
    )
    if 'temp_diff' in metrics: 
        ds['temp_diff'] = temp.isel(time=-1) - temp.isel(time=0)
    if 'temp_diff_acq' in metrics:
        # Absolute temperature difference between reference (time=0) and secondary (time=-1) acquisitions
        ds['temp_diff_acq'] = abs(temp.isel(time=0) - temp.isel(time=-1))
    if 'freeze_thaw_cycles' in metrics:
        # Count how many times the temperature crosses the 0°C threshold
        # We convert to boolean (True if > 0), then use .diff() to find where the boolean state changes
        is_above_freezing = temp > 0
        ds['freeze_thaw_cycles'] = (is_above_freezing.astype(int).diff(dim='time') != 0).sum(dim='time')
        # filter out anything above 100 cycles
        ds['freeze_thaw_cycles'] = ds['freeze_thaw_cycles'].where(ds['freeze_thaw_cycles'] <= 100, other=np.nan)
        # filter out anything negative 
        ds['freeze_thaw_cycles'] = ds['freeze_thaw_cycles'].where(ds['freeze_thaw_cycles'] >= 0, other=np.nan)
    if 'diurnal_temp_range' in metrics:
        # Resample to daily max and min, subtract to get daily range, then average over the 12 days
        # (Assuming your time coordinate is recognized as datetime by xarray)
        daily_max = temp.resample(time='1D').max()
        daily_min = temp.resample(time='1D').min()
        ds['diurnal_temp_range'] = (daily_max - daily_min).mean(dim='time')
    
    # precip metrics 
    precip = aorc_ds['APCP_surface']
    if 'total_precip' in metrics:
        ds['total_precip'] = precip.sum(dim='time')
    if 'total_rain' in metrics:
        # Sum of precipitation only when temperature is above 0°C
        # other=0 ensures we don't introduce NaNs that break the sum
        ds['total_rain'] = precip.where(temp > 0, other=0).sum(dim='time')
    if 'total_snow' in metrics:
        # Sum of precipitation only when temperature is at or below 0°C
        ds['total_snow'] = precip.where(temp <= 0, other=0).sum(dim='time')
        
    if 'acq_day_precip' in metrics:
        # Sum of precipitation strictly on the reference acquisition day and secondary acquisition day
        ds['acq_day_precip'] = precip.isel(time=0) + precip.isel(time=-1)

    # wind metrics 
    wind_speed = np.sqrt(aorc_ds['UGRD_10maboveground']**2 + aorc_ds['VGRD_10maboveground']**2)

    if 'hours_blowing_snow' in metrics:
        # Count the number of time steps (usually hours in AORC) where wind exceeds 6 m/s
        # 6 m/s is a standard rule-of-thumb threshold for dry snow transport
        ds['hours_blowing_snow'] = (wind_speed > 6.0).sum(dim='time')
        # filter out negative and anything above 700 hours 
        ds['hours_blowing_snow'] = ds['hours_blowing_snow'].where((ds['hours_blowing_snow'] >= 0) & (ds['hours_blowing_snow'] <= 700), other=np.nan)
    if 'mean_wind' in metrics: 
        ds['mean_wind'] = np.sqrt(aorc_ds['UGRD_10maboveground']**2 + aorc_ds['VGRD_10maboveground']**2).mean(dim='time')
    if 'max_wind' in metrics: 
        ds['max_wind'] = np.sqrt(aorc_ds['UGRD_10maboveground']**2 + aorc_ds['VGRD_10maboveground']**2).max(dim='time')

    return ds


def get_years(
    date_pairs
): 
    years = set()
    for start_date, end_date in date_pairs:
        years.update([start_date.year, end_date.year])
    return years