import shapely
from shapely.geometry import Polygon
import geopandas as gpd
import py3dep

import numpy as np
import xarray as xr
import rioxarray 

# i don't think this will work - figure it out
from validation import validate_aoi, validate_date_pairs, make_reference_grid
# from nlcd import get_nlcd_layers

def assemble_data(
    aoi, 
    date_pairs, 
    fp_dest,
    crs = 'EPSG:4326', 
    resolution = 30,
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
    if crs == 'EPSG:4326' and resolution == 30:
        res_deg = 0.00381807117 # 30 m at 45 deg latitude 
    # 
    ref = make_reference_grid(aoi=aoi, crs=crs, resolution=res_deg)

    data = xr.Dataset()

    # 3. Get NLCD layers 
    nlcd = get_nlcd_layers(aoi=aoi_gdf, ref_grid=ref)
    print(f"Got NLCD: {type(nlcd)}")

    # 4. Get DEM 
    print(f'Making call to py3dep.get_map with resolution {resolution} in m.')
    dem = py3dep.get_dem(geometry=aoi, resolution=resolution, crs=crs)
    dem.rio.write_crs(crs)
    dem = dem.rio.reproject_match(ref)
    print(f"Got DEM: {type(dem)}")
    topo = get_topo_layers(dem=dem, ref_grid=ref)

    
    # 5. Loop through date pairs
    # for idx, (start_date, end_date) in enumerate(date_pairs) :
    # 5a. Get AORC for each range 
    aorc = get_aorc_layers(aoi=aoi, date_pairs=date_pairs, crs=crs, ref_grid=ref)
    # print(f"Got AORC: {type(aorc)}")

    # 5b. Get UCLA for each range 

    # snow = get_snow_layers(aoi=aoi, date_pairs=date_pairs, ref_grid=ref)
    # print(f"Got snow: {type(snow)}")

    static = xr.merge([nlcd, dem, topo])
    # dynamic = xr.merge([aorc, snow])
    return static#, dynamic
    


def get_nlcd_layers(
    aoi, 
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

    # check that geometry is a GeoDataFrame and has a CRS 
    if not isinstance(aoi, gpd.GeoDataFrame):
        raise ValueError("Input geometry must be a GeoDataFrame.")
    # if not out_fp:
    #     raise ValueError("Output file path must be provided.")
    if aoi.crs != 'EPSG:4326':
        raise ValueError("Input geometry must be in EPSG:4326 (WGS84).")
    
    # get nlcd
    nlcd_layers = gh.nlcd_bygeom(geometry=aoi,
                                 years=years)[0]

    nlcd_reproj = {}
    if ref_grid is not None : 
        nlcd_reproj = nlcd_layers.rio.reproject_match(ref_grid)
        return nlcd_reproj

    # save as tifs 
    # for key, value in nlcd_reproj.items():
    #     # check that file doesn't already exist 
    #     if os.path.exists(f"{out_fp}_{key}.tif"):
    #         raise ValueError(f"File {out_fp}_{key}.tif already exists.")
    #     else:
    #         value.rio.to_raster(f"{out_fp}_{key}.tif")

    # will download 2019 automatically - should I use 2021 instead? 
    return nlcd_layers

import xrspatial as xrs

def get_topo_layers(
    dem,
    ref_grid = None
):
    ds = {}

    if dem.rio.crs == 'EPSG:4326':
        dem_reproj = dem.rio.reproject('EPSG:5070')
    else:
        dem_reproj = dem

    ds['slope'] = xrs.slope(dem_reproj)
    ds['aspect'] = xrs.aspect(dem_reproj)
    ds['curve'] = xrs.curvature(dem_reproj)

    if ref_grid is not None : 
            for key, value in ds.items():
                ds[key] = value.rio.reproject_match(ref_grid)

    return ds

import earthaccess

def get_snow_layers(
    aoi, 
    date_pairs,
    ref_grid = None
):
    # turn aoi into list of points in counter-clockwise order 
    xx, yy = aoi.exterior.coords.xy
    x = xx.tolist()
    y = yy.tolist()

    years = get_years(date_pairs)

    grans = earthaccess.search_data(
        short_name='WUS_UCLA_SR', 
        cloud_hosted=True, 
        temporal=(years[0], years[-1]),
        polygon=zip(x,y)
    )

    fileset = earthaccess.open(grans)
    ds = xr.open_mfdataset(fileset, chunks={})


def get_snow_metrics(
    ds,
    metrics = 'all'
) -> dict:
    
    out = {}

    valid_metrics = [
        'total_swe',
        'max_swe',
        'snow_cover_change'
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
    if 'snow_cover_change' in metrics: 
        out['snow_cover_change'] = ds['SCA_Post']

    return out


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

    s3_out = s3fs.S3FileSystem(anon=True)
    # print(years)
    fileset = [s3fs.S3Map(
                    root=f"s3://{base_url}/{yr}.zarr", s3=s3_out, check=False
                ) for yr in years]
    ds_full = xr.open_mfdataset(fileset, engine='zarr')

    # clip to the aoi 
    ds_clip = ds_full.rio.clip(aoi.geometry.values, crs=crs)

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
        pair_name = start_date.strftime('%Y%m%d') + "_" + end_date.strftime('%Y%m%d')
        ds_metrics['pair'] = pair_name

        ds_list.append(ds_metrics)

        # reproject to reference grid 
        if ref_grid is not None : 
            ds_metrics = ds_metrics.rio.reproject_match(ref_grid)

    ds = xr.concat(ds_list, dim='pair')
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
        ds['mean_temp'] = aorc_ds['TMP_2maboveground'].mean(dim='time')
    if 'max_temp' in metrics: 
        ds['max_temp'] = aorc_ds['TMP_2maboveground'].max(dim='time')
    if 'total_precip' in metrics: 
        ds['total_precip'] = aorc_ds['PRECTOTCORR'].sum(dim='time')

    return ds


def get_years(
    date_pairs
): 
    years = set()
    for start_date, end_date in date_pairs:
        years.update([start_date.year, end_date.year])
    return years