def get_3dep_dem(geometry,
                 out_fp,
                 crs_dest='EPSG:26911',
                 res_dest=30) :
    """
    Get DEM layer. Saves the layer as a tif to the specified output filepath
    if it doesn't already exists, returns the reprojected layer as a rasterio 
    object. 
    
    Parameters 
    ----------
    geometry: geopandas.GeoDataFrame
        A GeoDataFrame geometry in EPSG:4326 (WGS84) 
    out_fp: str
        A filepath to save the output tif to
    crs_dest: str, optional
        The target CRS for reprojecting the DEM layer. Default is 
        'EPSG:26911' (UTM 11N).
    res_dest: int, optional
        The target resolution for reprojecting the DEM layer. Default is
        30m.
    """