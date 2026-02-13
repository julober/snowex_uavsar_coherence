def get_nlcd_layers(geometry, 
                    out_fp,
                    years={'cover': [2019], 'canopy': [2019]},
                    crs_dest='EPSG:26911',
                    res_dest=30) :
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
    if not isinstance(geometry, gpd.GeoDataFrame):
        raise ValueError("Input geometry must be a GeoDataFrame.")
    if not out_fp:
        raise ValueError("Output file path must be provided.")
    if geometry.crs != 'EPSG:4326':
        raise ValueError("Input geometry must be in EPSG:4326 (WGS84).")
    
    # get nlcd
    nlcd_layers = gh.nlcd_bygeom(geometry=geometry,
                                years=years)
    
    nlcd_reproj = {}
    for key, value in nlcd_layers.items():
        nlcd_reproj[key] = value.rio.reproject(crs_dest, resolution=res_dest)

    # save as tifs 
    for key, value in nlcd_reproj.items():
        # check that file doesn't already exist 
        if os.path.exists(f"{out_fp}_{key}.tif"):
            raise ValueError(f"File {out_fp}_{key}.tif already exists.")
        else:
            value.rio.to_raster(f"{out_fp}_{key}.tif")

    # will download 2019 automatically - should I use 2021 instead? 
    return nlcd_reproj

# big ugly function to get a colormap for the land cover types  
def get_landcover_cmap() : 
    """
    Generate a colormap for the NLCD land cover types. Returns the colormap, 
    normalization, bounds, and a list of names for the land cover classes. 
    """
    # from https://colab.research.google.com/github/gee-community/geeViz/blob/master/examples/Annual_NLCD_Viewer_Notebook.ipynb#scrollTo=a588e083

    import matplotlib.colors as mcolors
    from matplotlib.colors import ListedColormap
    import numpy as np

    land_cover_vis = {
        'LC_class_values': [11, 12, 21, 22, 23, 24, 31, 
                            41, 42, 43, 52, 71, 81, 82, 90, 95],
        'LC_class_palette' : ['466b9f', 'd1def8', 'dec5c5', 'd99282', 
                              'eb0000', 'ab0000', 'b3ac9f', '68ab5f', 
                              '1c5f2c', 'b5c58f', 'ccb879', 'dfdfc2', 
                              'dcd939', 'ab6c28', 'b8d9eb', '6c9fb8'],
        'LC_class_names' : ["Open Water", "Perennial Ice/Snow", 
                            "Developed, Open Space", 
                            "Developed, Low Intensity", 
                            "Developed, Medium Intensity", 
                            "Developed, High Intensity", "Barren Land", 
                            "Deciduous Forest", "Evergreen Forest", 
                            "Mixed Forest", "Shrub/Scrub", 
                            "Grassland/Herbaceous", "Pasture/Hay", 
                            "Cultivated Crops", "Woody Wetlands", 
                            "Emergent Herbaceous Wetlands"]
        }
    col_dict = {land_cover_vis['LC_class_values'][i]: '#' + land_cover_vis['LC_class_palette'][i] for i in range(len(land_cover_vis['LC_class_values']))}

    ncolors = len(col_dict.keys())
    cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
    np.random.seed(2112)
    lu_index = np.zeros((ncolors, ncolors))
    for nn,k in enumerate(col_dict.keys()):
        lu_index[nn, :] = k

    bounds = np.array([k for k in col_dict.keys()])
    bounds = bounds[:-1] + np.diff(bounds) / 2
    bounds = np.concatenate((np.array([0]), bounds, np.array([97])))
    norm = mcolors.BoundaryNorm(bounds, ncolors)
    return cm, norm, bounds, land_cover_vis['LC_class_names']