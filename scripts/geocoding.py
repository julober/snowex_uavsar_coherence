import pandas as pd
from scipy.spatial import Delaunay
import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
import time
from uavsar_pytools.convert.tiff_conversion import read_annotation
from pathlib import Path

def interp_weights(xyz, uvw, d=2, save=False, out_fp=None):
    """
    Feturns vertices, weights, and invalid_mask (points outside convex hull).

    This takes a while to run, but once run can be reused for multiple SLCs. 
    
    xyz: (M,2) source points (lon, lat)
    uvw: (N,2) target points (lon, lat)
    """
    tri = Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    invalid = simplex == -1

    # for invalid entries, clip indices to 0 to avoid negative indexing issues in np.take
    simplex_safe = np.where(invalid, 0, simplex)
    vertices = np.take(tri.simplices, simplex_safe, axis=0)
    temp = np.take(tri.transform, simplex_safe, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True))).astype(np.float32)

    # zero out weights for invalid points
    if invalid.any():
        weights[invalid, :] = 0.0

    if save:
        if out_fp is None:
            raise ValueError("out_fp must be provided if save=True")
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        np.savez_compressed(out_fp, vertices=vertices, 
                            weights=weights, invalid=invalid)

    return vertices, weights, invalid

def interpolate_weights(values_flat, vtx, wts, invalid_mask=None):
    """
    values_flat: 1D array with source values length M (flattened slc)
    vtx: (N, d+1) vertex indices
    wts: (N, d+1) barycentric weights
    invalid_mask: boolean array length N; outputs for invalid locations will be set to np.nan
    """
    # values taken at vertices shape (N, d+1)
    vals_at_vertices = np.take(values_flat, vtx)  # shape (N, d+1)
    gridded = np.einsum('nj,nj->n', vals_at_vertices, wts)

    if invalid_mask is not None and invalid_mask.any():
        gridded = gridded.astype(np.complex64) if np.iscomplexobj(values_flat) else gridded.astype(np.float32)
        gridded[invalid_mask] = np.nan

    return gridded

def save_geotiff(mode, data, out_fp, transform, crs='EPSG:4326', nodata=np.nan):
    """
    Save real and imag arrays as a 2-band GeoTIFF.
    real_arr, imag_arr: 2D numpy arrays with same shape (rows, cols)
    out_fp: output filepath
    transform: affine transform from rasterio.transform.from_origin
    crs: CRS string
    nodata: nodata value to write
    """
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    if mode == 'slc' :
        if len(data) != 2 :
            raise ValueError("For 'slc' mode, data must be a tuple of (real_arr, imag_arr)")
        real_arr, imag_arr = data
        dtype = 'float32'
        height, width = real_arr.shape
        with rasterio.open(
            out_fp,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=2,
            dtype=dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(real_arr.astype(dtype), 1)
            dst.write(imag_arr.astype(dtype), 2)
    elif mode == 'lkv' :
        if len(data) != 3 :
            raise ValueError("For 'lkv' mode, data must be a tuple of (east_arr, north_arr, up_arr)")
        east_arr, north_arr, up_arr = data
        dtype = 'float32'
        height, width = east_arr.shape
        # out_fp_e = out_fp.replace('.tif', '_east.tif')
        # out_fp_n = out_fp.replace('.tif', '_north.tif')
        # out_fp_u = out_fp.replace('.tif', '_up.tif')
        with rasterio.open(
            out_fp,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=3,
            dtype=dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(east_arr.astype(dtype), 1)
            dst.write(north_arr.astype(dtype), 2)
            dst.write(up_arr.astype(dtype), 3)
    else :
        raise ValueError("mode must be either 'slc' or 'lkv'")
    return out_fp

def geolocate_uavsar(slc_fp, 
                     ann_fp, 
                     llh_fp, 
                     out_dir, 
                     tgt_rows = None, 
                     tgt_cols = None,
                     vtx = None, 
                     wts = None,
                     invalid = None, 
                     verbose = False) : 
    
    if out_dir is None : 
        out_dir = os.getcwd()

    if verbose :
        print(f"Output directory:{out_dir}")

    # read ann file for rows and columns 
    metadata = read_annotation(ann_fp)
    metadata = pd.DataFrame(metadata).T
    rows, cols = metadata.loc['slc_2_2x8 rows', 'value'], \
                 metadata.loc['slc_2_2x8 columns', 'value']
    if verbose :
        print(f"Data files have {rows} rows and {cols} columns.")

    if tgt_rows is None :
        tgt_rows = rows
        if verbose :
            print(f"Target rows not provided, using source rows: {tgt_rows}")
    if tgt_cols is None :
        tgt_cols = cols
        if verbose :
            print(f"Target columns not provided, using source columns: {tgt_cols}")
    if (tgt_cols * tgt_rows) != vtx.shape[0] or (tgt_cols * tgt_rows) != wts.shape[0] :
        raise ValueError("Provided weights or vertices do not match target grid size. Should be (tgt_rows * tgt_cols).")
    

    # Read llh file
    lats = np.fromfile(llh_fp, 'f4')[0::3]
    lons =  np.fromfile(llh_fp, 'f4')[1::3]
    heights = np.fromfile(llh_fp, 'f4')[2::3]

    src_pts = np.column_stack([lons.ravel(), lats.ravel()])

    # Create target grid
    lat_space = np.linspace(np.min(lats), np.max(lats), num=tgt_rows)
    lon_space = np.linspace(np.min(lons), np.max(lons), num=tgt_cols)
    tgt_lons, tgt_lats = np.meshgrid(lon_space, lat_space, indexing='xy')
    tgt_pts = np.column_stack([tgt_lons.ravel(), tgt_lats.ravel()])

    # Only SLCs and LKVs supported for now. 
    mode = None
    if (Path(slc_fp).suffix != '.slc' and Path(slc_fp).suffix != '.lkv') : 
        raise ValueError("Datafile is not a .slc or a .lkv file")
    elif (Path(slc_fp).suffix == '.slc') : 
        mode = 'slc'
        slc = np.fromfile(slc_fp, '<c8')
    elif (Path(slc_fp).suffix == '.lkv') :
        mode = 'lkv'
        arr = np.fromfile(slc_fp, 'f4').reshape(rows, cols, 3)
        east = arr[..., 0]
        north = arr[..., 1]
        up = arr[..., 2]



    if vtx is None or wts is None or invalid is None : 
        print("Vertices, weights, or invalid mask not given, calculating interpolation weights...")
        # create source points from llh file 
        start_time = time.perf_counter()
        vtx, wts, invalid = interp_weights(src_pts, tgt_pts)
        end_time = time.perf_counter()
        print(f"Weights calculated in {(end_time - start_time / 60):.2f} minutes.")

    # interpolate slc to target grid 
    print(f"Interpolating {slc_fp} to target grid...")
    if mode == 'slc' :
        gridded_flat = interpolate_weights(slc, vtx, wts, invalid_mask=invalid)
        gridded = gridded_flat.reshape(tgt_lons.shape)
    else :  # mode == 'lkv'
        gridded_flat_e = interpolate_weights(east, vtx, wts, invalid_mask=invalid)
        gridded_flat_n = interpolate_weights(north, vtx, wts, invalid_mask=invalid)
        gridded_flat_u = interpolate_weights(up, vtx, wts, invalid_mask=invalid)
        gridded = np.stack((gridded_flat_e.reshape(tgt_lons.shape),
                            gridded_flat_n.reshape(tgt_lons.shape),
                            gridded_flat_u.reshape(tgt_lons.shape)), axis=0)

    # build a geotransform for rasterio; careful: from_origin expects (west, north, xres, yres)
    xres = lon_space[1] - lon_space[0]
    yres = lat_space[1] - lat_space[0]
    west = lon_space.min()
    north = lat_space.max()
    transform = from_origin(west, north, xres, yres)

    if mode == 'slc' :
        new_fname = os.path.basename(slc_fp).replace('.slc', '_geoslc.tif')
        out_fp = os.path.join(out_dir, new_fname)
        
        real_out = np.real(gridded).astype('float32')
        imag_out = np.imag(gridded).astype('float32')
        real_out = np.flipud(np.real(gridded)).astype('float32')
        imag_out = np.flipud(np.imag(gridded)).astype('float32')
        save_geotiff('slc', (real_out, imag_out), out_fp, transform, crs='EPSG:4326')
        print(f'Saved two-band GeoTIFF to {out_fp}')
    elif mode == 'lkv' :
        new_fname = os.path.basename(slc_fp).replace('.lkv', '_geolkv.tif')
        out_fp = os.path.join(out_dir, new_fname)

        e_out = np.flipud(gridded[0].astype('float32'))
        n_out = np.flipud(gridded[1].astype('float32'))
        u_out = np.flipud(gridded[2].astype('float32'))
        save_geotiff('lkv', (e_out, n_out, u_out), out_fp, transform, crs='EPSG:4326')
        print(f'Saved three-band GeoTIFF to {out_fp}')

    return gridded, vtx, wts
