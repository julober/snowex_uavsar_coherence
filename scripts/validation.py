import warnings 
from datetime import date
from datetime import datetime 

import numpy as np
import pandas as pd
# import xarray as xr

def validate_date_pairs(
    dates,
    *kwargs
) -> list:
    """
    Validate and normalize a date input.
    Checks: 
    - validates each individual date using validate_date
    - checks that first date is before the second date in time. 

    Parameters
    ----------
    date_pairs : a list of date pairs. There should be two dates at each index 
        and each pair should have a beginning date first, then an ending date. 

    *kwags : optional arguments to pass along to validate_date()
    """

    # ndates = len(dates)
    ds = list()

    for idx, pair in enumerate(dates) : 
        # check that there are two dates 
        if len(pair) != 2 : 
            raise ValueError(f"Dates at index {idx} not a pair: {pair}")
        date1 = pair[0]
        date2 = pair[1]

        date1 = validate_date(date1,
                              *kwargs)
        date2 = validate_date(date2,
                              *kwargs)

        # check that dates are in the right order 
        if date2 <= date1 : 
            raise ValueError(f"Dates at index {idx} not in order: {date1}, {date2}")
        
        ds.append((date1, date2))

     # sanity check that we didn't lose anything 
    if len(dates) != len(ds) :
        raise ValueError(f"Number of dates given {len(dates)} doesn't match number of dates validated {len(ds)}")
    
    return ds 

def validate_date(
    date: str | pd.Timestamp | np.datetime64 | datetime,
    strip_timezone: bool = True,
    timezone: str | None = None,
    allow_future: bool = False,
    param_name: str = "date",
) -> pd.Timestamp:
    """
    Validate and normalize a date input.
    This function is taken directly from https://github.com/ZachHoppinen/sarvalanche/
    on 2/24/2026 

    Parameters
    ----------
    date : str | pd.Timestamp | np.datetime64 | datetime
        Date to validate. Can be string (ISO format), pandas Timestamp,
        numpy datetime64, or Python datetime.
    strip_timezone : bool, default=True
        If True, remove timezone information from the result.
        If False and timezone is None, preserve original timezone.
    timezone : str | None, default=None
        If provided, localize naive datetime or convert aware datetime
        to this timezone (e.g., 'UTC', 'US/Pacific').
        Ignored if strip_timezone is True.
    allow_future : bool, default=False
        If True, allow dates in the future. If False, raise error
        for dates after the current time.
    param_name : str, default='date'
        Name of parameter for error messages.

    Returns
    -------
    pd.Timestamp
        Validated and normalized timestamp.

    Raises
    ------
    ValueError
        If date cannot be parsed, is in the future (when allow_future=False),
        or is invalid.
    TypeError
        If date is of unsupported type.

    Examples
    --------
    >>> # Basic validation
    >>> validate_date('2024-03-15')
    Timestamp('2024-03-15 00:00:00')

    >>> # With timezone
    >>> validate_date('2024-03-15', strip_timezone=False, timezone='UTC')
    Timestamp('2024-03-15 00:00:00+0000', tz='UTC')

    >>> # Future date check
    >>> validate_date('2030-01-01')  # Raises ValueError

    >>> # Allow future dates
    >>> validate_date('2030-01-01', allow_future=True)
    Timestamp('2030-01-01 00:00:00')
    """

    # Try to convert to pandas Timestamp
    try:
        if isinstance(date, pd.Timestamp):
            ts = date
        elif isinstance(date, np.datetime64):
            ts = pd.Timestamp(date)
        elif isinstance(date, datetime):
            ts = pd.Timestamp(date)
        elif isinstance(date, str):
            ts = pd.to_datetime(date)
        else:
            raise TypeError(
                f"{param_name} must be string, pd.Timestamp, np.datetime64, "
                f"or datetime, got {type(date)}"
            )
    except (ValueError, pd.errors.ParserError) as e:
        raise ValueError(
            f"Could not parse {param_name}='{date}' as a valid date. "
            f"Error: {e}"
        ) from e

    # Check for NaT (Not a Time)
    if pd.isna(ts):
        raise ValueError(f"{param_name} is NaT (Not a Time)")

    # Handle timezone
    if strip_timezone:
        # Remove timezone info
        ts = ts.tz_localize(None) if ts.tz is not None else ts
    elif timezone is not None:
        # Apply requested timezone
        if ts.tz is None:
            # Naive datetime - localize
            ts = ts.tz_localize(timezone)
        else:
            # Already has timezone - convert
            ts = ts.tz_convert(timezone)

    # Check if date is in the future
    if not allow_future:
        now = pd.Timestamp.now(tz=ts.tz)
        if ts > now:
            raise ValueError(
                f"{param_name}='{ts}' is in the future. "
                f"Current time: {now}. "
                f"Set allow_future=True to allow future dates."
            )

    return ts

import numpy as np
from shapely.geometry import Polygon, Point, box
from shapely.geometry.base import BaseGeometry

def validate_aoi(aoi):
    """
    Validate and normalize an AOI.
    This function is taken directly from https://github.com/ZachHoppinen/sarvalanche/
    on 2/24/2026 

    Returns
    -------
    shapely.geometry.BaseGeometry
        Polygon or Point geometry.
    """

    geom = None

    # ---- Shapely geometry ----
    if isinstance(aoi, BaseGeometry):
        geom = aoi

    # ---- Iterable ----
    elif isinstance(aoi, (list, tuple, np.ndarray)):
        if len(aoi) == 4:
            xmin, ymin, xmax, ymax = map(float, aoi)
            xmin, xmax = sorted((xmin, xmax))
            ymin, ymax = sorted((ymin, ymax))
            geom = box(xmin, ymin, xmax, ymax)

        elif len(aoi) == 2:
            x, y = map(float, aoi)
            geom = Point(x, y)

    # ---- Dict ----
    elif isinstance(aoi, dict):
        key_sets = [
            ("xmin", "ymin", "xmax", "ymax"),
            ("west", "south", "east", "north"),
            ("minx", "miny", "maxx", "maxy"),
        ]

        for keys in key_sets:
            if all(k in aoi for k in keys):
                xmin, ymin, xmax, ymax = (float(aoi[k]) for k in keys)
                xmin, xmax = sorted((xmin, xmax))
                ymin, ymax = sorted((ymin, ymax))
                geom = box(xmin, ymin, xmax, ymax)
                break

        if geom is None:
            raise ValueError(
                f"AOI dict keys not recognized: {list(aoi.keys())}. "
                f"Expected one of: {key_sets}"
            )

    else:
        raise TypeError(
            f"AOI must be geometry, iterable, or dict; got {type(aoi)}"
        )

    # ---- Geometry sanity checks ----
    if geom.is_empty:
        raise ValueError("AOI geometry is empty")

    if isinstance(geom, Polygon) and geom.area == 0:
        raise ValueError("AOI polygon has zero area")

    return geom

import xarray as xr
import rioxarray
from rasterio.transform import from_bounds

def make_reference_grid(
    *,
    aoi,
    crs,
    resolution,
    dtype="float32",
    fill_value=np.nan,
    name="reference",
):
    """
    Create an xarray DataArray usable as a reprojection reference grid.
    This function is taken directly from https://github.com/ZachHoppinen/sarvalanche/
    on 2/24/2026 

    Parameters
    ----------
    aoi : shapely.Polygon
        (minx, miny, maxx, maxy) in target CRS
    crs : str or CRS
        Target CRS (e.g. "EPSG:32611")
    resolution : float or (float, float)
        Pixel size in CRS units
    """

    minx, miny, maxx, maxy = aoi.bounds

    if isinstance(resolution, (int, float)):
        xres = yres = float(resolution)
    else:
        xres, yres = map(float, resolution)

    # Number of pixels
    width = int(np.ceil((maxx - minx) / xres))
    height = int(np.ceil((maxy - miny) / yres))

    # Affine transform (north-up)
    transform = from_bounds(
        minx, miny, minx + width * xres, miny + height * yres,
        width, height
    )

    # Pixel-centered coordinates
    x = minx + (np.arange(width) + 0.5) * xres
    y = maxy - (np.arange(height) + 0.5) * yres

    data = np.full((height, width), fill_value, dtype=dtype)

    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"x": x, "y": y},
        name=name,
    )

    da = da.rio.write_crs(crs)
    da = da.rio.write_transform(transform)

    return da

import xarray as xr

def validate_alignment(
    datasets, 
    coord_names=['y', 'x']
):
    """
    Checks if a list of xarray objects share identical coordinate values.
    
    Args:
        datasets (list): List of xr.Dataset or xr.DataArray objects.
        coord_names (list): The specific coordinates to verify (default: lat/lon).
        
    Returns:
        bool: True if all match perfectly, False otherwise.
    """
    errors = []
    
    # 1. Check if all datasets actually contain the required coordinates
    for i, ds in enumerate(datasets):
        missing = [c for c in coord_names if c not in ds.coords]
        if missing:
            errors.append(f"Dataset [{i}] is missing coordinates: {missing}")

    if errors:
        for error in errors:
            print(error)
        return False

    # check that datasets with pair dimensions have the same values
    for i, ds in enumerate(datasets):
        if 'pair' in ds.dims:
            pair_values = ds['pair'].values
            for j, other_ds in enumerate(datasets):
                if i != j and 'pair' in other_ds.dims:
                    other_pair_values = other_ds['pair'].values
                    if not np.array_equal(pair_values, other_pair_values):
                        errors.append(f"Dataset [{i}] has different 'pair' values than Dataset [{j}].")

    # 2. Try to align them using 'exact' join
    try:
        # If they don't match perfectly, xarray raises a ValueError
        xr.align(*datasets, join="exact")
        # print("Validation Passed: All grids match perfectly.")
        return True
        
    except ValueError as e:
        print("Grid mismatch detected.")
        print("-" * 55)
        
        # 3. Generate the Overview Output
        header = f"{'Index':<7} | {'Name':<15} | " + " | ".join([f"{c:<10}" for c in coord_names])
        print(header)
        print("-" * 55)
        
        for i, ds in enumerate(datasets):
            # Get the size of the requested coordinates
            dims_str = " | ".join([f"{ds[c].size:<10}" for c in coord_names if c in ds.coords])
            name = getattr(ds, 'name', None) or f'Dataset_{i}'
            print(f"{i:<7} | {str(name)[:15]:<15} | {dims_str}")
            
        print("-" * 55)
        print(f"Error Detail: {e}")
        return False
    