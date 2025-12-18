import os
import numpy as np
import xarray as xr
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process WRF NetCDF files into two big NumPy arrays"
    )
    parser.add_argument(
        "-i", "--indir",
        type=str,
        default='/N/slate/kmluong/PROJECT2/level_1_data',
        help="Directory containing input .nc files"
    )
    parser.add_argument(
        "-o", "--outdir",
        type=str,
        default='/N/slate/kmluong/PROJECT2/level_2_data',
        help="Directory where output .npy files will be saved"
    )
    parser.add_argument(
        "-f", "--frames",
        type=int,
        default=5,
        help="Number of consecutive frames to include (default: 2)"
    )
    parser.add_argument(
        "-vl", "--var_levels",
        type=str,
        nargs="+",
        default=[
            "U10m",
            "V10m",
            "SST",
            "LANDMASK",
            "U28",
            "V28",
            "U05",
            "V05",
            "T23",
            "QVAPOR10",
            "PHB10",
        ],
        help=(
            "List of variable-level codes. "
            "Format e.g. U01, V02, QVAPOR03 or PSFC for surface. "
            "This will be parsed into (name, level) pairs."
        )
    )
    parser.add_argument(
        "-p", "--prefix",
        type=str,
        default="wrf_tropical_cyclone_track_{frames}_dataset",
        help="Output filename prefix; may include “{frames}”"
    )
    return parser.parse_args()

def var_extract(
    ds: xr.Dataset,
    var_levels=None,
    frames: int = 2
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build X and Z arrays from a WRF-style Dataset.

    Returns
    -------
    X : np.ndarray
        shape = (ns, nf, nh, nw, nc)
    Z : np.ndarray
        shape = (ns, 4)  # [lon, lat, sin, cos]
    file_year : int
        Year of the first time step in this dataset.
    """
    # default var_levels
    if var_levels is None:
        var_levels = [
            ('U', 1), ('U', 2), ('U', 3),
            ('V', 1), ('V', 2), ('V', 3),
            ('T', 1), ('T', 2), ('T', 3),
            ('QVAPOR', 1), ('QVAPOR', 2), ('QVAPOR', 3),
            ('PSFC', None)
        ]

    # time-of-year features
    t = ds['Time']
    doy     = t.dt.dayofyear.values
    is_leap = t.dt.is_leap_year.values
    yearlen = np.where(is_leap, 366, 365)
    theta   = 2 * np.pi * doy / yearlen
    sinθ, cosθ = np.sin(theta), np.cos(theta)

    lon_arr  = ds['cen_lon'].values
    lat_arr  = ds['cen_lat'].values
    year_arr = t.dt.year.values
    file_year = int(year_arr[0])

    # extract each variable as (Time, y, x)
    var_das, var_names = [], []
    for name, lvl in var_levels:
        da = ds[name]
        if lvl is not None:
            lvl_dim = next(d for d in da.dims if 'bottom_top' in d)
            if name == 'PHB':
                da = da.isel({lvl_dim: lvl}) + ds['PH'].isel({lvl_dim: lvl})
            elif name == 'PH':
                da = da.isel({lvl_dim: lvl}) + ds['PHB'].isel({lvl_dim: lvl})
            else:
                da = da.isel({lvl_dim: lvl})
        var_das.append(da)
        var_names.append(f"{name}_{lvl}" if lvl is not None else name)

    # sample indices: every 4th time, then frames at unit spacing
    n_time = ds.sizes['Time']
    bases = np.arange(0, n_time, 4)
    X_list, Z_list = [], []

    for base in bases:
        idx_hist = base + np.arange(frames)
        if idx_hist.max() >= n_time:
            continue

        # build X: (frames, vars, y, x)
        hist_vars = []
        for da in var_das:
            h = da.isel(Time=idx_hist).rename({'Time': 'frame'})
            hist_vars.append(h)
        sample_X = xr.concat(hist_vars, dim='var')
        sample_X = sample_X.assign_coords(var=var_names)
        sample_X = sample_X.transpose('frame', 'var', 'y', 'x')
        # reset the frame‐coordinate to a simple 0..frames-1 index
        sample_X = sample_X.assign_coords(frame=np.arange(sample_X.sizes['frame']))
        X_list.append(sample_X.expand_dims({'sample': [base]}))

        # build Z: [lon, lat, sin, cos]
        zarr = np.array([lon_arr[base], lat_arr[base], sinθ[base], cosθ[base]])
        sample_Z = xr.DataArray(
            zarr,
            dims=('feature',),
            coords={'feature': ['lon', 'lat', 'sin', 'cos']}
        )
        Z_list.append(sample_Z.expand_dims({'sample': [base]}))

    # concatenate and convert to NumPy
    X = xr.concat(X_list, dim='sample').values  # (ns, nf, nc, nh, nw)
    X = np.transpose(X, (0, 1, 3, 4, 2))        # → (ns, nf, nh, nw, nc)
    Z = xr.concat(Z_list, dim='sample').values  # (ns, 4)

    return X, Z, file_year

def process_data(
    indir: str,
    outdir: str,
    var_levels=None,
    frames: int = 2,
    prefix: str = "wrf_tropical_cyclone_track_{frames}_dataset"
):
    """
    Loop over all .nc files in indir, extract X/Z,
    concatenate into two big arrays and save them.
    """
    os.makedirs(outdir, exist_ok=True)
    X_list, Z_list = [], []

    for fname in os.listdir(indir):
        if not fname.endswith('.nc'):
            continue
        ds = xr.open_dataset(os.path.join(indir, fname))
        n_time = int(ds.sizes['Time'])
        if n_time < frames:
            print(f"Skipping {fname}: Time len {n_time} < frames {frames}")
            continue 
        X, Z, _ = var_extract(ds, var_levels, frames)
        X_list.append(X)
        Z_list.append(Z)

    X_all = np.concatenate(X_list, axis=0)
    Z_all = np.concatenate(Z_list, axis=0)

    # format the prefix with the actual frames value
    out_prefix = prefix.format(frames=frames)
    np.save(os.path.join(outdir, f"{out_prefix}_X.npy"), X_all)
    np.save(os.path.join(outdir, f"{out_prefix}_Z.npy"), Z_all)

if __name__ == "__main__":
    args = parse_args()
    # parse var_levels strings into (name, level) pairs
    var_levels = [
        (v[:-2], int(v[-2:])) if v[-2:].isdigit() else ((v[:-1], None) if v[-1:]=='m' else (v, None))
        for v in args.var_levels
    ]
    process_data(
        indir=args.indir,
        outdir=args.outdir,
        var_levels=var_levels,
        frames=args.frames,
        prefix=args.prefix
    )

