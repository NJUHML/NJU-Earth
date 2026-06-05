import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
PRESSURE_ORDER = ["z", "rh", "t", "u", "v"]
SINGLE_ORDER = ["u10", "v10", "t2m", "msl"]

MODEL_CHANNELS = [
    *(f"{var}_{level}" for level in PRESSURE_LEVELS for var in PRESSURE_ORDER),
    *SINGLE_ORDER,
]

PRESSURE_VAR_ALIASES = {
    "z": ["z", "geopotential"],
    "rh": ["r", "rh", "relative_humidity"],
    "t": ["t", "temperature"],
    "u": ["u", "u_component_of_wind"],
    "v": ["v", "v_component_of_wind"],
}
SINGLE_VAR_ALIASES = {
    "u10": ["u10", "10m_u_component_of_wind"],
    "v10": ["v10", "10m_v_component_of_wind"],
    "t2m": ["t2m", "2m_temperature"],
    "msl": ["msl", "mean_sea_level_pressure"],
}


def import_xarray():
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise RuntimeError("xarray is required to convert ERA5 NetCDF files to npy.") from exc
    return xr


def find_coord(obj, candidates):
    for name in candidates:
        if name in obj.coords or name in obj.dims:
            return name
    raise KeyError(f"None of these coordinates were found: {candidates}")


def resolve_var(ds, aliases, label):
    lower_to_name = {name.lower(): name for name in ds.data_vars}
    for alias in aliases:
        name = lower_to_name.get(alias.lower())
        if name is not None:
            return ds[name]

    wanted = {alias.lower().replace(" ", "_") for alias in aliases}
    for name, data_array in ds.data_vars.items():
        attrs = {
            str(data_array.attrs.get("short_name", "")).lower().replace(" ", "_"),
            str(data_array.attrs.get("long_name", "")).lower().replace(" ", "_"),
            str(data_array.attrs.get("standard_name", "")).lower().replace(" ", "_"),
        }
        if attrs & wanted:
            return ds[name]

    if label == "rh":
        raise KeyError("Relative humidity was not found. This model expects RH.")
    raise KeyError(f"Variable {label} was not found. Tried aliases: {aliases}")


def select_time(ds, timestamp):
    time_name = find_coord(ds, ["valid_time", "time"])
    values = np.asarray(ds[time_name].values).astype("datetime64[ns]")
    target = np.datetime64(pd.to_datetime(timestamp).to_datetime64()).astype("datetime64[ns]")
    matches = np.where(values == target)[0]
    if len(matches) == 0:
        available = ", ".join(str(value) for value in values[:5])
        raise KeyError(f"Time {target} was not found in {time_name}. First available values: {available}")
    return ds.isel({time_name: int(matches[0])})


def select_level(data_array, level):
    level_name = find_coord(data_array, ["pressure_level", "level", "isobaricInhPa", "isobaricInPa"])
    values = np.asarray(data_array[level_name].values, dtype=np.float64)
    target = float(level)

    distances = np.abs(values - target)
    index = int(np.argmin(distances))
    if distances[index] > 1e-3:
        target_pa = target * 100.0
        distances_pa = np.abs(values - target_pa)
        index_pa = int(np.argmin(distances_pa))
        if distances_pa[index_pa] <= 1e-3:
            index = index_pa
        else:
            raise KeyError(f"Pressure level {level} hPa was not found. Available levels: {values.tolist()}")

    return data_array.isel({level_name: index})


def field_to_numpy(data_array, expected_shape):
    lat_name = find_coord(data_array, ["latitude", "lat"])
    lon_name = find_coord(data_array, ["longitude", "lon"])
    dims = list(data_array.dims)
    array = np.asarray(data_array.values, dtype=np.float32)

    for axis in range(len(dims) - 1, -1, -1):
        dim = dims[axis]
        if dim in {lat_name, lon_name}:
            continue
        if dim == "expver":
            array = np.nanmax(array, axis=axis)
            dims.pop(axis)
        elif array.shape[axis] == 1:
            array = np.take(array, indices=0, axis=axis)
            dims.pop(axis)
        else:
            raise ValueError(f"Unexpected extra dimension {dim} with size {array.shape[axis]}")

    lat_axis = dims.index(lat_name)
    lon_axis = dims.index(lon_name)
    array = np.moveaxis(array, [lat_axis, lon_axis], [0, 1])
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D latitude-longitude field after squeezing, got shape {array.shape}")

    lat_values = np.asarray(data_array[lat_name].values, dtype=np.float64)
    if lat_values[0] < lat_values[-1]:
        array = array[::-1, :]

    lon_values = np.asarray(data_array[lon_name].values, dtype=np.float64)
    wrapped_lon = np.mod(lon_values, 360.0)
    lon_order = np.argsort(wrapped_lon)
    array = array[:, lon_order]

    if array.shape != tuple(expected_shape):
        raise ValueError(f"Expected field shape {tuple(expected_shape)}, got {array.shape}")
    if np.isnan(array).any():
        raise ValueError("NaN values remain after ERA5 field extraction")
    return array


def resolve_era5_file(root, kind, timestamp):
    root = Path(root)
    timestamp = pd.to_datetime(timestamp)
    year = f"{timestamp.year:04d}"
    ymdh = timestamp.strftime("%Y%m%d_%H")
    names = [
        f"era5_{kind}_{ymdh}.nc",
        f"era5_{kind}_{ymdh}.netcdf",
        f"era5_{kind}_{ymdh}.grib",
        f"{ymdh}.nc",
        f"{ymdh}.netcdf",
        f"{ymdh}.grib",
    ]
    parents = [
        root / kind / year,
        root / kind,
        root / year,
        root,
    ]

    for parent in parents:
        for name in names:
            path = parent / name
            if path.exists():
                return path

    tried = [str(parent / names[0]) for parent in parents]
    raise FileNotFoundError(f"Could not find {kind} ERA5 file for {ymdh}. Tried examples: {tried}")


def open_dataset(path, cache):
    path = Path(path)
    if path not in cache:
        xr = import_xarray()
        cache[path] = xr.open_dataset(path)
    return cache[path]


def build_array(pressure_ds, single_ds, timestamp, expected_shape):
    pressure_step = select_time(pressure_ds, timestamp)
    single_step = select_time(single_ds, timestamp)

    fields = []
    for level in PRESSURE_LEVELS:
        for var in PRESSURE_ORDER:
            data_array = resolve_var(pressure_step, PRESSURE_VAR_ALIASES[var], var)
            fields.append(field_to_numpy(select_level(data_array, level), expected_shape))

    for var in SINGLE_ORDER:
        data_array = resolve_var(single_step, SINGLE_VAR_ALIASES[var], var)
        fields.append(field_to_numpy(data_array, expected_shape))

    return np.stack(fields, axis=0).astype(np.float32, copy=False)


def convert(args):
    timestamps = pd.date_range(pd.to_datetime(args.start), pd.to_datetime(args.end), freq=args.freq)
    if len(timestamps) == 0:
        raise ValueError("No timestamps were selected")

    era5_dir = Path(args.era5_dir)
    pressure_root = Path(args.pressure_dir) if args.pressure_dir else era5_dir
    single_root = Path(args.single_dir) if args.single_dir else era5_dir
    output_dir = Path(args.output_dir)
    expected_shape = tuple(args.expected_shape)

    cache = {}
    try:
        for timestamp in timestamps:
            output_path = output_dir / f"{timestamp.year:04d}" / f"{timestamp.strftime('%Y%m%d_%H')}.npy"
            if output_path.exists() and not args.overwrite:
                print(f"Skip existing: {output_path}")
                continue

            pressure_path = resolve_era5_file(pressure_root, "pressure", timestamp)
            single_path = resolve_era5_file(single_root, "single", timestamp)
            pressure_ds = open_dataset(pressure_path, cache)
            single_ds = open_dataset(single_path, cache)

            array = build_array(pressure_ds, single_ds, timestamp, expected_shape)
            if len(MODEL_CHANNELS) != array.shape[0]:
                raise ValueError(f"Expected {len(MODEL_CHANNELS)} channels, got {array.shape[0]}")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, array)
            print(f"Wrote {output_path} {array.shape} float32")
    finally:
        for ds in cache.values():
            ds.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ERA5 NetCDF files to NJU-Earth RH-model npy inputs.")
    parser.add_argument("--era5-dir", default="era5_raw", help="Root directory created by download_era5.py.")
    parser.add_argument("--pressure-dir", default=None, help="Optional pressure-level ERA5 root override.")
    parser.add_argument("--single-dir", default=None, help="Optional single-level ERA5 root override.")
    parser.add_argument("--output-dir", default="input", help="Output npy directory.")
    parser.add_argument("--start", required=True, help="First timestamp, e.g. 2024-01-01 00:00:00.")
    parser.add_argument("--end", required=True, help="Last timestamp, e.g. 2024-01-02 00:00:00.")
    parser.add_argument("--freq", default="6h", help="Output timestamp frequency.")
    parser.add_argument("--expected-shape", nargs=2, type=int, default=[721, 1440], metavar=("H", "W"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing npy files.")
    return parser.parse_args()


def main():
    convert(parse_args())


if __name__ == "__main__":
    main()
