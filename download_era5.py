import argparse
from datetime import datetime
from pathlib import Path


PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
PRESSURE_VARIABLES = [
    "geopotential",
    "relative_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]
SINGLE_LEVEL_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
]


def parse_date(value):
    return datetime.fromisoformat(value).date()


def iter_timestamps(start_date, end_date, hours):
    import pandas as pd

    for day in pd.date_range(start_date, end_date, freq="1D"):
        for hour in hours:
            yield day.replace(hour=int(hour.split(":", 1)[0]))


def normalize_hours(hours):
    normalized = []
    for item in hours:
        value = str(item)
        if ":" in value:
            hour = int(value.split(":", 1)[0])
        else:
            hour = int(value)
        if not 0 <= hour <= 23:
            raise ValueError(f"Invalid hour: {item}")
        normalized.append(f"{hour:02d}:00")
    return normalized


def build_request(kind, timestamp, data_format, download_format, grid, area):
    request = {
        "product_type": ["reanalysis"],
        "year": [timestamp.strftime("%Y")],
        "month": [timestamp.strftime("%m")],
        "day": [timestamp.strftime("%d")],
        "time": [timestamp.strftime("%H:00")],
        "data_format": data_format,
        "download_format": download_format,
    }

    if kind == "pressure":
        request["variable"] = PRESSURE_VARIABLES
        request["pressure_level"] = [str(level) for level in PRESSURE_LEVELS]
    elif kind == "single":
        request["variable"] = SINGLE_LEVEL_VARIABLES
    else:
        raise ValueError(f"Unsupported ERA5 kind: {kind}")

    if grid is not None:
        request["grid"] = [float(grid[0]), float(grid[1])]
    if area is not None:
        request["area"] = [float(value) for value in area]

    return request


def retrieve_timestamp(client, dataset, request, target, overwrite):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        print(f"Skip existing: {target}")
        return

    print(f"Downloading {dataset} -> {target}")
    client.retrieve(dataset, request, str(target))


def download_era5(args):
    try:
        import cdsapi
    except ModuleNotFoundError as exc:
        raise RuntimeError("cdsapi is required. Install it before downloading ERA5.") from exc

    start_date = parse_date(args.start)
    end_date = parse_date(args.end)
    if end_date < start_date:
        raise ValueError("--end must be on or after --start")

    hours = normalize_hours(args.hours)
    client = cdsapi.Client()
    output_dir = Path(args.output_dir)
    extension = args.extension or ("nc" if args.data_format == "netcdf" else "grib")

    kinds = []
    if not args.single_only:
        kinds.append("pressure")
    if not args.pressure_only:
        kinds.append("single")

    for timestamp in iter_timestamps(start_date, end_date, hours):
        for kind in kinds:
            dataset = "reanalysis-era5-pressure-levels" if kind == "pressure" else "reanalysis-era5-single-levels"
            request = build_request(
                kind=kind,
                timestamp=timestamp,
                data_format=args.data_format,
                download_format=args.download_format,
                grid=None if args.no_grid else args.grid,
                area=args.area,
            )
            target = output_dir / kind / timestamp.strftime("%Y") / f"era5_{kind}_{timestamp.strftime('%Y%m%d_%H')}.{extension}"
            retrieve_timestamp(client, dataset, request, target, overwrite=args.overwrite)


def parse_args():
    parser = argparse.ArgumentParser(description="Download ERA5 fields needed by the NJU-Earth RH model.")
    parser.add_argument("--start", required=True, help="Start date, e.g. 2024-01-01.")
    parser.add_argument("--end", required=True, help="End date, e.g. 2024-01-31.")
    parser.add_argument("--output-dir", default="era5_raw", help="Directory for downloaded per-timestamp files.")
    parser.add_argument("--hours", nargs="+", default=["00", "06", "12", "18"], help="UTC hours to download.")
    parser.add_argument("--data-format", default="netcdf", choices=["netcdf", "grib"], help="CDS data format.")
    parser.add_argument("--download-format", default="unarchived", help="CDS download format.")
    parser.add_argument("--extension", default=None, help="Downloaded file extension. Defaults to nc or grib.")
    parser.add_argument("--grid", nargs=2, type=float, default=[0.25, 0.25], metavar=("DLAT", "DLON"))
    parser.add_argument("--no-grid", action="store_true", help="Do not include a grid keyword in CDS requests.")
    parser.add_argument("--area", nargs=4, type=float, metavar=("N", "W", "S", "E"), help="Optional area subset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--pressure-only", action="store_true", help="Download only pressure-level fields.")
    parser.add_argument("--single-only", action="store_true", help="Download only single-level fields.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.pressure_only and args.single_only:
        raise ValueError("--pressure-only and --single-only cannot both be set")
    download_era5(args)


if __name__ == "__main__":
    main()
