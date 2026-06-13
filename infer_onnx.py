import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_kwargs):
        return iterable


_DLL_DIRECTORY_HANDLES = []


def add_nvidia_dll_directories():
    """Make pip-installed NVIDIA runtime libraries visible to ONNX Runtime.

    On Windows this extends the DLL search path for cuDNN sub-libraries. On
    Linux, onnxruntime.preload_dlls(directory="") handles pip NVIDIA packages;
    system CUDA/cuDNN should be exposed through LD_LIBRARY_PATH before Python
    starts.
    """
    if not hasattr(os, "add_dll_directory"):
        return

    try:
        import nvidia
    except ModuleNotFoundError:
        return

    for package_root in nvidia.__path__:
        for bin_dir in Path(package_root).glob("*/bin"):
            if bin_dir.is_dir():
                bin_dir_str = str(bin_dir)
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(bin_dir_str))
                path_parts = os.environ.get("PATH", "").split(os.pathsep)
                if bin_dir_str not in path_parts:
                    os.environ["PATH"] = bin_dir_str + os.pathsep + os.environ.get("PATH", "")


class Preprocessing:
    def __init__(self, max_time=72, time_step=6, auxiliary_path="auxiliary"):
        auxiliary_path = Path(auxiliary_path)
        norm_file = np.load(auxiliary_path / "normalize.npz")
        self.mean = norm_file["mean"].astype(np.float32)
        self.std = norm_file["std"].astype(np.float32)
        self.constants = np.load(auxiliary_path / "constants.npy").astype(np.float32)
        self.max_time = max_time
        self.time_step = time_step

    def process(self, data1, data2, start_date):
        data1 = data1.astype(np.float32)
        data2 = data2.astype(np.float32)

        data1 = (data1 - self.mean[:, None, None]) / self.std[:, None, None]
        data2 = (data2 - self.mean[:, None, None]) / self.std[:, None, None]

        data1 = np.expand_dims(data1, axis=0).astype(np.float32)  # (1, C, H, W)
        data2 = np.expand_dims(data2, axis=0).astype(np.float32)  # (1, C, H, W)
        constants = np.expand_dims(self.constants, axis=0).astype(np.float32)

        hour_of_day_list = []
        day_of_year_list = []
        start_date = pd.to_datetime(start_date)

        for i in range(0, self.max_time, self.time_step):
            forecast_date = start_date + pd.Timedelta(hours=i)
            hour_of_day_list.append(np.array([forecast_date.hour], dtype=np.int64))
            day_of_year_list.append(np.array([forecast_date.dayofyear], dtype=np.int64))

        return data1, data2, constants, hour_of_day_list, day_of_year_list


class Postprocessing:
    def __init__(self, auxiliary_path="auxiliary"):
        auxiliary_path = Path(auxiliary_path)
        norm_file = np.load(auxiliary_path / "normalize.npz")
        self.mean = norm_file["mean"].astype(np.float32)
        self.std = norm_file["std"].astype(np.float32)

    def process(self, output):
        output = np.asarray(output, dtype=np.float32)
        if output.ndim == 4:
            output = output[0]
        if output.ndim != 3:
            raise ValueError(f"Expected ONNX output with shape (1, C, H, W) or (C, H, W), got {output.shape}")

        output = output * self.std[:, None, None] + self.mean[:, None, None]
        return output.astype(np.float32, copy=False)


def _resolve_case_input(input_path, timestamp):
    input_path = Path(input_path)
    timestamp = pd.to_datetime(timestamp)
    name = f"{timestamp.strftime('%Y%m%d_%H')}.npy"
    candidates = [
        input_path / f"{timestamp.year}" / name,
        input_path / name,
    ]

    for path in candidates:
        if path.exists():
            return path

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Input file not found for {timestamp}: tried {tried}")


def create_session(onnx_model_path, gpu_id=0, use_cuda=True):
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "onnxruntime is required to run infer_onnx.py. Install onnxruntime-gpu for CUDA "
            "or onnxruntime for CPU."
        ) from exc

    providers = []
    available = ort.get_available_providers()
    if use_cuda and "CUDAExecutionProvider" in available:
        add_nvidia_dll_directories()
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory="")
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": int(gpu_id),
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "cudnn_conv_use_max_workspace": "1",
                    "arena_extend_strategy": "kNextPowerOfTwo",
                },
            )
        )
    elif use_cuda:
        print("CUDAExecutionProvider is not available; falling back to CPUExecutionProvider.")

    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(str(onnx_model_path), providers=providers)
    active_providers = session.get_providers()
    if use_cuda and "CUDAExecutionProvider" in available and "CUDAExecutionProvider" not in active_providers:
        raise RuntimeError(
            "CUDAExecutionProvider is available but was not activated. Check CUDA/cuDNN DLL installation "
            "or run with --cpu to force CPU inference."
        )
    if use_cuda and "CUDAExecutionProvider" in active_providers and hasattr(session, "disable_fallback"):
        session.disable_fallback()

    input_names = {item.name for item in session.get_inputs()}
    expected = {"data", "constants", "hour_of_day", "day_of_year"}
    missing = expected - input_names
    if missing:
        raise ValueError(f"ONNX model inputs are missing {sorted(missing)}; actual inputs are {sorted(input_names)}")

    return session


def infer_one_case(
    session,
    preprocessing,
    postprocessing,
    input_path,
    output_path,
    start_time,
    max_time,
    time_step,
    show_inner_pbar=True,
):
    start_time = pd.to_datetime(start_time)

    save_path = Path(output_path) / start_time.strftime("%Y%m%d_%H")
    save_path.mkdir(parents=True, exist_ok=True)

    input_t1 = start_time - pd.Timedelta(hours=time_step)
    input_t2 = start_time

    data1 = np.load(_resolve_case_input(input_path, input_t1))
    data2 = np.load(_resolve_case_input(input_path, input_t2))

    data1, data2, constants, hour_of_day_list, day_of_year_list = preprocessing.process(data1, data2, start_time)
    prev1, prev2 = data1, data2

    output_name = session.get_outputs()[0].name
    step_iter = range(max_time // time_step)
    if show_inner_pbar:
        step_iter = tqdm(step_iter, desc=start_time.strftime("%Y%m%d_%H"), leave=False)

    for i in step_iter:
        data = np.stack((prev1, prev2), axis=2).astype(np.float32, copy=False)  # (1, C, 2, H, W)
        ort_inputs = {
            "data": data,
            "constants": constants,
            "hour_of_day": hour_of_day_list[i],
            "day_of_year": day_of_year_list[i],
        }

        out = session.run([output_name], ort_inputs)[0].astype(np.float32, copy=False)
        prev1, prev2 = prev2, out

        forecast = postprocessing.process(out)
        lead_time = (i + 1) * time_step
        save_file = save_path / f"{lead_time}.npy"
        np.save(save_file, forecast)


def infer(work_path, start_time, max_time=72, time_step=6, gpu_id=0):
    work_path = Path(work_path)
    session = create_session(work_path / "NJU-Earth.onnx", gpu_id=gpu_id, use_cuda=True)
    auxiliary_path = work_path / "auxiliary"
    preprocessing = Preprocessing(max_time=max_time, time_step=time_step, auxiliary_path=auxiliary_path)
    postprocessing = Postprocessing(auxiliary_path=auxiliary_path)

    infer_one_case(
        session=session,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        input_path=work_path / "input",
        output_path=work_path / "output",
        start_time=start_time,
        max_time=max_time,
        time_step=time_step,
        show_inner_pbar=True,
    )


def _path_from_workdir(work_path, maybe_path):
    path = Path(maybe_path)
    if path.is_absolute():
        return path
    return Path(work_path) / path


def parse_args():
    parser = argparse.ArgumentParser(description="Run autoregressive NJU-Earth ONNX inference.")
    parser.add_argument("--work-path", default=".", help="Project or inference work directory.")
    parser.add_argument("--onnx-model", default="NJU-Earth.onnx", help="ONNX model path, relative to work-path if not absolute.")
    parser.add_argument("--input-path", default="input", help="Input npy directory, relative to work-path if not absolute.")
    parser.add_argument("--output-path", default="output", help="Output directory, relative to work-path if not absolute.")
    parser.add_argument("--start-time", required=True, help="Forecast initialization time, e.g. 2024-01-01 00:00:00.")
    parser.add_argument("--end-time", default=None, help="Optional final initialization time for batch inference.")
    parser.add_argument("--case-freq", default="12h", help="Batch initialization frequency when end-time is set.")
    parser.add_argument("--max-time", type=int, default=72, help="Forecast length in hours.")
    parser.add_argument("--time-step", type=int, default=6, help="Forecast step in hours.")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device id for ONNX Runtime.")
    parser.add_argument("--cpu", action="store_true", help="Force CPUExecutionProvider.")
    return parser.parse_args()


def main():
    args = parse_args()
    work_path = Path(args.work_path)
    onnx_model = _path_from_workdir(work_path, args.onnx_model)
    input_path = _path_from_workdir(work_path, args.input_path)
    output_path = _path_from_workdir(work_path, args.output_path)

    session = create_session(onnx_model, gpu_id=args.gpu_id, use_cuda=not args.cpu)
    auxiliary_path = work_path / "auxiliary"
    preprocessing = Preprocessing(max_time=args.max_time, time_step=args.time_step, auxiliary_path=auxiliary_path)
    postprocessing = Postprocessing(auxiliary_path=auxiliary_path)

    if args.end_time:
        case_dates = pd.date_range(pd.to_datetime(args.start_time), pd.to_datetime(args.end_time), freq=args.case_freq)
    else:
        case_dates = [pd.to_datetime(args.start_time)]

    for start_time in tqdm(case_dates, desc="All cases"):
        infer_one_case(
            session=session,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            input_path=input_path,
            output_path=output_path,
            start_time=start_time,
            max_time=args.max_time,
            time_step=args.time_step,
            show_inner_pbar=True,
        )


if __name__ == "__main__":
    main()
