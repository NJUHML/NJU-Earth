# NJU-Earth V1.0 Inference Guide

[English](#english) | [中文](#中文)

## English

NJU-Earth is a 0.25-degree global weather forecasting model developed by the School of Atmospheric Sciences, Nanjing University, together with Alibaba DAMO Academy, based on Baguan. The model uses a Transformer backbone, encodes multi-time weather fields with multi-scale weather feature pre-encoding, and injects time information plus static geographic constants through cross-attention feature fusion.

This directory provides the ONNX inference workflow and ERA5 preprocessing tools:

- `download_era5.py`: download ERA5 pressure-level and single-level fields.
- `era5_to_npy.py`: convert ERA5 NetCDF files to model-ready `.npy` inputs.
- `infer_onnx.py`: run autoregressive ONNX inference with `NJU-Earth.onnx`.
- `make_onnx.py`: export the PyTorch checkpoint to ONNX.

### Requirements

Linux with Python 3.10+ is recommended.

```bash
python -m pip install cdsapi xarray netcdf4 tqdm onnxruntime-gpu
```

If `onnxruntime-gpu` cannot find CUDA/cuDNN, install the pip CUDA 12/cuDNN 9 runtime packages:

```bash
python -m pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12
```

If you use system CUDA/cuDNN, make sure the library paths are available before starting Python:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Configure your CDS API account and key before running `download_era5.py`.

### Model Files

Download NJU-Earth V1.0 ONNX weights from: [weights](https://box.nju.edu.cn/d/6aad54574243408485ed/)

Place the model files in this directory with these exact names:

```text
NJU-Earth.onnx
NJU-Earth_weights.data
```

### Directory Layout

Recommended layout:

```text
NJU-Earth_V1.0/
├── NJU-Earth.onnx
├── NJU-Earth_weights.data
├── auxiliary/
│   ├── normalize.npz
│   └── constants.npy
├── download_era5.py
├── era5_to_npy.py
├── infer_onnx.py
├── era5_raw/
│   ├── pressure/YYYY/era5_pressure_YYYYMMDD_HH.nc
│   └── single/YYYY/era5_single_YYYYMMDD_HH.nc
├── input/YYYY/YYYYMMDD_HH.npy
└── output/YYYYMMDD_HH/
    ├── 6.npy
    ├── 12.npy
    ├── ...
    └── 72.npy
```

ERA5 raw files are stored one timestamp per file. Each timestamp requires one pressure-level file and one single-level file.

### Variable Order

The model uses ERA5 relative humidity `rh`, not specific humidity. The channel order is fixed:

```python
DEFAULT_VARS = [
    'z_50', 'rh_50', 't_50', 'u_50', 'v_50', 'z_100', 'rh_100', 't_100', 'u_100', 'v_100',
    'z_150', 'rh_150', 't_150', 'u_150', 'v_150', 'z_200', 'rh_200', 't_200', 'u_200', 'v_200',
    'z_250', 'rh_250', 't_250', 'u_250', 'v_250', 'z_300', 'rh_300', 't_300', 'u_300', 'v_300',
    'z_400', 'rh_400', 't_400', 'u_400', 'v_400', 'z_500', 'rh_500', 't_500', 'u_500', 'v_500',
    'z_600', 'rh_600', 't_600', 'u_600', 'v_600', 'z_700', 'rh_700', 't_700', 'u_700', 'v_700',
    'z_850', 'rh_850', 't_850', 'u_850', 'v_850', 'z_925', 'rh_925', 't_925', 'u_925', 'v_925',
    'z_1000', 'rh_1000', 't_1000', 'u_1000', 'v_1000',
    'u10', 'v10', 't2m', 'msl',
]
```

### Single-Case Inference

Example: initialize at `2024-01-01 06:00:00` and forecast 72 hours.

`infer_onnx.py` reads two input frames:

```text
start_time - time_step = 2024-01-01 00:00:00
start_time             = 2024-01-01 06:00:00
```

Prepare ERA5 and `.npy` inputs for both timestamps.

#### 1. Download ERA5

Each command downloads one timestamp:

```bash
python download_era5.py --start 2024-01-01 --end 2024-01-01 --output-dir era5_raw --hours 00
python download_era5.py --start 2024-01-01 --end 2024-01-01 --output-dir era5_raw --hours 06
```

Expected files:

```text
era5_raw/pressure/2024/era5_pressure_20240101_00.nc
era5_raw/pressure/2024/era5_pressure_20240101_06.nc
era5_raw/single/2024/era5_single_20240101_00.nc
era5_raw/single/2024/era5_single_20240101_06.nc
```

If `start_time` is `00:00:00`, the previous frame is usually `18:00:00` on the previous day.

#### 2. Convert ERA5 to NPY

```bash
python era5_to_npy.py --era5-dir era5_raw --output-dir input --start "2024-01-01 00:00:00" --end "2024-01-01 06:00:00" --freq 6h
```

Expected outputs:

```text
input/2024/20240101_00.npy
input/2024/20240101_06.npy
```

#### 3. Run Inference

```bash
python infer_onnx.py --work-path . --start-time "2024-01-01 06:00:00" --max-time 72 --time-step 6 --gpu-id 0
```

Forecasts are saved to:

```text
output/20240101_06/6.npy
output/20240101_06/12.npy
...
output/20240101_06/72.npy
```

### Batch Inference

Example: initialize every 12 hours from `2024-01-01 00:00:00` to `2024-01-02 12:00:00`, forecasting 72 hours each time.

Initialization times:

```text
2024-01-01 00:00:00
2024-01-01 12:00:00
2024-01-02 00:00:00
2024-01-02 12:00:00
```

Because the first case needs `2023-12-31 18:00:00`, prepare ERA5 and `.npy` inputs from `2023-12-31 18:00:00` through `2024-01-02 12:00:00`.

```bash
python era5_to_npy.py --era5-dir era5_raw --output-dir input --start "2023-12-31 18:00:00" --end "2024-01-02 12:00:00" --freq 6h

python infer_onnx.py --work-path . --start-time "2024-01-01 00:00:00" --end-time "2024-01-02 12:00:00" --case-freq 12h --max-time 72 --time-step 6 --gpu-id 0
```

Outputs are grouped by initialization time:

```text
output/20240101_00/
output/20240101_12/
output/20240102_00/
output/20240102_12/
```

### Export ONNX from Checkpoint

`make_onnx.py` exports the PyTorch checkpoint to `NJU-Earth.onnx` plus the external data file `NJU-Earth_weights.data`.

CPU export is recommended for this large model:

```bash
python make_onnx.py
```

The script uses the legacy ONNX exporter (`dynamo=False`) and disables checkpointing during export to avoid non-equivalent patch-grid artifacts. It also removes stale external data files before writing new ones.

### GPU Check

Verify that ONNX Runtime can use CUDA:

```bash
python - <<'PY'
from infer_onnx import create_session
s = create_session("NJU-Earth.onnx", gpu_id=0, use_cuda=True)
print(s.get_providers())
PY
```

Expected providers include:

```text
CUDAExecutionProvider
CPUExecutionProvider
```

If only `CPUExecutionProvider` appears, check `nvidia-smi`, `LD_LIBRARY_PATH`, and CUDA/cuDNN installation.

### Common Arguments

- `--max-time`: forecast length in hours, default `72`.
- `--time-step`: model step in hours, default `6`.
- `--case-freq`: initialization interval for batch inference, for example `12h`.
- `--gpu-id`: CUDA device id used by ONNX Runtime.
- `--cpu`: force CPU inference.
- `--overwrite`: overwrite existing files during download or conversion.

### Notes

- ERA5 raw NetCDF files should be split by timestamp; do not merge a full day or month into one file.
- Input `.npy` files must include both `start_time - time_step` and `start_time`.
- ERA5 download hours are UTC hours.
- Output `.npy` files are denormalized and have shape `(69, 721, 1440)`.
- If `.npy` inputs already exist, skip download/conversion and run `infer_onnx.py` directly.
- Training range: `[20000102, 20191231]`.

---

## 中文

NJU-Earth 是南京大学大气科学学院联合阿里巴巴达摩院在 Baguan 基础上研发的 0.25° 全球气象模型。该模型采用 Transformer 主干网络，通过多尺度天气特征预编码技术将多时刻气象场编码为高维特征，并通过交叉注意力特征融合技术引入时间信息和静态地理常数。

当前目录提供 ONNX 推理流程和 ERA5 数据预处理工具：

- `download_era5.py`：下载 ERA5 pressure-level 和 single-level 数据。
- `era5_to_npy.py`：把 ERA5 NetCDF 转成模型输入 `.npy`。
- `infer_onnx.py`：使用 `NJU-Earth.onnx` 做 ONNX 自回归推理。
- `make_onnx.py`：从 PyTorch checkpoint 导出 ONNX。

### 依赖

推荐 Linux + Python 3.10+：

```bash
python -m pip install cdsapi xarray netcdf4 tqdm onnxruntime-gpu
```

如果 `onnxruntime-gpu` 找不到 CUDA/cuDNN，可以安装 pip 版 CUDA 12/cuDNN 9 运行时：

```bash
python -m pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12
```

如果使用系统 CUDA/cuDNN，请在启动 Python 前设置库路径：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

使用 `download_era5.py` 前，需要提前配置 CDS API 账号和 key。

### 模型文件

NJU-Earth V1.0 ONNX 权重下载地址：[weights](https://box.nju.edu.cn/d/6aad54574243408485ed/)

下载后请放在当前目录，并保持以下文件名：

```text
NJU-Earth.onnx
NJU-Earth_weights.data
```

### 文件结构

推荐目录结构：

```text
NJU-Earth_V1.0/
├── NJU-Earth.onnx
├── NJU-Earth_weights.data
├── auxiliary/
│   ├── normalize.npz
│   └── constants.npy
├── download_era5.py
├── era5_to_npy.py
├── infer_onnx.py
├── era5_raw/
│   ├── pressure/YYYY/era5_pressure_YYYYMMDD_HH.nc
│   └── single/YYYY/era5_single_YYYYMMDD_HH.nc
├── input/YYYY/YYYYMMDD_HH.npy
└── output/YYYYMMDD_HH/
    ├── 6.npy
    ├── 12.npy
    ├── ...
    └── 72.npy
```

ERA5 原始数据按“一个时刻一个文件”保存。每个时刻需要一个 pressure-level 文件和一个 single-level 文件。

### 变量顺序

模型使用 ERA5 相对湿度 `rh`，不是比湿。通道顺序固定：

```python
DEFAULT_VARS = [
    'z_50', 'rh_50', 't_50', 'u_50', 'v_50', 'z_100', 'rh_100', 't_100', 'u_100', 'v_100',
    'z_150', 'rh_150', 't_150', 'u_150', 'v_150', 'z_200', 'rh_200', 't_200', 'u_200', 'v_200',
    'z_250', 'rh_250', 't_250', 'u_250', 'v_250', 'z_300', 'rh_300', 't_300', 'u_300', 'v_300',
    'z_400', 'rh_400', 't_400', 'u_400', 'v_400', 'z_500', 'rh_500', 't_500', 'u_500', 'v_500',
    'z_600', 'rh_600', 't_600', 'u_600', 'v_600', 'z_700', 'rh_700', 't_700', 'u_700', 'v_700',
    'z_850', 'rh_850', 't_850', 'u_850', 'v_850', 'z_925', 'rh_925', 't_925', 'u_925', 'v_925',
    'z_1000', 'rh_1000', 't_1000', 'u_1000', 'v_1000',
    'u10', 'v10', 't2m', 'msl',
]
```

### 单个样本推理

示例：对 `2024-01-01 06:00:00` 起报，做未来 72 小时预报。

`infer_onnx.py` 会读取两个输入时刻：

```text
start_time - time_step = 2024-01-01 00:00:00
start_time             = 2024-01-01 06:00:00
```

所以需要先准备这两个时刻的 ERA5 文件和 `.npy` 输入。

#### 1. 下载 ERA5

每条命令下载一个时刻：

```bash
python download_era5.py --start 2024-01-01 --end 2024-01-01 --output-dir era5_raw --hours 00
python download_era5.py --start 2024-01-01 --end 2024-01-01 --output-dir era5_raw --hours 06
```

如果起报时刻是 `00:00:00`，前一帧通常在前一天 `18:00:00`，下载日期需要覆盖前一天。

#### 2. ERA5 转 NPY

```bash
python era5_to_npy.py --era5-dir era5_raw --output-dir input --start "2024-01-01 00:00:00" --end "2024-01-01 06:00:00" --freq 6h
```

转换后应得到：

```text
input/2024/20240101_00.npy
input/2024/20240101_06.npy
```

#### 3. 运行推理

```bash
python infer_onnx.py --work-path . --start-time "2024-01-01 06:00:00" --max-time 72 --time-step 6 --gpu-id 0
```

结果保存到：

```text
output/20240101_06/6.npy
output/20240101_06/12.npy
...
output/20240101_06/72.npy
```

### 批量推理

示例：从 `2024-01-01 00:00:00` 到 `2024-01-02 12:00:00`，每 12 小时起报一次，每次预报 72 小时。

因为第一个起报时刻需要 `2023-12-31 18:00:00`，所以 ERA5 和 `.npy` 输入要覆盖 `2023-12-31 18:00:00` 到 `2024-01-02 12:00:00`。

```bash
python era5_to_npy.py --era5-dir era5_raw --output-dir input --start "2023-12-31 18:00:00" --end "2024-01-02 12:00:00" --freq 6h

python infer_onnx.py --work-path . --start-time "2024-01-01 00:00:00" --end-time "2024-01-02 12:00:00" --case-freq 12h --max-time 72 --time-step 6 --gpu-id 0
```

结果会按起报时刻分别保存到：

```text
output/20240101_00/
output/20240101_12/
output/20240102_00/
output/20240102_12/
```

### 从 Checkpoint 导出 ONNX

`make_onnx.py` 会导出 `NJU-Earth.onnx` 和外部权重文件 `NJU-Earth_weights.data`。

推荐使用 CPU 导出：

```bash
python make_onnx.py
```

脚本会使用 legacy ONNX exporter（`dynamo=False`），并在导出时关闭 checkpoint，以避免 patch 网格伪影。脚本也会在写入前清理旧 external data 文件。

### GPU 检查

确认 ONNX Runtime 能启用 CUDA：

```bash
python - <<'PY'
from infer_onnx import create_session
s = create_session("NJU-Earth.onnx", gpu_id=0, use_cuda=True)
print(s.get_providers())
PY
```

正常应包含：

```text
CUDAExecutionProvider
CPUExecutionProvider
```

如果只有 `CPUExecutionProvider`，请检查 `nvidia-smi`、`LD_LIBRARY_PATH` 和 CUDA/cuDNN 安装。

### 常用参数

- `--max-time`：最大预报时效，默认 `72` 小时。
- `--time-step`：模型步长，默认 `6` 小时。
- `--case-freq`：批量起报间隔，例如 `12h`。
- `--gpu-id`：ONNX Runtime 使用的 GPU 编号。
- `--cpu`：强制使用 CPU 推理。
- `--overwrite`：下载或转换时覆盖已有文件。

### 注意事项

- ERA5 原始 NetCDF 文件应按单个时刻拆分保存，不要把一天或一个月合并到同一个文件。
- 输入 `.npy` 必须同时包含 `start_time - time_step` 和 `start_time` 两个时刻。
- ERA5 下载时间是 UTC 小时。
- 输出 `.npy` 已反标准化，形状为 `(69, 721, 1440)`。
- 如果已有 `.npy` 输入，可以跳过下载和转换，直接运行 `infer_onnx.py`。
- 训练集范围：`[20000102, 20191231]`。
