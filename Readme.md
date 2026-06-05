# NJU-Earth V1.0 推理说明

NJU-Earth 是南京大学大气科学学院联合阿里巴巴达摩院在 Baguan 基础上研发的 0.25° 全球气象模型。该模型采用 Transformer 主干网络，通过多尺度天气特征预编码技术将多时刻气象场编码为高维特征，并通过交叉注意力特征融合技术引入时间信息和静态地理常数作为条件，从而实现对物理背景信息的有效注入。

当前仓库提供 NJU-Earth 的 ONNX 推理流程和 ERA5 数据预处理工具：

- `download_era5.py`：下载推理所需 ERA5 pressure-level 和 single-level 数据。
- `era5_to_npy.py`：把 ERA5 NetCDF 转成模型输入 `.npy`。
- `infer_onnx.py`：使用 `NJU-Earth.onnx` 做 ONNX 自回归推理。

## 依赖

Linux 推荐使用 Python 3.10+。安装基础依赖：

```bash
python -m pip install cdsapi xarray netcdf4 tqdm onnxruntime-gpu
```

如果机器没有系统级 CUDA/cuDNN，或 `onnxruntime-gpu` 找不到 CUDA/cuDNN，可以安装 pip 版 CUDA 12/cuDNN 9 运行时：

```bash
python -m pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12
```

如果使用系统安装的 CUDA/cuDNN，请确保相关库路径已经在启动 Python 前加入 `LD_LIBRARY_PATH`。例如：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

使用 `download_era5.py` 前，需要提前配置 CDS API 账号和 key。

## 模型权重

NJU-Earth V1.0 ONNX 权重下载地址：[[weights](https://box.nju.edu.cn/d/6aad54574243408485ed/)]


下载后请将模型文件放在仓库根目录，并保持以下文件名：

```text
NJU-Earth.onnx
NJU-Earth_weights.data
```

## 文件结构

推荐目录结构如下：

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
│   ├── pressure/
│   │   └── YYYY/
│   │       └── era5_pressure_YYYYMMDD_HH.nc
│   └── single/
│       └── YYYY/
│           └── era5_single_YYYYMMDD_HH.nc
├── input/
│   └── YYYY/
│       └── YYYYMMDD_HH.npy
└── output/
    └── YYYYMMDD_HH/
        ├── 6.npy
        ├── 12.npy
        ├── ...
        └── 72.npy
```

ERA5 原始数据按“一个时刻一个文件”保存。每个时刻会下载两个文件：一个 pressure-level 文件，一个 single-level 文件。

## 变量顺序

模型使用 ERA5 相对湿度 `rh`，不是比湿。通道顺序固定如下：

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

## 单个样本：指定起报时刻推理

示例：对 `2024-01-01 06:00:00` 起报，做未来 72 小时预报。

`infer_onnx.py` 会读取两个输入时刻：

```text
start_time - time_step = 2024-01-01 00:00:00
start_time             = 2024-01-01 06:00:00
```

所以需要先准备这两个时刻的 ERA5 文件和 `.npy` 输入。

### 1. 下载 ERA5

每条命令只下载一个时刻：

```bash
python download_era5.py --start 2024-01-01 --end 2024-01-01 --output-dir era5_raw --hours 00
python download_era5.py --start 2024-01-01 --end 2024-01-01 --output-dir era5_raw --hours 06
```

下载后文件结构应为：

```text
era5_raw/
  pressure/2024/era5_pressure_20240101_00.nc
  pressure/2024/era5_pressure_20240101_06.nc
  single/2024/era5_single_20240101_00.nc
  single/2024/era5_single_20240101_06.nc
```

如果起报时刻是 `00:00:00`，前一帧通常在前一天 `18:00:00`，下载日期需要覆盖前一天。例如起报 `2024-01-01 00:00:00` 时，应下载 `2023-12-31` 和 `2024-01-01`。

### 2. ERA5 转 npy

```bash
python era5_to_npy.py --era5-dir era5_raw --output-dir input --start "2024-01-01 00:00:00" --end "2024-01-01 06:00:00" --freq 6h
```

转换后应得到：

```text
input/2024/20240101_00.npy
input/2024/20240101_06.npy
```

### 3. 运行单个起报时刻推理

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

## 批量推理

示例：从 `2024-01-01 00:00:00` 到 `2024-01-02 12:00:00`，每 12 小时起报一次，每次预报 72 小时。

起报时刻为：

```text
2024-01-01 00:00:00
2024-01-01 12:00:00
2024-01-02 00:00:00
2024-01-02 12:00:00
```

因为第一个起报时刻 `2024-01-01 00:00:00` 需要前一帧 `2023-12-31 18:00:00`，所以 ERA5 和 npy 输入要从 `2023-12-31 18:00:00` 覆盖到 `2024-01-02 12:00:00`。

### 1. 下载 ERA5

Linux/bash 示例：

```bash
times=(
  "2023-12-31 18"
  "2024-01-01 00"
  "2024-01-01 06"
  "2024-01-01 12"
  "2024-01-01 18"
  "2024-01-02 00"
  "2024-01-02 06"
  "2024-01-02 12"
)

for t in "${times[@]}"; do
  day="${t% *}"
  hour="${t#* }"
  python download_era5.py --start "$day" --end "$day" --output-dir era5_raw --hours "$hour"
done
```

该循环会为每个时刻分别下载 pressure 和 single 文件，例如：

```text
era5_raw/pressure/2023/era5_pressure_20231231_18.nc
era5_raw/single/2023/era5_single_20231231_18.nc
era5_raw/pressure/2024/era5_pressure_20240101_00.nc
era5_raw/single/2024/era5_single_20240101_00.nc
...
```

### 2. ERA5 转 npy

```bash
python era5_to_npy.py --era5-dir era5_raw --output-dir input --start "2023-12-31 18:00:00" --end "2024-01-02 12:00:00" --freq 6h
```

### 3. 运行批量推理

```bash
python infer_onnx.py --work-path . --start-time "2024-01-01 00:00:00" --end-time "2024-01-02 12:00:00" --case-freq 12h --max-time 72 --time-step 6 --gpu-id 0
```

结果会按起报时刻分别保存：

```text
output/20240101_00/
output/20240101_12/
output/20240102_00/
output/20240102_12/
```

每个目录下都有 `6.npy` 到 `72.npy` 的逐时效结果。

## GPU 检查

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

如果只看到 `CPUExecutionProvider`，说明 `onnxruntime-gpu` 没有找到 CUDA/cuDNN。优先检查 `nvidia-smi`、`LD_LIBRARY_PATH`、以及是否安装了上面的 CUDA/cuDNN pip 包。

## 常用参数

- `--max-time`：最大预报时效，默认 `72` 小时。
- `--time-step`：模型步长，默认 `6` 小时。
- `--case-freq`：批量起报间隔，例如 `12h` 表示每 12 小时起报一次。
- `--gpu-id`：ONNX Runtime 使用的 GPU 编号。
- `--cpu`：强制使用 CPU 推理。
- `--overwrite`：下载或转换时覆盖已有文件。

## 注意事项

- ERA5 原始 nc 文件按单个时刻拆分保存，不把一天或一个月合并到同一个 nc 文件。
- 输入 `.npy` 必须同时包含 `start_time - time_step` 和 `start_time` 两个时刻。
- ERA5 下载时间是 UTC 小时。
- 输出 `.npy` 已经做了反标准化，形状为 `(69, 721, 1440)`。
- 如果只想用已有 `.npy` 输入，可以跳过下载和转换步骤，直接运行 `infer_onnx.py`。
