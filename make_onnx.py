import argparse
import gc
import math
import sys
from pathlib import Path

import torch
import torch.onnx
from omegaconf import OmegaConf


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aiweather.models.AiCast import AiCast


torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def register_export_symbolics(opset):
    def deg2rad_symbolic(g, input_tensor):
        scale = g.op("Constant", value_t=torch.tensor(math.pi / 180.0, dtype=torch.float32))
        return g.op("Mul", input_tensor, scale)

    torch.onnx.register_custom_op_symbolic("aten::deg2rad", deg2rad_symbolic, opset)


def remove_prefix_from_state_dict(state_dict, prefix="net."):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_state_dict[key[len(prefix):]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def remove_deterministic_buffers(state_dict):
    # These buffers are rebuilt from fixed latitude/longitude definitions in Time_Embed.
    # The module buffers are meshgrid views, so copying checkpoint values into them can
    # fail because multiple target elements refer to the same storage location.
    for key in ("time_embed.lat_grid", "time_embed.lon_grid"):
        state_dict.pop(key, None)
    return state_dict


def normalize_external_data_path(onnx_path, external_data_name):
    try:
        import onnx
    except ModuleNotFoundError:
        print("onnx package is not installed; keep the exporter default external data file name.")
        return

    external_data_path = onnx_path.parent / external_data_name
    if external_data_path.exists():
        external_data_path.unlink()

    model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save_model(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_name,
        size_threshold=1024,
        convert_attribute=False,
    )

    default_external_data_path = onnx_path.parent / f"{onnx_path.name}.data"
    if default_external_data_path != external_data_path and default_external_data_path.exists():
        default_external_data_path.unlink()


def remove_previous_export_files(onnx_path, external_data_name):
    candidates = [
        onnx_path,
        onnx_path.parent / f"{onnx_path.name}.data",
        onnx_path.parent / external_data_name,
    ]
    for path in candidates:
        if path.exists():
            path.unlink()


def export_aicast_to_onnx(args):
    register_export_symbolics(args.opset)

    device = torch.device(args.device)
    if device.type == "cuda" and not args.allow_cuda_export:
        print(
            f"检测到 --device {args.device}。为避免 ONNX 导出阶段爆显存，"
            "本脚本默认改用 CPU 导出；如确认要用 GPU，请额外加 --allow-cuda-export。"
        )
        device = torch.device("cpu")

    print("正在加载模型配置...")
    cfg = OmegaConf.load(args.config)
    cfg.Model.use_checkpoint = False
    model = AiCast(**cfg.Model)

    print("正在加载模型权重...")
    weight = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    state_dict = remove_deterministic_buffers(
        remove_prefix_from_state_dict(weight["state_dict"], prefix="net.")
    )
    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
    del weight
    gc.collect()

    model = model.to(device)
    model.requires_grad_(False)
    model.eval()

    print("正在构造虚拟输入张量...")
    dummy_inputs = (
        torch.zeros(1, 69, 2, 721, 1440, dtype=torch.float32, device=device),
        torch.zeros(1, 3, 721, 1440, dtype=torch.float32, device=device),
        torch.tensor([0], dtype=torch.int64, device=device),
        torch.tensor([1], dtype=torch.int64, device=device),
    )

    onnx_path = Path(args.output).resolve()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    remove_previous_export_files(onnx_path, args.external_data_name)
    print(f"开始导出 ONNX 模型至: {onnx_path}")

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_inputs,
            str(onnx_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=args.constant_folding,
            input_names=["data", "constants", "hour_of_day", "day_of_year"],
            output_names=["output"],
            dynamic_axes={
                "data": {0: "batch_size"},
                "constants": {0: "batch_size"},
                "hour_of_day": {0: "batch_size"},
                "day_of_year": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            external_data=True,
            dynamo=False,
        )

    if args.external_data_name:
        normalize_external_data_path(onnx_path, args.external_data_name)

    print("ONNX 模型导出成功。")


def parse_args():
    parser = argparse.ArgumentParser(description="Export AiCast/NJU-Earth to ONNX.")
    parser.add_argument("--device", default="cpu", help="Export device. CPU is recommended for this large model.")
    parser.add_argument(
        "--allow-cuda-export",
        action="store_true",
        help="Allow CUDA export. Without this flag, CUDA requests are redirected to CPU to avoid GPU OOM.",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "aiweather/config/pretrain_config.yaml"),
        help="Model config path.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "aiweather/weights/finetune_6h_72/final.ckpt"),
        help="PyTorch checkpoint path.",
    )
    parser.add_argument(
        "--output",
        default=str(THIS_DIR / "NJU-Earth.onnx"),
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--external-data-name",
        default="NJU-Earth_weights.data",
        help="External tensor data file name stored next to the ONNX file.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--constant-folding",
        action="store_true",
        help="Enable ONNX constant folding. Disabled by default to reduce export memory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    export_aicast_to_onnx(parse_args())
