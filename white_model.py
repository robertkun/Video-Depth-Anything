#!/usr/bin/env python3
"""用 Video-Depth-Anything 的深度，把视频渲染成白色模型。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MISSING_DEPS = (
    "转白模还缺依赖。请在运行任务台的 Python 里安装 numpy、PyTorch，"
    "以及 Video-Depth-Anything 的 requirements.txt。"
)


def report(percent: int, message: str) -> None:
    print(f"WHITE_MODEL {max(0, min(100, int(percent)))} {message}", flush=True)


def clay_params(contrast: int, relief: int) -> tuple[float, float, float]:
    """对比 0–100 拉开远近，体积 0–100 加强表面明暗。默认 40/50 就是当前效果。"""
    contrast_value = max(0, min(100, int(contrast)))
    relief_value = max(0, min(100, int(relief)))
    power = 0.65 + (contrast_value / 100.0) * 2.0
    scale = 8.0 + (relief_value / 100.0) * 32.0
    shade_mix = 0.45 + (relief_value / 100.0) * 0.54
    return power, scale, shade_mix


def clay_frames(depths, progress=None, contrast: int = 50, relief: int = 30):
    """把深度图着成白色模型：近处白、远处暗，再用侧面光画出体积。"""
    import numpy as np

    power, scale, shade_mix = clay_params(contrast, relief)
    light = np.array([0.62, 0.18, 0.42], dtype=np.float32)
    light /= np.linalg.norm(light)
    frames = []
    total = len(depths)
    for index, depth in enumerate(depths):
        values = depth.astype(np.float32)
        low, high = np.percentile(values, [2, 98])
        span = float(high - low) if high > low else 1.0
        near = np.clip((values - low) / span, 0.0, 1.0)
        tone = np.power(near, power)
        gradient_y, gradient_x = np.gradient(near)
        normal = np.stack((-gradient_x * scale, -gradient_y * scale, np.ones_like(near)), axis=-1)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6
        shade = np.clip(normal @ light, 0.0, 1.0)
        gray = np.clip(tone * ((1.0 - shade_mix) + shade_mix * shade), 0.0, 1.0)
        color = (gray * 255.0).astype(np.uint8)
        frames.append(np.repeat(color[..., None], 3, axis=-1))
        if progress is not None and (index % 10 == 0 or index + 1 == total):
            progress(index + 1, total)
    return np.stack(frames, axis=0)


def load_model(encoder: str, device: str):
    import torch
    from video_depth_anything.video_depth import VideoDepthAnything

    configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    checkpoint = ROOT / "checkpoints" / f"video_depth_anything_{encoder}.pth"
    if not checkpoint.is_file():
        raise SystemExit(
            "还没有 Video-Depth-Anything 的模型权重。"
            "请在 tools/Video-Depth-Anything 目录执行 bash get_weights.sh 后再试。"
        )
    model = VideoDepthAnything(**configs[encoder])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    return model.to(device).eval()


def load_depth_cache(path: Path, max_res: int):
    if not path.is_file():
        return None
    try:
        import numpy as np

        with np.load(path) as data:
            if int(data["max_res"]) != int(max_res):
                return None
            depths = np.asarray(data["depths"], dtype=np.float32)
            fps = float(data["fps"])
        if depths.ndim != 3 or depths.shape[0] == 0:
            return None
        return depths, fps
    except (OSError, ValueError, KeyError):
        return None


def save_depth_cache(path: Path, depths, fps: float, max_res: int) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        depths=np.asarray(depths, dtype=np.float16),
        fps=np.float32(fps),
        max_res=np.int32(max_res),
    )


def write_settings(output: Path, contrast: int, relief: int, max_res: int) -> None:
    payload = {"contrast": int(contrast), "relief": int(relief), "max_res": int(max_res)}
    output.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="把视频转成白色模型")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--max-res", type=int, default=640)
    parser.add_argument("--max-len", type=int, default=-1)
    parser.add_argument("--target-fps", type=int, default=-1)
    parser.add_argument("--contrast", type=int, default=50)
    parser.add_argument("--relief", type=int, default=30)
    parser.add_argument("--depth-cache", default="")
    args = parser.parse_args()

    from utils.dc_utils import read_video_frames, save_video

    source = Path(args.input)
    if not source.is_file():
        raise SystemExit(f"找不到视频：{source}")
    cache_path = Path(args.depth_cache) if args.depth_cache else None
    cached = load_depth_cache(cache_path, args.max_res) if cache_path else None
    if cached is None:
        try:
            import torch
        except ImportError as exc:
            raise SystemExit(MISSING_DEPS) from exc
        if torch.cuda.is_available():
            device = "cuda"
            use_fp32 = False
        else:
            device = "cpu"
            use_fp32 = True

        report(2, "正在加载模型")
        model = load_model(args.encoder, device)

        def read_progress(done, total):
            span = max(int(total), 1)
            report(8 + int(7 * done / span), f"正在读取视频 {done}/{total}")

        report(8, "正在读取视频")
        frames, fps = read_video_frames(
            str(source),
            args.max_len,
            args.target_fps,
            args.max_res,
            progress=read_progress,
        )

        def depth_progress(done, total):
            span = max(int(total), 1)
            report(15 + int(65 * done / span), f"正在估计深度 {done}/{total}")

        report(15, "正在估计深度")
        depths, fps = model.infer_video_depth(
            frames,
            fps,
            input_size=args.input_size,
            device=device,
            fp32=use_fp32,
            progress=depth_progress,
        )
        if cache_path is not None:
            save_depth_cache(cache_path, depths, fps, args.max_res)
    else:
        depths, fps = cached
        report(80, "正在按新参数生成白模")

    def shade_progress(done, total):
        span = max(int(total), 1)
        report(80 + int(8 * done / span), f"正在生成白模 {done}/{total}")

    if cached is None:
        report(80, "正在生成白模")
    shaded = clay_frames(
        depths,
        progress=shade_progress,
        contrast=args.contrast,
        relief=args.relief,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part.mp4")

    def save_progress(done, total):
        span = max(int(total), 1)
        report(88 + int(11 * done / span), f"正在写出视频 {done}/{total}")

    report(88, "正在写出视频")
    save_video(shaded, str(temporary), fps=fps, progress=save_progress)
    temporary.replace(output)
    write_settings(output, args.contrast, args.relief, args.max_res)
    report(100, "白模已生成")
    print(f"已写出 {output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
