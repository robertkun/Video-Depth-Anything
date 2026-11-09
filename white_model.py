#!/usr/bin/env python3
"""用 Video-Depth-Anything 的深度，把视频渲染成白色模型。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
    if not frames:
        raise RuntimeError("视频没有解出任何画面")
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


def _boxes(data: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size = int.from_bytes(data[pos : pos + 4], "big")
        header = 8
        if size == 1 and pos + 16 <= end:
            size = int.from_bytes(data[pos + 8 : pos + 16], "big")
            header = 16
        if size < header or pos + size > end:
            break
        yield data[pos + 4 : pos + 8], pos, size, header
        pos += size


def _find_avcc(data: bytes, start: int, end: int):
    pos = start
    while pos < end:
        found = data.find(b"avcC", pos, end)
        if found < 4:
            return None
        size = int.from_bytes(data[found - 4 : found], "big")
        box_pos = found - 4
        if 11 <= size <= 4096 and box_pos + size <= end:
            return b"avcC", box_pos, size, 8
        pos = found + 4
    return None


def _child(data: bytes, start: int, end: int, name: bytes):
    for typ, pos, size, header in _boxes(data, start, end):
        if typ == name:
            return pos, size, header
    return None


def _walk_child(data: bytes, start: int, end: int, names: tuple[bytes, ...]):
    found = []

    def walk(box_start: int, box_end: int, depth: int) -> None:
        for typ, pos, size, header in _boxes(data, box_start, box_end):
            if typ in names:
                found.append((typ, pos, size, header))
            if typ in (b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts"):
                walk(pos + header, pos + size, depth + 1)

    walk(start, end, 0)
    return found


def _avc_parameter_sets(body: bytes) -> tuple[list[bytes], list[bytes], int] | None:
    if len(body) < 7:
        return None
    length_size = (body[4] & 3) + 1
    count = body[5] & 0x1F
    pos = 6
    sps = []
    for _ in range(count):
        if pos + 2 > len(body):
            return None
        length = int.from_bytes(body[pos : pos + 2], "big")
        pos += 2
        if length <= 0 or pos + length > len(body):
            return None
        sps.append(body[pos : pos + length])
        pos += length
    if pos >= len(body):
        return None
    pps_count = body[pos]
    pos += 1
    pps = []
    for _ in range(pps_count):
        if pos + 2 > len(body):
            return None
        length = int.from_bytes(body[pos : pos + 2], "big")
        pos += 2
        if length <= 0 or pos + length > len(body):
            return None
        pps.append(body[pos : pos + length])
        pos += length
    if not sps or not pps:
        return None
    return sps, pps, length_size


def _sample_is_avcc(sample: bytes, length_size: int) -> list[int] | None:
    pos = 0
    kinds = []
    while pos + length_size < len(sample):
        length = int.from_bytes(sample[pos : pos + length_size], "big")
        pos += length_size
        if length <= 0 or pos + length > len(sample):
            return None
        nal_type = sample[pos] & 0x1F
        if sample[pos] & 0x80 or nal_type == 0 or nal_type > 12:
            return None
        kinds.append(nal_type)
        pos += length
    if pos != len(sample) or not kinds:
        return None
    return kinds


def _video_samples(data: bytes):
    moov = _child(data, 0, len(data), b"moov")
    if moov is None:
        return None
    moov_pos, moov_size, moov_header = moov
    moov_end = moov_pos + moov_size
    for typ, pos, size, header in _boxes(data, moov_pos + moov_header, moov_end):
        if typ != b"trak":
            continue
        trak_end = pos + size
        hdlr = _child(data, pos + header, trak_end, b"mdia")
        if hdlr is None:
            continue
        mdia_pos, mdia_size, mdia_header = hdlr
        handler = _child(data, mdia_pos + mdia_header, mdia_pos + mdia_size, b"hdlr")
        if handler is None:
            continue
        handler_pos, _, handler_header = handler
        name = data[handler_pos + handler_header + 8 : handler_pos + handler_header + 12]
        if name != b"vide":
            continue
        boxes = _walk_child(data, pos + header, trak_end, (b"stsz", b"stco", b"stsc", b"mdhd", b"stts"))
        avcc = _find_avcc(data, pos, trak_end)
        stsz = next((item for item in boxes if item[0] == b"stsz"), None)
        stco = next((item for item in boxes if item[0] == b"stco"), None)
        stsc = next((item for item in boxes if item[0] == b"stsc"), None)
        mdhd = next((item for item in boxes if item[0] == b"mdhd"), None)
        stts = next((item for item in boxes if item[0] == b"stts"), None)
        if not all((avcc, stsz, stco, stsc, mdhd)):
            return None
        _, avcc_pos, avcc_size, avcc_header = avcc
        parsed = _avc_parameter_sets(data[avcc_pos + avcc_header : avcc_pos + avcc_size])
        if parsed is None:
            return None
        sps, pps, length_size = parsed
        _, stsz_pos, stsz_size, stsz_header = stsz
        stsz_body = data[stsz_pos + stsz_header : stsz_pos + stsz_size]
        if len(stsz_body) < 12:
            return None
        fixed = int.from_bytes(stsz_body[4:8], "big")
        count = int.from_bytes(stsz_body[8:12], "big")
        if fixed:
            sizes = [fixed] * count
        elif len(stsz_body) < 12 + count * 4:
            return None
        else:
            sizes = [int.from_bytes(stsz_body[12 + 4 * i : 16 + 4 * i], "big") for i in range(count)]
        _, stco_pos, stco_size, stco_header = stco
        stco_body = data[stco_pos + stco_header : stco_pos + stco_size]
        chunk_count = int.from_bytes(stco_body[4:8], "big")
        if len(stco_body) < 8 + chunk_count * 4:
            return None
        offsets = [int.from_bytes(stco_body[8 + 4 * i : 12 + 4 * i], "big") for i in range(chunk_count)]
        _, stsc_pos, stsc_size, stsc_header = stsc
        stsc_body = data[stsc_pos + stsc_header : stsc_pos + stsc_size]
        entry_count = int.from_bytes(stsc_body[4:8], "big")
        entries = []
        for index in range(entry_count):
            base = 8 + index * 12
            if base + 8 > len(stsc_body):
                return None
            entries.append(
                (
                    int.from_bytes(stsc_body[base : base + 4], "big"),
                    int.from_bytes(stsc_body[base + 4 : base + 8], "big"),
                )
            )
        per_chunk = []
        for index, (first, samples_per_chunk) in enumerate(entries):
            nxt = entries[index + 1][0] if index + 1 < len(entries) else chunk_count + 1
            per_chunk.extend([samples_per_chunk] * (nxt - first))
        if len(per_chunk) != len(offsets):
            return None
        samples = []
        sample_index = 0
        for offset, samples_per_chunk in zip(offsets, per_chunk):
            cursor = offset
            for _ in range(samples_per_chunk):
                if sample_index >= len(sizes):
                    return None
                size = sizes[sample_index]
                if cursor < 0 or size < 0 or cursor + size > len(data):
                    return None
                blob = data[cursor : cursor + size]
                samples.append({"blob": blob, "types": _sample_is_avcc(blob, length_size)})
                cursor += size
                sample_index += 1
        if sample_index != len(sizes):
            return None
        _, mdhd_pos, _, mdhd_header = mdhd
        version = data[mdhd_pos + mdhd_header]
        if version == 0:
            timescale = int.from_bytes(data[mdhd_pos + mdhd_header + 12 : mdhd_pos + mdhd_header + 16], "big")
        else:
            timescale = int.from_bytes(data[mdhd_pos + mdhd_header + 20 : mdhd_pos + mdhd_header + 24], "big")
        delta = 0
        if stts is not None:
            _, stts_pos, stts_size, stts_header = stts
            stts_body = data[stts_pos + stts_header : stts_pos + stts_size]
            if len(stts_body) >= 16:
                delta = int.from_bytes(stts_body[12:16], "big")
        fps = (timescale / delta) if timescale and delta else 30.0
        if not (1 <= fps <= 240):
            fps = 30.0
        return samples, sps, pps, fps, length_size
    return None


def _cv2_reads_frame(path: Path) -> bool:
    try:
        import cv2
    except ImportError:
        return False
    # 坏掉的分片会让 ffmpeg 往 stderr 刷一长串 NAL 错误，探测时先关掉。
    saved = os.dup(2)
    sink = os.open(os.devnull, os.O_WRONLY)
    os.dup2(sink, 2)
    os.close(sink)
    try:
        capture = cv2.VideoCapture(str(path))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
    finally:
        os.dup2(saved, 2)
        os.close(saved)
    return bool(ok and frame is not None and getattr(frame, "size", 0))


def _remux_valid_h264(source: Path) -> Path:
    parsed = _video_samples(source.read_bytes())
    if parsed is None:
        raise SystemExit("视频解不出画面：文件不是可读取的 H.264。")
    samples, sps, pps, fps, length_size = parsed
    valid = [sample for sample in samples if sample["types"]]
    if not valid:
        raise SystemExit("视频解不出画面：H.264 分片长度对不上，一帧都没有。")
    start = 0
    for index, sample in enumerate(valid):
        if 5 in sample["types"]:
            start = index
            break
    kept = valid[start:]
    dropped = len(samples) - len(kept)
    annexb = bytearray()
    start_code = b"\x00\x00\x00\x01"
    for blob in list(sps) + list(pps):
        annexb += start_code + blob
    for sample in kept:
        blob = sample["blob"]
        pos = 0
        while pos + length_size <= len(blob):
            length = int.from_bytes(blob[pos : pos + length_size], "big")
            pos += length_size
            annexb += start_code + blob[pos : pos + length]
            pos += length
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("视频开头的画面已损坏，且本机没有 ffmpeg，无法跳过这段坏数据。")
    handle, raw_name = tempfile.mkstemp(prefix="whitemodel-", suffix=".h264")
    os.close(handle)
    raw_path = Path(raw_name)
    out_handle, out_name = tempfile.mkstemp(prefix="whitemodel-", suffix=".mp4")
    os.close(out_handle)
    out_path = Path(out_name)
    try:
        raw_path.write_bytes(annexb)
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                f"{fps:.6f}",
                "-i",
                str(raw_path),
                "-c:v",
                "copy",
                "-an",
                str(out_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0 or not _cv2_reads_frame(out_path):
            detail = (result.stderr or result.stdout or "").strip()
            raise SystemExit(detail or "跳过损坏画面后仍然解不出帧。")
    except BaseException:
        out_path.unlink(missing_ok=True)
        raise
    finally:
        raw_path.unlink(missing_ok=True)
    report(5, f"已跳过开头 {dropped} 帧无法解码的画面，从第一个完整画面继续")
    return out_path


def _readable_source(source: Path) -> tuple[Path, Path | None]:
    if _cv2_reads_frame(source):
        return source, None
    report(4, "正在检查损坏的画面")
    repaired = _remux_valid_h264(source)
    return repaired, repaired


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

        readable, temporary_source = _readable_source(source)
        report(8, "正在读取视频")
        try:
            frames, fps = read_video_frames(
                str(readable),
                args.max_len,
                args.target_fps,
                args.max_res,
                progress=read_progress,
            )
        finally:
            if temporary_source is not None:
                temporary_source.unlink(missing_ok=True)

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
