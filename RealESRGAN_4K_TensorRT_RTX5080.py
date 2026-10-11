import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

import os
import warnings
import logging

# Suppress Python warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Reduce Python logging output
logging.disable(logging.WARNING)

# Reduce TensorRT logging
os.environ["TRT_LOGGER_SEVERITY"] = "ERROR"

# Reduce PyTorch diagnostic logging
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_LOGS"] = "-all"

# Reduce TensorFlow logging if installed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

INPUT_FOLDER = Path(".")
OUTPUT_FOLDER = Path("ai_upscaled")
MODEL_WEIGHTS = Path("weights/RealESRGAN_x4plus.pth")
WEIGHTS_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.1.0/RealESRGAN_x4plus.pth")
GPU_ID = 0
TILE_SIZE = 256            # Fixed TensorRT input side; try 384 or 512 after testing
TILE_PAD = 12              # Overlap per side, in input pixels
HALF_PRECISION = True
PRE_DOWNSCALE = False      # True: 1080p -> 540p -> AI 4x -> 4K; faster, less detail
TARGET_WIDTH, TARGET_HEIGHT = 3840, 2160
NVENC_PRESET = "p4"
NVENC_QUALITY = 19
PRINT_EVERY_SECONDS = 2
EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi"}


def fmt(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}"


def probe(path, ffprobe):
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=True
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    fps = Fraction(stream["avg_frame_rate"])
    if fps <= 0:
        raise ValueError("Invalid frame rate")
    count = stream.get("nb_frames")
    if count and count != "N/A":
        count = int(count)
    else:
        duration = float(stream.get("duration") or data["format"]["duration"])
        count = max(1, round(duration * float(fps)))
    return int(stream["width"]), int(stream["height"]), str(fps), count


class TensorRTUpsampler:
    """Compile RRDBNet with Torch-TensorRT and process fixed-shape tiles.

    TensorRT executes the neural network. Frame conversion/tiling remains in
    Python/PyTorch. Compilation is performed once per script run.
    """

    def __init__(self):
        import torch
        import torch_tensorrt
        from basicsr.archs.rrdbnet_arch import RRDBNet

        if TILE_SIZE <= 2 * TILE_PAD:
            raise ValueError("TILE_SIZE must exceed twice TILE_PAD")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; install RTX 50-series compatible PyTorch")
        torch.cuda.set_device(GPU_ID)
        self.device = torch.device(f"cuda:{GPU_ID}")
        self.dtype = torch.float16 if HALF_PRECISION else torch.float32
        print("GPU:", torch.cuda.get_device_name(GPU_ID))
        print(f"TensorRT fixed tile: {TILE_SIZE}x{TILE_SIZE}, overlap: {TILE_PAD}")
        if not MODEL_WEIGHTS.exists():
            MODEL_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading official x4plus weights...")
            urllib.request.urlretrieve(WEIGHTS_URL, MODEL_WEIGHTS)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        state = torch.load(MODEL_WEIGHTS, map_location="cpu", weights_only=True)
        state = state.get("params_ema", state.get("params", state))
        model.load_state_dict(state, strict=True)
        model = model.eval().to(self.device, dtype=self.dtype)
        print("Compiling RealESRGAN-x4plus for TensorRT (first run can take minutes)...")
        started = time.monotonic()
        try:
            self.engine = torch_tensorrt.compile(
                model,
                ir="dynamo",
                inputs=[torch_tensorrt.Input(
                    shape=(1, 3, TILE_SIZE, TILE_SIZE), dtype=self.dtype)],
                enabled_precisions={self.dtype},
                require_full_compilation=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "TensorRT compilation failed; check torch-tensorrt/TensorRT "
                "versions and Blackwell support. No PyTorch fallback was used."
            ) from exc
        print(f"TensorRT compilation completed in {fmt(time.monotonic()-started)}")
        del model
        torch.cuda.empty_cache()

    def enhance(self, bgr):
        import torch
        import torch.nn.functional as F
        h, w = bgr.shape[:2]
        # Normalize BGR uint8 -> RGB float tensor on CUDA.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = (torch.from_numpy(rgb.copy()).permute(2, 0, 1)
                  .unsqueeze(0).to(self.device, dtype=self.dtype) / 255.0)
        core = TILE_SIZE - 2 * TILE_PAD
        # Replicate-padding supports even very small images.
        padded = F.pad(tensor, (TILE_PAD, TILE_PAD, TILE_PAD, TILE_PAD), mode="replicate")
        output = torch.empty((1, 3, h * 4, w * 4), device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            for y in range(0, h, core):
                for x in range(0, w, core):
                    actual_h = min(core, h-y)
                    actual_w = min(core, w-x)
                    patch = padded[:, :, y:y+actual_h+2*TILE_PAD,
                                   x:x+actual_w+2*TILE_PAD]
                    pad_bottom = TILE_SIZE - patch.shape[2]
                    pad_right = TILE_SIZE - patch.shape[3]
                    if pad_bottom or pad_right:
                        patch = F.pad(patch, (0, pad_right, 0, pad_bottom), mode="replicate")
                    predicted = self.engine(patch.contiguous())
                    if isinstance(predicted, (tuple, list)):
                        predicted = predicted[0]
                    output[:, :, y*4:(y+actual_h)*4, x*4:(x+actual_w)*4] = (
                        predicted[:, :, TILE_PAD*4:(TILE_PAD+actual_h)*4,
                                  TILE_PAD*4:(TILE_PAD+actual_w)*4]
                    )
        result = (output.squeeze(0).permute(1, 2, 0).float()
                  .clamp(0, 1).mul(255).byte().cpu().numpy())
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def create_upsampler():
    return TensorRTUpsampler()


def read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        block = stream.read(remaining)
        if not block:
            if remaining == size:
                return None
            raise RuntimeError("Truncated raw frame from FFmpeg decoder")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def suspicious_black(source, result):
    """Reject near-black output when the input has visible detail."""
    src = cv2.resize(source, (160, 90), interpolation=cv2.INTER_AREA)
    dst = cv2.resize(result, (160, 90), interpolation=cv2.INTER_AREA)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    return (src_gray.mean() > 20 and src_gray.std() > 8
            and dst_gray.mean() < 4 and dst_gray.std() < 3)


def source_timestamps(path, ffprobe):
    """Return decoded video-frame presentation times, in display order."""
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "json", str(path)
    ]
    data = json.loads(subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=True
    ).stdout)
    frames = data.get("frames", [])
    times = []
    for frame in frames:
        stamp = frame.get("best_effort_timestamp_time")
        if stamp is None or stamp == "N/A":
            raise RuntimeError("Source frame timestamps unavailable; cannot safely preserve timing")
        times.append(float(stamp))
    if not times:
        raise RuntimeError("No presentation timestamps found")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise RuntimeError("Non-increasing source timestamps; manual timing normalization needed")
    return times


def source_duration(path, ffprobe):
    data = json.loads(subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True
    ).stdout)
    return float(data["format"]["duration"])


def concat_escape(path):
    # ffconcat uses forward slashes and single-quoted paths.
    return "'" + path.resolve().as_posix().replace("'", "'\\''") + "'"


def process_video(video, ffmpeg, ffprobe, upsampler):
    target = OUTPUT_FOLDER / f"{video.stem}_AI_4K.mp4"
    if target.exists():
        print("Skipping existing output:", target)
        return

    width, height, fps, estimated = probe(video, ffprobe)
    timestamps = source_timestamps(video, ffprobe)
    duration = source_duration(video, ffprobe)
    work_width, work_height = width, height
    if PRE_DOWNSCALE:
        ratio = min(1.0, TARGET_WIDTH / (4 * width), TARGET_HEIGHT / (4 * height))
        work_width = max(2, int(width * ratio) // 2 * 2)
        work_height = max(2, int(height * ratio) // 2 * 2)

    print(f"Source: {width}x{height}; {len(timestamps)} frames; "
          f"duration {fmt(duration)}; source average FPS {fps}")
    print("Preserving original frame presentation times (VFR-safe).")
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="realesrgan_pts_") as temp:
        root = Path(temp)
        frames_dir = root / "enhanced"
        frames_dir.mkdir()
        partial = root / "encoded.mp4"
        concat_file = root / "frames.ffconcat"

        decoder_cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(video),
            "-map", "0:v:0", "-an", "-sn"
        ]
        if (work_width, work_height) != (width, height):
            decoder_cmd += ["-vf", f"scale={work_width}:{work_height}:flags=lanczos"]
        decoder_cmd += ["-fps_mode", "passthrough", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]

        frame_bytes = work_width * work_height * 3
        processed = 0
        start = last_print = time.monotonic()
        with tempfile.TemporaryFile() as decode_log:
            decoder = subprocess.Popen(
                decoder_cmd, stdout=subprocess.PIPE, stderr=decode_log, bufsize=0
            )
            try:
                while True:
                    raw = read_exact(decoder.stdout, frame_bytes)
                    if raw is None:
                        break
                    if processed >= len(timestamps):
                        raise RuntimeError(
                            (f"Frame-count mismatch: FFmpeg decoded at least {processed + 1} "
                             f"frames, but FFprobe reported {len(timestamps)} timestamps. "
                             "Check frame synchronization and source timestamps.")
                        )
                    source = np.frombuffer(raw, dtype=np.uint8).reshape(
                        work_height, work_width, 3
                    )
                    enhanced = upsampler.enhance(source)
                    if suspicious_black(source, enhanced):
                        raise RuntimeError(
                            f"Black AI frame detected at {processed + 1}; output withheld"
                        )
                    scale = min(
                        TARGET_WIDTH / enhanced.shape[1],
                        TARGET_HEIGHT / enhanced.shape[0]
                    )
                    fitted_w = max(2, min(
                        TARGET_WIDTH, round(enhanced.shape[1] * scale / 2) * 2
                    ))
                    fitted_h = max(2, min(
                        TARGET_HEIGHT, round(enhanced.shape[0] * scale / 2) * 2
                    ))
                    resized = cv2.resize(
                        enhanced, (fitted_w, fitted_h),
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
                    )
                    canvas = np.zeros(
                        (TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8
                    )
                    x = (TARGET_WIDTH - fitted_w) // 2
                    y = (TARGET_HEIGHT - fitted_h) // 2
                    canvas[y:y + fitted_h, x:x + fitted_w] = resized
                    filename = frames_dir / f"frame_{processed:08d}.png"
                    if not cv2.imwrite(
                        str(filename), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 1]
                    ):
                        raise RuntimeError(f"Cannot write {filename}")
                    processed += 1
                    now = time.monotonic()
                    if now - last_print >= PRINT_EVERY_SECONDS:
                        elapsed = now - start
                        rate = processed / max(elapsed, .001)
                        eta = fmt((len(timestamps) - processed) / rate)
                        print(
                            f"\rAI frames: {processed}/{len(timestamps)} "
                            f"({100 * processed / len(timestamps):.1f}%) "
                            f"| {rate:.2f} FPS | elapsed {fmt(elapsed)} | ETA {eta}    ",
                            end="", flush=True
                        )
                        last_print = now

                decoder.stdout.close()
                rc = decoder.wait()
                print()
                if rc:
                    decode_log.seek(0)
                    raise RuntimeError(
                        decode_log.read().decode("utf-8", "replace")[-2000:]
                    )
                if processed != len(timestamps):
                    raise RuntimeError(
                        f"Frame count mismatch: decoded {processed}, "
                        f"timestamps {len(timestamps)}"
                    )
            except BaseException:
                if decoder.poll() is None:
                    decoder.kill()
                decoder.wait()
                raise
            finally:
                if decoder.stdout and not decoder.stdout.closed:
                    decoder.stdout.close()

        # FFmpeg concat demuxer uses explicit per-frame durations. The
        # final repeated file ensures the final duration is applied.
        relative_times = [t - timestamps[0] for t in timestamps]
        last_duration = duration - relative_times[-1]
        if last_duration <= 0:
            last_duration = (
                relative_times[-1] - relative_times[-2]
                if processed > 1 else 1 / float(Fraction(fps))
            )
        with concat_file.open("w", encoding="utf-8") as f:
            f.write("ffconcat version 1.0\n")
            for i in range(processed):
                filename = frames_dir / f"frame_{i:08d}.png"
                f.write(f"file {concat_escape(filename)}\n")
                interval = (
                    relative_times[i + 1] - relative_times[i]
                    if i + 1 < processed else last_duration
                )
                f.write(f"duration {interval:.9f}\n")
            f.write(f"file {concat_escape(frames_dir / f'frame_{processed-1:08d}.png')}\n")

        print("Encoding 4K video with source timestamps...")
        cmd = [
            ffmpeg, "-hide_banner", "-y", "-loglevel", "error",
            "-safe", "0", "-f", "concat", "-i", str(concat_file),
            "-i", str(video), "-map", "0:v:0", "-map", "1:a?",
            "-fps_mode", "vfr", "-enc_time_base", "1:90000",
            "-c:v", "hevc_nvenc", "-preset", NVENC_PRESET,
            "-rc", "vbr", "-cq", str(NVENC_QUALITY), "-b:v", "0",
            "-pix_fmt", "yuv420p", "-c:a", "copy",
            "-t", f"{duration:.6f}", "-movflags", "+faststart",
            str(partial)
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace"
        )
        if result.returncode:
            raise RuntimeError("Encoding failed:\n" + result.stderr[-3000:])

        if not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError("Encoded output is empty")
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-show_entries", "format=duration", "-of", "json", str(partial)],
            capture_output=True, text=True, check=True
        )
        info = json.loads(result.stdout)
        video_stream = info["streams"][0]
        if (video_stream["width"], video_stream["height"]) != (
            TARGET_WIDTH, TARGET_HEIGHT
        ):
            raise RuntimeError("Output resolution mismatch")
        output_duration = float(info["format"]["duration"])
        if abs(output_duration - duration) > max(1.0, duration * 0.01):
            raise RuntimeError(
                f"Output duration {output_duration:.2f}s differs from "
                f"source {duration:.2f}s; output withheld"
            )
        shutil.move(str(partial), str(target))
        print(f"Saved: {target} | source {duration:.2f}s, output {output_duration:.2f}s")

def main():
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("FFmpeg and FFprobe must be on PATH")
    upsampler = create_upsampler()
    videos = sorted(
        p for p in INPUT_FOLDER.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    print(f"Found {len(videos)} video(s).")
    for index, video in enumerate(videos, 1):
        print(f"\n[{index}/{len(videos)}] {video.name}")
        try:
            process_video(video, ffmpeg, ffprobe, upsampler)
        except Exception as exc:
            print("ERROR:", exc)


if __name__ == "__main__":
    main()
