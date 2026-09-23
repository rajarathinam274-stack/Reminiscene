"""Adaptive video keyframe selection.

Do NOT embed every frame.  Sample candidate frames at a base rate, then keep
only frames whose perceptual difference from the last kept keyframe exceeds
an adaptive threshold (scene/slide changes, significant visual difference).
Each keyframe retains: video source, timestamp, frame id, duration until next.

Requires FFmpeg; raises MissingDependency with a clear message otherwise.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Keyframe:
    frame_id: str
    path: Path
    time: float          # seconds into the video
    duration: float      # seconds this keyframe represents
    score: float         # visual-change score that triggered selection


class MissingDependency(Exception):
    pass


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise MissingDependency("FFmpeg is required for video keyframe extraction")
    return exe


def probe_duration(path: Path) -> Optional[float]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=20, check=True,
        )
        return float(json.loads(out.stdout)["format"]["duration"])
    except Exception:
        return None


def _mean_abs_diff_png(a: Path, b: Path) -> Optional[float]:
    """Cheap visual difference via Pillow grayscale downsampled images."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return None
    with Image.open(a) as ia, Image.open(b) as ib:
        ga = np.asarray(ia.convert("L").resize((64, 64)), dtype=np.float32)
        gb = np.asarray(ib.convert("L").resize((64, 64)), dtype=np.float32)
    return float(abs(ga - gb).mean() / 255.0)


def adaptive_keyframes(
    video_path: str | Path,
    out_dir: str | Path,
    sample_fps: float = 0.5,
    change_threshold: float = 0.12,
    max_keyframes: int = 120,
) -> list[Keyframe]:
    """Extract candidate frames, keep those marking meaningful visual change."""
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg()
    duration = probe_duration(video_path) or 0.0

    prefix = out_dir / "cand_%06d.png"
    interval = 1.0 / max(0.05, sample_fps)
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(video_path),
         "-vf", f"fps={sample_fps}", str(prefix)],
        check=True, timeout=3600,
    )
    candidates = sorted(out_dir.glob("cand_*.png"))
    if not candidates:
        return []

    kept: list[tuple[Path, float, float]] = [(candidates[0], 0.0, interval)]
    prev = candidates[0]
    # Adaptive threshold: running mean of diffs so static lectures still pick
    # up slide changes even when global motion is low.
    diffs: list[float] = []
    for i, cand in enumerate(candidates[1:], start=1):
        t = i * interval
        d = _mean_abs_diff_png(prev, cand)
        if d is None:
            # No Pillow/numpy: fall back to file-size heuristic delta
            d = abs(cand.stat().st_size - prev.stat().st_size) / max(1, prev.stat().st_size)
        diffs.append(d)
        thr = min(change_threshold, max(0.04, 0.6 * (sum(diffs) / len(diffs))))
        if d >= thr:
            kept.append((cand, t, interval))
            prev = cand
            if len(kept) >= max_keyframes:
                break

    keyframes: list[Keyframe] = []
    for idx, (p, t, iv) in enumerate(kept):
        kf_path = out_dir / f"key_{idx:04d}.png"
        p.replace(kf_path)
        dur = (kept[idx + 1][1] - t) if idx + 1 < len(kept) else max(iv, (duration - t) if duration else iv)
        keyframes.append(Keyframe(
            frame_id=f"{video_path.stem}#kf{idx:04d}",
            path=kf_path,
            time=round(t, 3),
            duration=round(max(0.1, dur), 3),
            score=1.0 if idx == 0 else round(min(1.0, diffs[min(idx, len(diffs) - 1)])),
        ))
    # cleanup unselected candidates
    for leftover in out_dir.glob("cand_*.png"):
        leftover.unlink(missing_ok=True)
    return keyframes
