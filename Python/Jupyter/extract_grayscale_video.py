#!/usr/bin/env python3
"""
Extract grayscale frames from a video and save them as a new video (preserving original audio).

Pipeline:
  1. Build a downscaled analysis copy of the original video with FFmpeg
     (scale=iw/4:ih/4, same fps, no audio, fps_mode passthrough).
  2. Detect grayscale frames on the analysis copy using a tolerance-based
     check (because lossy compression slightly breaks R==G==B).
  3. Cut segments from the ORIGINAL video using the detected frame indices,
     then concatenate them preserving the audio track.

Dependencies: pip install opencv-python numpy
System dependency: FFmpeg (ffmpeg command must be in PATH)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Frame-level grayscale detection (tolerance-based, works on compressed copy)
# ---------------------------------------------------------------------------

def is_grayscale_frame_fast(
    frame_bgr: np.ndarray,
    threshold: float = 0.9,
    tol: int = 3,
) -> bool:
    """
    Return True if the frame is grayscale (or near-grayscale).

    Uses a tolerance `tol` because lossy compression (especially 4:2:0
    chroma subsampling) can make a perfectly gray frame slightly colored.

    A pixel is considered "gray" if |R-G| <= tol and |G-B| <= tol.
    The frame is considered grayscale if the share of gray pixels > threshold.
    """
    b = frame_bgr[..., 0].astype(np.int16)
    g = frame_bgr[..., 1].astype(np.int16)
    r = frame_bgr[..., 2].astype(np.int16)
    gray_mask = (np.abs(r - g) <= tol) & (np.abs(g - b) <= tol)
    return bool(gray_mask.mean() > threshold)


# ---------------------------------------------------------------------------
# Step 1: build a downscaled analysis copy
# ---------------------------------------------------------------------------

def build_analysis_copy(
    video_path: str,
    analysis_path: str,
    scale_factor: int = 4,
) -> None:
    vf = (
        f"scale="
        f"trunc(iw/{scale_factor}/2)*2:"
        f"trunc(ih/{scale_factor}/2)*2"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", vf,
        # "-r" УБРАН — FFmpeg возьмёт fps из входа
        "-fps_mode", "passthrough",  # кадры не дублируются и не выкидываются
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "35",
        "-pix_fmt", "yuv420p",
        analysis_path,
    ]
    print(f"Building analysis copy (scale=1/{scale_factor}, inherit fps)...")
    print("ffmpeg:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def count_frames_and_fps(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total, fps


# ---------------------------------------------------------------------------
# Step 2: detect grayscale frames on the analysis copy
# ---------------------------------------------------------------------------

def detect_grayscale_frames(
    analysis_path: str,
    threshold: float = 0.9,
    tol: int = 3,
):
    """Return (list_of_indices, fps) detected on the analysis copy."""
    cap = cv2.VideoCapture(analysis_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open analysis copy: {analysis_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Unable to read video frame rate.")

    indices = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if is_grayscale_frame_fast(frame, threshold=threshold, tol=tol):
            indices.append(idx)
        idx += 1

    cap.release()
    return indices, fps


def group_consecutive(indices, max_gap: int = 0):
    """Group consecutive (or nearly consecutive) frame indices into (start, end) ranges."""
    if not indices:
        return []
    ranges = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i - prev <= max_gap + 1:
            prev = i
        else:
            ranges.append((start, prev))
            start = prev = i
    ranges.append((start, prev))
    return ranges


# ---------------------------------------------------------------------------
# Step 3: cut segments from the ORIGINAL and concatenate
# ---------------------------------------------------------------------------

def extract_and_concat_segments(
    original_video: str,
    frame_ranges,
    fps: float,
    output_path: str,
    crf: int = 18,
    preset: str = "slow",
) -> None:
    """
    Cut frame ranges from the ORIGINAL video and concatenate.
    - Video is re-encoded (to get exact frame-accurate boundaries).
    - Audio is copied during segmentation and re-encoded to AAC on concat
      to avoid timestamp drift.
    """
    if not frame_ranges:
        print("No grayscale frames detected; no output generated.")
        return

    original_video = str(Path(original_video).resolve())
    output_path = str(Path(output_path).resolve())

    with tempfile.TemporaryDirectory(prefix="gray_frames_") as tmpdir:
        tmpdir = Path(tmpdir)
        segment_files = []

        for i, (start_frame, end_frame) in enumerate(frame_ranges):
            start_time = start_frame / fps
            duration = (end_frame - start_frame + 1) / fps
            seg_path = tmpdir / f"seg_{i:04d}.mp4"

            # -ss after -i: frame-accurate. -vf setsar=1 keeps pixel aspect
            # square without changing resolution (prevents surprise rescaling).
            cmd = [
                "ffmpeg", "-y",
                "-i", original_video,
                "-ss", f"{start_time:.6f}",
                "-t", f"{duration:.6f}",
                "-vf", "setsar=1",
                "-c:v", "libx264",
                "-preset", preset,
                "-crf", str(crf),
                "-c:a", "copy",
                str(seg_path),
            ]
            print(
                f"Extracting segment {i + 1}/{len(frame_ranges)}: "
                f"frames {start_frame}-{end_frame}, "
                f"time {start_time:.3f}s + {duration:.3f}s"
            )
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            segment_files.append(seg_path)

        list_file = tmpdir / "concat_list.txt"
        with open(list_file, "w", encoding="utf-8") as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")

        # Re-encode audio on concat to avoid audio timestamp drift between
        # short segments; video can be copied because all segments share
        # identical encoding parameters.
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]
        print("Concatenating segments...")
        subprocess.run(concat_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"Done: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract grayscale frames from a video and preserve the original audio."
    )
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Path to the output video (default: <input_name>_grayscale.mp4)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.9,
        help="Share of gray pixels required to treat a frame as grayscale (default: 0.9)"
    )
    parser.add_argument(
        "--tol", type=int, default=3,
        help="Per-pixel channel tolerance |R-G| and |G-B| (default: 3). "
             "Needed because lossy compression slightly breaks R==G==B."
    )
    parser.add_argument(
        "--merge-gap", type=int, default=2,
        help="Merge ranges with gaps no greater than N frames (default: 2)"
    )
    parser.add_argument(
        "--scale-factor", type=int, default=4,
        help="Downscale factor for the analysis copy (default: 4)"
    )
    parser.add_argument(
        "--crf", type=int, default=18,
        help="CRF for cutting segments from the original (default: 18)"
    )
    parser.add_argument(
        "--preset", default="slow",
        help="x264 preset for cutting segments (default: slow)"
    )
    parser.add_argument(
        "--keep-analysis", action="store_true",
        help="Keep the intermediate analysis copy (for debugging)"
    )
    args = parser.parse_args()

    input_path = Path(args.input_video)
    if not input_path.exists():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        output_path = input_path.with_name(input_path.stem + "_grayscale.mp4")
    else:
        output_path = Path(args.output)

    # --- sanity check on the original ---
    orig_frames, orig_fps = count_frames_and_fps(str(input_path))
    print(f"Original: {orig_frames} frames, fps={orig_fps:.3f}")

    # --- step 1: analysis copy ---
    tmp_root = Path(tempfile.mkdtemp(prefix="gray_analysis_"))
    analysis_path = tmp_root / "analysis.mp4"
    try:
        build_analysis_copy(str(input_path), str(analysis_path), args.scale_factor)

        an_frames, an_fps = count_frames_and_fps(str(analysis_path))
        print(f"Analysis copy: {an_frames} frames, fps={an_fps:.3f}")

        # If frame counts differ, indices cannot be mapped 1:1 -> abort.
        if an_frames != orig_frames:
            print(
                "ERROR: analysis copy has a different frame count than the original "
                f"({an_frames} vs {orig_frames}). "
                "Frame indices cannot be mapped; aborting.",
                file=sys.stderr,
            )
            sys.exit(2)
        if abs(an_fps - orig_fps) > 1e-3:
            print(
                f"ERROR: fps mismatch ({an_fps:.3f} vs {orig_fps:.3f}); aborting.",
                file=sys.stderr,
            )
            sys.exit(2)

        # --- step 2: detect ---
        print("Detecting grayscale frames on the analysis copy...")
        gray_indices, _ = detect_grayscale_frames(
            str(analysis_path), threshold=args.threshold, tol=args.tol
        )
        print(f"Found {len(gray_indices)} grayscale frames.")

        if not gray_indices:
            print("No grayscale frames detected; exiting.")
            return

        frame_ranges = group_consecutive(gray_indices, max_gap=args.merge_gap)
        print(f"Merged into {len(frame_ranges)} time segment(s).")

        # --- step 3: cut from original ---
        extract_and_concat_segments(
            str(input_path),
            frame_ranges,
            orig_fps,
            str(output_path),
            crf=args.crf,
            preset=args.preset,
        )
    finally:
        if args.keep_analysis:
            keep_to = Path.cwd() / "analysis_debug.mp4"
            shutil.copy(analysis_path, keep_to)
            print(f"Analysis copy kept at: {keep_to}")
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()