from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from werkzeug.utils import secure_filename


def save_uploaded_video(file, upload_dir: str | Path, prefix: str) -> str:
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    original_name = secure_filename(getattr(file, "filename", "") or f"{prefix}.mp4")
    suffix = Path(original_name).suffix or ".mp4"
    target = upload_dir / f"{prefix}{suffix}"
    file.save(target)
    return str(target)


def _fallback_frame(output_path: Path, label: str) -> str:
    frame = np.full((720, 1280, 3), 247, dtype=np.uint8)
    cv2.rectangle(frame, (40, 40), (1240, 680), (229, 222, 210), 2)
    cv2.putText(frame, "Kiwami Capture", (80, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (17, 17, 17), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (184, 137, 69), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        "Placeholder frame generated because extraction was not available.",
        (80, 340),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (102, 102, 102),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), frame)
    return str(output_path)


def _candidate_frame_numbers(frame_number: int, total_frames: int) -> list[int]:
    candidates = [frame_number]
    for offset in (1, -1, 2, -2, 3, -3):
        candidate = frame_number + offset
        if 0 <= candidate < total_frames and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def extract_key_frames(video_path: str, output_dir: str | Path, max_frames: int = 6) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[str] = []

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        for index in range(max_frames):
            frame_paths.append(_fallback_frame(output_dir / f"fallback_{index + 1}.png", f"Frame {index + 1}"))
        return frame_paths

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        capture.release()
        for index in range(max_frames):
            frame_paths.append(_fallback_frame(output_dir / f"fallback_{index + 1}.png", f"Frame {index + 1}"))
        return frame_paths

    sample_count = min(max_frames, total_frames)
    sample_indexes = sorted({int(round(x)) for x in np.linspace(0, max(total_frames - 1, 0), sample_count)})
    if not sample_indexes:
        sample_indexes = [0]

    for index, frame_number in enumerate(sample_indexes, start=1):
        frame = None
        for candidate in _candidate_frame_numbers(frame_number, total_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, candidate)
            ok, candidate_frame = capture.read()
            if ok and candidate_frame is not None:
                frame = candidate_frame
                break
        if frame is None:
            path = output_dir / f"fallback_{index}.png"
            frame_paths.append(_fallback_frame(path, f"Frame {index}"))
            continue
        height, width = frame.shape[:2]
        if width > 1280:
            ratio = 1280 / float(width)
            frame = cv2.resize(frame, (1280, max(1, int(height * ratio))))
        path = output_dir / f"frame_{index:02d}.png"
        cv2.imwrite(str(path), frame)
        frame_paths.append(str(path))

    capture.release()

    if not frame_paths:
        for index in range(max_frames):
            frame_paths.append(_fallback_frame(output_dir / f"fallback_{index + 1}.png", f"Frame {index + 1}"))

    return frame_paths


def create_frame_manifest(frame_paths: list[str]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for index, path in enumerate(frame_paths, start=1):
        file_path = Path(path)
        manifest.append(
            {
                "index": index,
                "filename": file_path.name,
                "path": str(file_path),
                "label": f"Key frame {index}",
            }
        )
    return manifest


def _select_ghost_motion_frames(master_frames: list[dict[str, Any]]) -> list[str]:
    if not master_frames:
        return []
    real_paths: list[str] = []
    for frame in master_frames:
        path = str(frame.get("path", "")).strip()
        if not path:
            continue
        if not Path(path).name.startswith("fallback_"):
            real_paths.append(path)
    if not real_paths:
        return []
    source = real_paths
    if len(source) == 1:
        return [source[0]]
    target_count = min(4, len(source))
    if target_count == 2:
        indexes = [0, len(source) - 1]
    else:
        step = (len(source) - 1) / float(target_count - 1)
        indexes = []
        for i in range(target_count):
            index = int(round(i * step))
            if index not in indexes:
                indexes.append(index)
    return [source[index] for index in indexes]


def _center_crop(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    crop_w = max(1, int(width * 0.82))
    crop_h = max(1, int(height * 0.82))
    left = max(0, (width - crop_w) // 2)
    top = max(0, (height - crop_h) // 2)
    return image[top : top + crop_h, left : left + crop_w]


def _label_box(image: np.ndarray, text: str, x: int, y: int) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    padding_x = 10
    padding_y = 8
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    left = max(0, x)
    top = max(0, y)
    right = min(image.shape[1] - 1, left + text_width + padding_x * 2)
    bottom = min(image.shape[0] - 1, top + text_height + padding_y * 2)
    overlay = image.copy()
    cv2.rectangle(overlay, (left, top), (right, bottom), (247, 243, 234), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)
    cv2.rectangle(image, (left, top), (right, bottom), (184, 137, 69), 1)
    text_x = left + padding_x
    text_y = top + padding_y + text_height
    cv2.putText(image, text, (text_x, text_y), font, scale, (17, 17, 17), thickness, cv2.LINE_AA)


def create_ghost_motion_overlay(master_frames: list[dict[str, Any]], output_dir: str | Path) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_paths = _select_ghost_motion_frames(master_frames)
    if not selected_paths:
        return ""

    images: list[np.ndarray] = []
    for path in selected_paths:
        image = cv2.imread(path)
        if image is not None:
            images.append(_center_crop(image))
    if not images:
        return ""

    base_width, base_height = 1280, 720
    normalized: list[np.ndarray] = []
    for image in images[:4]:
        image = cv2.resize(image, (base_width, base_height), interpolation=cv2.INTER_AREA)
        normalized.append(image.astype(np.float32))

    if len(normalized) == 1:
        weights = [1.0]
    elif len(normalized) == 2:
        weights = [0.65, 0.35]
    elif len(normalized) == 3:
        weights = [0.5, 0.3, 0.2]
    else:
        weights = [0.4, 0.25, 0.2, 0.15]

    composite = np.zeros((base_height, base_width, 3), dtype=np.float32)
    for image, weight in zip(normalized, weights):
        composite += image * weight

    composite = np.clip(composite, 0, 255).astype(np.uint8)
    labels = ["START", "STABILIZE", "PAUSE", "RELEASE"]
    positions = [
        (24, 24),
        (base_width - 260, 24),
        (24, base_height - 72),
        (base_width - 260, base_height - 72),
    ]
    cv2.rectangle(composite, (16, 16), (base_width - 16, base_height - 16), (184, 137, 69), 2)
    for label, (x, y) in zip(labels, positions):
        _label_box(composite, label, x, y)

    output_path = output_dir / "ghost_motion_overlay.png"
    cv2.imwrite(str(output_path), composite)
    return str(output_path)


def safe_video_metadata(video_path: str) -> dict[str, Any]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return {
            "path": video_path,
            "exists": False,
            "duration_seconds": 0,
            "fps": 0,
            "frame_count": 0,
            "width": 0,
            "height": 0,
        }

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = round(frame_count / fps, 2) if fps > 0 and frame_count > 0 else 0
    capture.release()
    return {
        "path": video_path,
        "exists": True,
        "duration_seconds": duration,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }
