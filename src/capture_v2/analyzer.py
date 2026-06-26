from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:
    mp = None

from src.capture_v2.sample_evidence import build_capture_v2_sample_data
from src.storage import OUTPUTS_DIR


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float | None:
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom <= 0:
        return None
    cosine = float(np.dot(ba, bc) / denom)
    cosine = _clamp(cosine, -1.0, 1.0)
    return float(math.degrees(math.acos(cosine)))


def _landmark_point(landmark: Any, width: int, height: int) -> tuple[float, float] | None:
    if landmark is None:
        return None
    x = getattr(landmark, 'x', None)
    y = getattr(landmark, 'y', None)
    if x is None or y is None:
        return None
    return float(x) * float(width), float(y) * float(height)


def _normalize_text(value: object, fallback: str = '') -> str:
    text = str(value or '').strip()
    return text if text else fallback


def _ordered_unique(indexes: list[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for index in indexes:
        if index in seen:
            continue
        seen.add(index)
        ordered.append(index)
    return ordered


def _sampling_indexes(total_frames: int, fps: float, sample_interval_sec: float, max_sampled_frames: int) -> list[int]:
    if max_sampled_frames <= 0:
        return []
    effective_fps = fps if fps > 0 else 30.0
    effective_interval = sample_interval_sec if sample_interval_sec > 0 else 0.5
    step = max(1, int(round(effective_fps * effective_interval)))
    if total_frames > 0:
        indexes = list(range(0, total_frames, step))
    else:
        indexes = [step * index for index in range(max_sampled_frames)]
    return indexes[:max_sampled_frames]


def _window_indexes(center_frame: int, radius_frames: int, total_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    start = max(0, center_frame - radius_frames)
    end = min(total_frames - 1, center_frame + radius_frames)
    return list(range(start, end + 1))


def _motion_centroid(diff: np.ndarray, frame_width: int, frame_height: int) -> dict[str, int] | None:
    if diff.size == 0:
        return None
    _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, 3)
    contours_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 18:
        return None
    moments = cv2.moments(contour)
    if not moments.get('m00'):
        return None
    x = int((moments['m10'] / moments['m00']) * (frame_width / float(diff.shape[1] or 1)))
    y = int((moments['m01'] / moments['m00']) * (frame_height / float(diff.shape[0] or 1)))
    return {'x': x, 'y': y}


def _detect_pose_angles(results: Any, frame_width: int, frame_height: int) -> tuple[dict[str, float | None], dict[str, int] | None, str]:
    angles: dict[str, float | None] = {'wrist': None, 'elbow': None, 'shoulder': None}
    trajectory_point: dict[str, int] | None = None
    evidence_detail = ''
    if results is None or getattr(results, 'pose_landmarks', None) is None:
        return angles, trajectory_point, evidence_detail

    pose_landmarks = results.pose_landmarks.landmark
    if mp is None:
        return angles, trajectory_point, evidence_detail

    side_candidates = []
    for side in ('LEFT', 'RIGHT'):
        try:
            shoulder = pose_landmarks[getattr(mp.solutions.pose.PoseLandmark, f'{side}_SHOULDER')]
            elbow = pose_landmarks[getattr(mp.solutions.pose.PoseLandmark, f'{side}_ELBOW')]
            wrist = pose_landmarks[getattr(mp.solutions.pose.PoseLandmark, f'{side}_WRIST')]
            hip = pose_landmarks[getattr(mp.solutions.pose.PoseLandmark, f'{side}_HIP')]
        except Exception:
            continue
        visibility = float(getattr(shoulder, 'visibility', 0.0) or 0.0) + float(getattr(elbow, 'visibility', 0.0) or 0.0) + float(getattr(wrist, 'visibility', 0.0) or 0.0)
        side_candidates.append((visibility, side, shoulder, elbow, wrist, hip))

    if not side_candidates:
        return angles, trajectory_point, evidence_detail

    _, side, shoulder_lm, elbow_lm, wrist_lm, hip_lm = max(side_candidates, key=lambda item: item[0])
    shoulder = _landmark_point(shoulder_lm, frame_width, frame_height)
    elbow = _landmark_point(elbow_lm, frame_width, frame_height)
    wrist = _landmark_point(wrist_lm, frame_width, frame_height)
    hip = _landmark_point(hip_lm, frame_width, frame_height)
    if shoulder and elbow and wrist:
        angles['elbow'] = _angle(shoulder, elbow, wrist)
        trajectory_point = {'x': int(wrist[0]), 'y': int(wrist[1])}
    if hip and shoulder and elbow:
        angles['shoulder'] = _angle(hip, shoulder, elbow)
    evidence_detail = f'{side.lower()} pose landmarks detected'
    return angles, trajectory_point, evidence_detail


def _detect_hand_angles(results: Any, frame_width: int, frame_height: int) -> tuple[dict[str, float | None], dict[str, int] | None, str]:
    angles: dict[str, float | None] = {'wrist': None, 'elbow': None, 'shoulder': None}
    trajectory_point: dict[str, int] | None = None
    evidence_detail = ''
    if results is None or getattr(results, 'multi_hand_landmarks', None) is None:
        return angles, trajectory_point, evidence_detail
    hand_landmarks = results.multi_hand_landmarks[0].landmark
    try:
        wrist = _landmark_point(hand_landmarks[0], frame_width, frame_height)
        index_mcp = _landmark_point(hand_landmarks[5], frame_width, frame_height)
        pinky_mcp = _landmark_point(hand_landmarks[17], frame_width, frame_height)
    except Exception:
        wrist = index_mcp = pinky_mcp = None
    if wrist and index_mcp and pinky_mcp:
        angles['wrist'] = _angle(index_mcp, wrist, pinky_mcp)
        trajectory_point = {'x': int(wrist[0]), 'y': int(wrist[1])}
    evidence_detail = 'hand landmarks detected'
    return angles, trajectory_point, evidence_detail


def _format_angle(timeline: list[dict[str, Any]], key: str) -> str:
    for item in timeline:
        signal = item.get('detected_signal') or {}
        if not isinstance(signal, dict):
            continue
        value = signal.get(key)
        if value is None:
            continue
        return f'{float(value):.1f}°'
    return 'Unavailable'


def _first_available_angle(timeline: list[dict[str, Any]], key: str) -> str:
    return _format_angle(timeline, key)


def _movement_speed_label(timeline: list[dict[str, Any]]) -> str:
    for item in timeline:
        signal = item.get('detected_signal') or {}
        if not isinstance(signal, dict):
            continue
        delta = signal.get('motion_delta')
        if delta is None:
            continue
        return f'motion delta {float(delta):.3f}'
    return 'Unavailable'


def _build_review_notes(analysis_mode: str, evidence_source: str, warnings: list[str]) -> list[str]:
    notes = [
        'Only sampled frames with extracted evidence are shown.',
        'Expert review is still required before Practice Memory is trusted.',
    ]
    if analysis_mode == 'focused':
        notes.insert(0, 'Focused analysis adds denser windows around candidate frames when possible.')
    elif analysis_mode == 'full':
        notes.insert(0, 'High Precision / Full Frame analysis inspects every readable frame.')
    else:
        notes.insert(0, 'Fast analysis samples local frames at a light interval.')
    if evidence_source == 'opencv_mediapipe':
        notes.append('MediaPipe landmarks were available on at least one sampled frame.')
    elif evidence_source == 'opencv_only':
        notes.append('MediaPipe was unavailable or did not yield usable landmarks, so OpenCV-only evidence was used.')
    if warnings:
        notes.append('Warnings stay local and are surfaced in the UI.')
    return notes


def _build_chat_prompts(analysis_mode: str, evidence_source: str, timeline: list[dict[str, Any]]) -> list[str]:
    if analysis_mode == 'full':
        return [
            'Which frame had the strongest supported evidence?',
            'Was any landmark data extracted?',
            'What did the full-frame pass confirm locally?',
        ]
    if analysis_mode == 'focused':
        return [
            'Which candidate frame had the strongest evidence?',
            'Did the dense window add any supported landmarks?',
            'What did the focused window confirm locally?',
        ]
    if evidence_source == 'opencv_mediapipe':
        return [
            'Which sampled frame had the strongest landmark evidence?',
            'Was any supported landmark data extracted?',
            'What do the extracted landmarks show?',
        ]
    if timeline:
        return [
            'Which sampled frame had the strongest motion delta?',
            'Was any supported evidence extracted?',
            'What does OpenCV-only evidence show?',
        ]
    return [
        'Why was no evidence extracted?',
        'What video format should I upload?',
        'What should I try next?',
    ]


def _build_chat_response(timeline: list[dict[str, Any]], analysis_mode: str, evidence_source: str, warnings: list[str]) -> str:
    if not timeline:
        if warnings:
            return f"No supported evidence was extracted. {' '.join(warnings[:2])}"
        return 'No supported evidence was extracted from the uploaded video.'
    detected = sum(1 for item in timeline if str(item.get('status') or '').lower() == 'detected')
    motion_values = [float((item.get('detected_signal') or {}).get('motion_delta') or 0.0) for item in timeline if isinstance(item, dict)]
    strongest_motion = max(motion_values) if motion_values else 0.0
    first = timeline[0]
    parts = [
        f"{len(timeline)} sampled frame(s) were analyzed locally.",
        f"Evidence source: {evidence_source}.",
    ]
    if analysis_mode == 'full':
        parts.append('A full-frame pass was requested and processed locally.')
    elif analysis_mode == 'focused':
        parts.append('A focused window pass added denser local sampling around candidate frames.')
    elif evidence_source == 'opencv_mediapipe':
        parts.append(f"MediaPipe found landmarks on {detected} sampled frame(s).")
    else:
        parts.append(f"OpenCV-only motion evidence was used on {detected} sampled frame(s).")
    parts.append(f"Strongest observed motion delta: {strongest_motion:.3f}.")
    parts.append(f"The first supported frame was {first.get('frame_id', '')} at {first.get('timestamp', '')}.")
    if warnings:
        parts.append(f"Warnings: {' '.join(warnings[:2])}")
    return ' '.join(part for part in parts if part)


def _build_empty_result(base: dict[str, Any], *, analysis_mode: str, analysis_mode_label: str, evidence_source: str, message: str, warnings: list[str], sampled_frame_count: int, frame_count: int, analysis_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    result = dict(base)
    result.update(
        {
            'analysis_mode': analysis_mode,
            'analysis_mode_label': analysis_mode_label,
            'analysis_source_label': evidence_source,
            'evidence_source': evidence_source,
            'analysis_status': 'error',
            'analysis_message': message,
            'warnings': warnings,
            'sampled_frame_count': sampled_frame_count,
            'frame_count': frame_count,
            'analysis_summary': analysis_summary or {},
            'sample_video': {
                'label': base.get('sample_video', {}).get('label', 'Local clip'),
                'source_video': base.get('sample_video', {}).get('source_video', ''),
                'calibration_status': 'unavailable',
                'overlay_labels': ['no supported evidence extracted'],
                'angles': {'wrist': 'Unavailable', 'elbow': 'Unavailable', 'shoulder': 'Unavailable'},
                'note': message,
            },
            'evidence_summary': {
                'timestamp': 'Unavailable',
                'frame_id': 'Unavailable',
                'calibration_status': analysis_mode,
                'wrist_angle': 'Unavailable',
                'movement_speed': 'Unavailable',
                'pause_detected': False,
                'confidence': 0.0,
                'status': 'not_detected',
                'analysis_mode': analysis_mode,
                'evidence_source': evidence_source,
            },
            'timeline': [],
            'review_notes': _build_review_notes(analysis_mode, evidence_source, warnings),
            'chat_prompts': _build_chat_prompts(analysis_mode, evidence_source, []),
            'export_preview_hint': 'Only Expert-confirmed items are treated as trusted Practice Memory.',
            'initial_chat_question': 'Why was no evidence extracted?',
            'initial_chat_response': _build_chat_response([], analysis_mode, evidence_source, warnings),
        }
    )
    return result


def analyze_local_capture_video(
    video_path: str,
    *,
    analysis_mode: str = 'fast',
    uploaded_name: str = '',
    sample_interval_sec: float = 0.5,
    max_sampled_frames: int = 12,
    focused_window_sec: float = 2.0,
    use_mediapipe: bool = True,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    base = build_capture_v2_sample_data()
    warnings: list[str] = []
    analysis_id = uuid.uuid4().hex[:12]
    output_root = Path(output_dir or (OUTPUTS_DIR / 'capture_v2' / analysis_id))
    frame_dir = output_root / 'frames'
    frame_dir.mkdir(parents=True, exist_ok=True)
    video_name = _normalize_text(uploaded_name, Path(video_path).name)
    mode = (analysis_mode or 'fast').strip().lower()
    if mode == 'sample':
        return base
    if mode not in {'fast', 'focused', 'full'}:
        warnings.append(f"Unknown analysis mode '{mode}', so fast analysis was used.")
        mode = 'fast'

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        warnings.append('OpenCV could not open the uploaded video, so no real evidence was extracted.')
        return _build_empty_result(
            base,
            analysis_mode=mode,
            analysis_mode_label='Fast Analysis' if mode == 'fast' else ('Focused Analysis' if mode == 'focused' else 'High Precision / Full Frame'),
            evidence_source='opencv_only',
            message='OpenCV could not open the uploaded video.',
            warnings=warnings,
            sampled_frame_count=0,
            frame_count=0,
            analysis_summary={'sampled_frame_count': 0, 'frame_count': 0, 'sample_interval_sec': sample_interval_sec, 'max_sampled_frames': max_sampled_frames, 'focused_window_sec': focused_window_sec},
        )

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_count = total_frames if total_frames > 0 else 0
    duration_seconds = (total_frames / fps) if fps > 0 and total_frames > 0 else None
    if fps <= 0:
        warnings.append('Video FPS metadata was missing, so a 30 FPS sampling interval was assumed.')
        fps = 30.0
    if total_frames <= 0:
        warnings.append('Video frame count metadata was missing; sampling will stop when OpenCV can no longer read frames.')

    if mode == 'full' and total_frames > 0:
        sample_indexes = list(range(total_frames))
    else:
        sample_indexes = _sampling_indexes(total_frames, fps, sample_interval_sec, max_sampled_frames)
    if not sample_indexes:
        sample_indexes = [0]

    mp_pose = None
    mp_hands = None
    if use_mediapipe and mp is not None:
        try:
            mp_pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
            mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        except Exception as exc:
            warnings.append(f'MediaPipe initialisation failed, so OpenCV-only analysis continued: {exc}')
            mp_pose = None
            mp_hands = None
    elif use_mediapipe and mp is None:
        warnings.append('MediaPipe is not installed, so OpenCV-only analysis continued.')

    timeline: list[dict[str, Any]] = []
    previous_small: np.ndarray | None = None
    observed_candidates: list[int] = []
    evidence_source = 'opencv_only'
    strongest_motion = 0.0

    def analyze_frame(frame_number: int) -> dict[str, Any] | None:
        nonlocal previous_small, evidence_source, strongest_motion, mp_pose, mp_hands
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ok, frame = capture.read()
        if not ok or frame is None:
            warnings.append(f'OpenCV could not read sampled frame {frame_number}; sampling stopped early.')
            return None
        height, width = frame.shape[:2]
        sample_path = frame_dir / f'sample_{len(timeline) + 1:03d}.jpg'
        cv2.imwrite(str(sample_path), frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        motion_delta = 0.0
        motion_centroid = None
        if previous_small is not None:
            diff = cv2.absdiff(previous_small, small)
            motion_delta = float(cv2.mean(diff)[0]) / 255.0
            motion_centroid = _motion_centroid(diff, width, height)
            strongest_motion = max(strongest_motion, motion_delta)
        previous_small = small

        pose_angles: dict[str, float | None] = {'wrist': None, 'elbow': None, 'shoulder': None}
        pose_point = None
        hand_angles: dict[str, float | None] = {'wrist': None, 'elbow': None, 'shoulder': None}
        hand_point = None
        landmark_detected = False
        evidence_detail = ''
        if mp_pose is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = mp_pose.process(rgb)
                hand_results = mp_hands.process(rgb) if mp_hands is not None else None
                pose_angles, pose_point, evidence_detail = _detect_pose_angles(pose_results, width, height)
                hand_angles, hand_point, hand_detail = _detect_hand_angles(hand_results, width, height)
                if hand_detail:
                    evidence_detail = f'{evidence_detail}; {hand_detail}' if evidence_detail else hand_detail
                if pose_point or hand_point:
                    landmark_detected = True
                    evidence_source = 'opencv_mediapipe'
            except Exception as exc:
                warnings.append(f'MediaPipe failed on sampled frame {frame_number}; OpenCV-only analysis continued: {exc}')
                mp_pose = None
                mp_hands = None
                evidence_source = 'opencv_only'

        detected_signal: dict[str, Any] = {'motion_delta': round(motion_delta, 4)}
        if motion_centroid is not None:
            detected_signal['motion_centroid'] = motion_centroid
        if pose_angles['elbow'] is not None:
            detected_signal['elbow_angle'] = round(pose_angles['elbow'], 2)
        if pose_angles['shoulder'] is not None:
            detected_signal['shoulder_angle'] = round(pose_angles['shoulder'], 2)
        if hand_angles['wrist'] is not None:
            detected_signal['wrist_angle'] = round(hand_angles['wrist'], 2)
        if pose_point is not None:
            detected_signal['pose_point'] = pose_point
        if hand_point is not None:
            detected_signal['hand_point'] = hand_point

        if landmark_detected:
            if pose_point is not None and hand_point is not None:
                detected_value = 'pose and hand landmarks detected'
            elif pose_point is not None:
                detected_value = 'pose landmarks detected'
            else:
                detected_value = 'hand landmarks detected'
            status = 'Detected'
            confidence = 0.95 if (pose_point or hand_point) else 0.85
        elif motion_delta > 0.03:
            detected_value = 'motion sampled'
            status = 'not_detected'
            confidence = round(min(0.65, max(0.2, motion_delta * 6.0)), 2)
        else:
            detected_value = 'not_detected'
            status = 'not_detected'
            confidence = 0.0

        trajectory_points = [point for point in (motion_centroid, pose_point, hand_point) if point is not None]
        if not landmark_detected:
            trajectory_points = []

        return {
            'id': f'{mode}-{len(timeline) + 1:03d}',
            'timestamp': f'{(frame_number / fps):.2f}s',
            'frame_id': f'frame_{frame_number:06d}',
            'detected_value': detected_value,
            'confidence': round(confidence, 2),
            'status': status,
            'decision': 'Pending',
            'keyframe_path': str(sample_path),
            'calibration_status': mode,
            'trajectory_points': trajectory_points,
            'expert_correction': '',
            'practice_point': '',
            'failure_pattern': '',
            'detected_signal': detected_signal,
            'analysis_source': evidence_source,
            'evidence_detail': evidence_detail,
        }

    try:
        for frame_number in sample_indexes:
            item = analyze_frame(frame_number)
            if item is None:
                break
            timeline.append(item)
            if mode == 'focused' and (item['detected_value'] != 'not_detected' or item['confidence'] > 0.0):
                observed_candidates.append(frame_number)

        if mode == 'focused' and frame_count > 0 and observed_candidates:
            radius = max(1, int(round(max(focused_window_sec, 0.5) * fps / 2.0)))
            dense_indexes: list[int] = []
            for frame_number in observed_candidates:
                dense_indexes.extend(_window_indexes(frame_number, radius, frame_count))
            for frame_number in _ordered_unique(dense_indexes):
                if any(item['frame_id'] == f'frame_{frame_number:06d}' for item in timeline):
                    continue
                item = analyze_frame(frame_number)
                if item is None:
                    continue
                timeline.append(item)
        if mode == 'full' and total_frames > 0 and len(timeline) < total_frames:
            warnings.append('Full-frame analysis stopped early, so partial evidence was returned instead of crashing.')
    finally:
        capture.release()
        if mp_pose is not None:
            try:
                mp_pose.close()
            except Exception:
                pass
        if mp_hands is not None:
            try:
                mp_hands.close()
            except Exception:
                pass

    if not timeline:
        warnings.append('No supported evidence could be extracted from the uploaded video.')

    if mode == 'full':
        analysis_mode_label = 'High Precision / Full Frame'
    elif mode == 'focused':
        analysis_mode_label = 'Focused Analysis'
    else:
        analysis_mode_label = 'Fast Analysis'

    if evidence_source == 'opencv_mediapipe':
        analysis_message = 'MediaPipe landmarks were detected locally on sampled frames.'
    elif mode == 'full' and warnings:
        analysis_message = 'Full-frame analysis returned partial local evidence with warnings.'
    elif mode == 'focused' and observed_candidates:
        analysis_message = 'Focused local analysis expanded around candidate frames.'
    elif any(item['detected_value'] != 'not_detected' for item in timeline):
        analysis_message = 'OpenCV-only motion evidence was extracted locally from sampled frames.'
    else:
        analysis_message = 'OpenCV sampled the uploaded video locally, but no supported evidence was extracted.'

    analysis_summary = {
        'sampled_frame_count': len(timeline),
        'frame_count': frame_count,
        'sample_interval_sec': sample_interval_sec,
        'max_sampled_frames': max_sampled_frames,
        'focused_window_sec': focused_window_sec,
        'duration_seconds': duration_seconds,
        'analysis_mode': mode,
        'evidence_source': evidence_source,
    }

    if timeline:
        first_frame = timeline[0]
        sample_video = {
            'label': video_name,
            'source_video': video_name,
            'calibration_status': mode,
            'overlay_labels': ['sampled frames', 'motion deltas', 'trajectory points'] + (['pose landmarks', 'hand landmarks'] if evidence_source == 'opencv_mediapipe' else []),
            'angles': {
                'wrist': _format_angle(timeline, 'wrist_angle'),
                'elbow': _format_angle(timeline, 'elbow_angle'),
                'shoulder': _format_angle(timeline, 'shoulder_angle'),
            },
            'note': analysis_message,
        }
    else:
        first_frame = {}
        sample_video = {
            'label': video_name,
            'source_video': video_name,
            'calibration_status': 'unavailable',
            'overlay_labels': ['no supported evidence extracted'],
            'angles': {'wrist': 'Unavailable', 'elbow': 'Unavailable', 'shoulder': 'Unavailable'},
            'note': analysis_message,
        }

    confidence_values = [float(item.get('confidence') or 0.0) for item in timeline if isinstance(item, dict)]
    result = dict(base)
    result.update(
        {
            'analysis_mode': mode,
            'analysis_mode_label': analysis_mode_label,
            'analysis_source_label': evidence_source,
            'evidence_source': evidence_source,
            'analysis_status': 'warning' if warnings and timeline else ('error' if not timeline else 'success'),
            'analysis_message': analysis_message,
            'warnings': warnings,
            'sampled_frame_count': len(timeline),
            'frame_count': frame_count,
            'analysis_summary': analysis_summary,
            'sample_video': sample_video,
            'evidence_summary': {
                'timestamp': first_frame.get('timestamp', 'Unavailable') if first_frame else 'Unavailable',
                'frame_id': first_frame.get('frame_id', 'Unavailable') if first_frame else 'Unavailable',
                'calibration_status': mode,
                'wrist_angle': _first_available_angle(timeline, 'wrist_angle'),
                'movement_speed': _movement_speed_label(timeline),
                'pause_detected': False,
                'confidence': round(max(confidence_values) if confidence_values else 0.0, 3),
                'status': 'Detected' if any(str(item.get('status') or '').lower() == 'detected' for item in timeline) else ('not_detected' if timeline else 'Unavailable'),
                'analysis_mode': mode,
                'evidence_source': evidence_source,
            },
            'timeline': timeline,
            'review_notes': _build_review_notes(mode, evidence_source, warnings),
            'chat_prompts': _build_chat_prompts(mode, evidence_source, timeline),
            'export_preview_hint': 'Only Expert-confirmed items are treated as trusted Practice Memory.',
            'initial_chat_question': _build_chat_prompts(mode, evidence_source, timeline)[0],
            'initial_chat_response': _build_chat_response(timeline, mode, evidence_source, warnings),
        }
    )
    return result
