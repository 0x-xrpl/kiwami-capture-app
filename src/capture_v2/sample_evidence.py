from __future__ import annotations

from copy import deepcopy
from typing import Any


def build_capture_v2_sample_data() -> dict[str, Any]:
    timeline = [
        {
            "id": "evt-001",
            "timestamp": "00:12.1",
            "frame_id": "frame_0388",
            "detected_value": "angle shift detected",
            "confidence": 0.84,
            "status": "Detected",
            "decision": "Pending",
            "keyframe_path": "sample_keyframes/frame_0388.png",
            "calibration_status": "uncalibrated",
            "trajectory_points": [{"x": 120, "y": 180}, {"x": 144, "y": 162}, {"x": 168, "y": 148}],
            "expert_correction": "",
            "practice_point": "",
            "failure_pattern": "",
        },
        {
            "id": "evt-002",
            "timestamp": "00:13.2",
            "frame_id": "frame_0396",
            "detected_value": "pause detected",
            "confidence": 0.72,
            "status": "Inferred",
            "decision": "Pending",
            "keyframe_path": "sample_keyframes/frame_0396.png",
            "calibration_status": "uncalibrated",
            "trajectory_points": [{"x": 170, "y": 150}, {"x": 174, "y": 150}, {"x": 175, "y": 151}],
            "expert_correction": "",
            "practice_point": "",
            "failure_pattern": "",
        },
        {
            "id": "evt-003",
            "timestamp": "00:15.0",
            "frame_id": "frame_0410",
            "detected_value": "correction movement candidate",
            "confidence": 0.68,
            "status": "Inferred",
            "decision": "Pending",
            "keyframe_path": "sample_keyframes/frame_0410.png",
            "calibration_status": "uncalibrated",
            "trajectory_points": [{"x": 172, "y": 150}, {"x": 180, "y": 140}, {"x": 190, "y": 132}],
            "expert_correction": "",
            "practice_point": "",
            "failure_pattern": "",
        },
    ]

    data: dict[str, Any] = {
        "page_title": "Kiwami Capture v2 — Evidence Capture Room",
        "subtitle": "Local-first skill evidence before Practice Memory.",
        "status_line": "Local Only ・ Estimated until calibrated ・ Expert review required",
        "sample_video": {
            "label": "Sample evidence clip",
            "source_video": "sample_clip.mp4",
            "calibration_status": "uncalibrated",
            "overlay_labels": [
                "hand landmarks",
                "pose landmarks",
                "wrist angle",
                "elbow angle",
                "shoulder angle",
                "trajectory path",
            ],
            "angles": {
                "wrist": "42° estimated",
                "elbow": "118° estimated",
                "shoulder": "64° estimated",
            },
            "note": "Angles are estimated until calibrated.",
        },
        "evidence_summary": {
            "timestamp": "00:13.2",
            "frame_id": "frame_0396",
            "calibration_status": "uncalibrated",
            "wrist_angle": "42° estimated",
            "movement_speed": "24 px/s",
            "pause_detected": True,
            "confidence": 0.72,
            "status": "Inferred",
        },
        "timeline": timeline,
        "review_notes": [
            "Expert review is required before Practice Memory is trusted.",
            "Corrections should turn inferred evidence into a durable practice point.",
        ],
        "chat_prompts": [
            "What does the evidence show?",
            "Which frame looks like a correction movement?",
            "What still needs expert confirmation?",
        ],
        "future_hooks": [
            "OpenCV keyframe extraction",
            "MediaPipe Hands",
            "MediaPipe Pose",
            "angle calculation",
            "trajectory calculation",
            "pause detection",
            "speed shift detection",
        ],
        "export_skill_name": "sample skill",
        "export_source_video": "sample_clip.mp4",
        "export_preview_hint": "Only Expert-confirmed items are treated as trusted Practice Memory.",
    }
    data["initial_chat_question"] = data["chat_prompts"][0]
    data["initial_chat_response"] = mock_local_llm_response(data, data["initial_chat_question"])
    return data


def mock_local_llm_response(evidence: dict[str, Any], question: str) -> str:
    timeline = list(evidence.get("timeline") or [])
    review_notes = list(evidence.get("review_notes") or [])
    question = (question or "").strip()

    if not timeline:
        return "This cannot be confirmed from the available evidence. Expert review is required."

    target = next((item for item in timeline if "pause" in str(item.get("detected_value", "")).lower()), None)
    if target is None:
        target = next((item for item in timeline if item.get("status") != "Rejected"), timeline[0])

    timestamp = str(target.get("timestamp") or "")
    frame_id = str(target.get("frame_id") or "")
    status = str(target.get("status") or "Inferred")

    candidate_values = [str(item.get("detected_value") or "") for item in timeline]
    mentions = []
    if any("pause" in value.lower() for value in candidate_values):
        mentions.append("a pause")
    if any("angle shift" in value.lower() for value in candidate_values):
        mentions.append("a wrist angle change")
    if not mentions:
        mentions.append(str(target.get("detected_value") or "evidence"))

    if status == "Expert-confirmed":
        response = f"At {timestamp}, {' and '.join(mentions)} were confirmed from {frame_id}."
        if review_notes:
            response += f" {review_notes[1] if len(review_notes) > 1 else review_notes[0]}"
        return response

    response = (
        f"At {timestamp}, {' and '.join(mentions)} were detected from {frame_id}. "
        "This is still unconfirmed. Expert review is required before it becomes Practice Memory."
    )
    return response


def build_export_items(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    export_items: list[dict[str, Any]] = []
    for item in evidence.get("timeline") or []:
        if not isinstance(item, dict):
            continue
        if item.get("status") != "Expert-confirmed" or item.get("decision") == "Rejected":
            continue
        export_items.append(
            {
                "skill_name": evidence.get("export_skill_name", "sample skill"),
                "source_video": evidence.get("export_source_video", "sample_clip.mp4"),
                "evidence_time": item.get("timestamp", ""),
                "frame_id": item.get("frame_id", ""),
                "keyframe_path": item.get("keyframe_path", ""),
                "detected_signal": {
                    "wrist_angle": item.get("wrist_angle", "42° estimated"),
                    "movement_speed": item.get("movement_speed", "24 px/s"),
                    "pause_detected": item.get("pause_detected", True),
                },
                "trajectory_points": deepcopy(item.get("trajectory_points") or []),
                "calibration_status": item.get("calibration_status", "uncalibrated"),
                "ai_suggestion": item.get("detected_value", ""),
                "expert_correction": item.get("expert_correction", ""),
                "practice_point": item.get("practice_point", ""),
                "failure_pattern": item.get("failure_pattern", ""),
                "status": "expert-confirmed",
                "confidence": item.get("confidence", 0.0),
            }
        )
    return export_items

