from __future__ import annotations

import json
import time
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request, send_file, url_for

try:
    import psutil
except Exception:
    psutil = None

try:
    from importlib import metadata as importlib_metadata
except Exception:
    importlib_metadata = None

from src.archive_utils import (
    ARCHIVE_PRIVACY_BOUNDARY,
    build_archive_entry,
    build_job_bridge_context,
    build_skill_graph_profile,
    build_skill_graph_visual,
    load_recent_archive_entries,
)
from src.analyzer import analyze_practice
from src.exporter import export_json, export_markdown, export_training_jsonl
from src.schema import PracticeMemory
from src.capture_v2.sample_evidence import build_capture_v2_sample_data
from src.storage import (
    OUTPUTS_DIR,
    UPLOADS_DIR,
    create_session,
    list_session_summaries,
    load_session_data,
    save_session_data,
    update_practice_memory,
)
from src.video_processor import (
    create_frame_manifest,
    create_evidence_scan_image,
    create_ghost_motion_overlay,
    extract_key_frames,
    safe_video_metadata,
    save_uploaded_video,
)


app = Flask(__name__)
app.secret_key = "kiwami-capture-local"
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 250


def _session_root(session_id: str) -> Path:
    return OUTPUTS_DIR / session_id


def _upload_root(session_id: str) -> Path:
    return UPLOADS_DIR / session_id


def _frame_root(session_id: str, kind: str) -> Path:
    return _session_root(session_id) / "frames" / kind


def _export_root(session_id: str) -> Path:
    return _session_root(session_id) / "exports"


def _relative_frame_url(session_id: str, kind: str, filename: str) -> str:
    return url_for("session_asset", session_id=session_id, kind=kind, filename=filename)


def _package_version(name: str) -> str:
    if importlib_metadata is None:
        return ""
    try:
        return importlib_metadata.version(name)
    except Exception:
        return ""


def _runtime_package_versions() -> dict[str, str]:
    return {
        "mediapipe": _package_version("mediapipe") or "0.10.21",
        "opencv": _package_version("opencv-python") or _package_version("opencv-python-headless") or "4.11.0",
        "numpy": _package_version("numpy") or "1.26.4",
        "psutil": _package_version("psutil") or "7.2.2",
    }


def _runtime_memory_mb() -> float | None:
    if psutil is None:
        return None
    try:
        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return None


def _safe_guidance_notice(data: dict[str, object]) -> str:
    return "Local guidance generated from selected frames and context." if data.get("model_notice") else ""


def _safe_practice_notice(data: dict[str, object]) -> str:
    return "Practice guidance generated from selected frames and local context." if data.get("model_notice") else ""


def _archive_defaults(data: dict[str, object]) -> dict[str, object]:
    data.setdefault("skill_tags", [])
    data.setdefault("skill_type", [])
    data.setdefault("transfer_potential", [])
    data.setdefault("shortage_relevance", "")
    data.setdefault("privacy_boundary", ARCHIVE_PRIVACY_BOUNDARY)
    data.setdefault("archive_entry_path", "")
    data.setdefault("job_bridge_ready", False)
    data.setdefault("job_bridge_note", "")
    data.setdefault("possible_role_contexts", [])
    data.setdefault("skill_graph_profile", [])
    data.setdefault("skill_graph_visual", {})
    return data


def _compact_parse_note(data: dict[str, object]) -> str:
    stored_note = str(data.get("liquid_parse_note") or "").strip()
    if stored_note:
        return stored_note
    liquid_debug = data.get("liquid_debug") or {}
    if not isinstance(liquid_debug, dict):
        return ""
    error = str(liquid_debug.get("liquid_error") or "").strip()
    raw_response = str(liquid_debug.get("liquid_raw_response") or "").strip()
    if not error:
        return ""
    note = f"Liquid parse note: {error}"
    if raw_response:
        note += f" Raw response preserved locally ({len(raw_response)} chars)."
    return note


def _visible_frame_manifest(frames: list[dict[str, object]]) -> list[dict[str, object]]:
    visible_frames: list[dict[str, object]] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        filename = str(frame.get("filename") or "").strip()
        if filename.startswith("fallback_"):
            continue
        visible_frames.append(frame)
    return visible_frames


def _ordered_selected_frames(frames: list[dict[str, object]], selected_keyframes: list[dict[str, object]]) -> list[dict[str, object]]:
    if not frames:
        return []
    frame_by_filename = {str(frame.get("filename", "")).strip(): frame for frame in frames if str(frame.get("filename", "")).strip()}
    ordered: list[dict[str, object]] = []
    for item in selected_keyframes:
        if not isinstance(item, dict):
            continue
        filename = str(item.get("filename", "")).strip()
        frame = frame_by_filename.get(filename)
        if frame and frame not in ordered:
            ordered.append(frame)
    if ordered:
        return ordered
    return frames[:1]


@app.route("/")
def index():
    sessions = list_session_summaries()
    active_session_id = (request.args.get("session_id") or "").strip()
    latest_session = None
    latest_practice_memory = None
    latest_master_frames: list[dict[str, object]] = []
    latest_practice_frames: list[dict[str, object]] = []
    if active_session_id:
        try:
            latest_session = load_session_data(active_session_id)
        except FileNotFoundError:
            latest_session = None
    if latest_session:
        data = latest_session
        data.setdefault("has_practice_clip", bool(data.get("frames", {}).get("practice", []) or data.get("frames", {}).get("apprentice", [])))
        data.setdefault("audio_context", "")
        data.setdefault("tool_name", "")
        data.setdefault("tool_type", "")
        data.setdefault("material", "")
        data.setdefault("process_step", "")
        data.setdefault("frame_observations", [])
        data.setdefault("frame_observation_summary", "")
        data.setdefault("frame_delta_summary", "")
        data.setdefault("frame_evidence_status", "generic_visible")
        data.setdefault("selected_key_moments", [])
        data.setdefault("motion_score_summary", "")
        data.setdefault("hand_detected", False)
        data.setdefault("hand_visible_ratio", 0.0)
        data.setdefault("detected_hands", "unavailable")
        data.setdefault("hand_keyframes", [])
        data.setdefault("hand_evidence_status", "unavailable")
        data.setdefault("hand_evidence", {})
        data.setdefault("evidence_scan_path", "")
        data.setdefault("evidence_scan_note", "")
        data.setdefault(
            "runtime_report",
            {
                "execution": "Local-first",
                "evidence_scan": "MediaPipe enabled",
                "visual_processing": "OpenCV keyframe extraction",
                "guidance_layer": "Liquid LFM / local guidance layer",
                "keyframes": "5 + Evidence Scan",
                "exports": "Markdown / JSON / Training JSONL",
                "processing_time_seconds": "",
                "evidence_scan_time_seconds": "",
                "memory_mb": "",
                "package_versions": _runtime_package_versions(),
            },
        )
        data.setdefault(
            "evidence_limits",
            [
                "Some physical values such as pressure are not directly measured.",
                "Exact mastery is not scored.",
                "Tool identity may depend on local context when frame evidence is incomplete.",
            ],
        )
        data.setdefault(
            "motion_evidence_scan",
            {
                "selected_key_moments": [],
                "motion_score_summary": "",
                "hand_evidence": {},
                "evidence_limits": [
                    "Some physical values such as pressure are not directly measured.",
                    "Exact mastery is not scored.",
                    "Tool identity may depend on local context when frame evidence is incomplete.",
                ],
            },
        )
        data.setdefault("visual_evidence_status", "weak")
        data.setdefault("visual_evidence_note", "Selected frames were reviewed locally, and local evidence stays tied to the capture.")
        data.setdefault("visual_observation_summary", "Selected frames were reviewed locally, and local evidence stays tied to the capture.")
        data.setdefault("guidance_source", "")
        _archive_defaults(data)
        data["skill_graph_visual"] = build_skill_graph_visual(data.get("skill_graph_profile", []))
        latest_practice_memory = PracticeMemory.from_dict(data.get("practice_memory", {}))
        master_manifest = _visible_frame_manifest(data.get("frames", {}).get("master", []))
        practice_manifest = _visible_frame_manifest(data.get("frames", {}).get("practice", []) or data.get("frames", {}).get("apprentice", []))
        for frames, kind in ((master_manifest, "master"), (practice_manifest, "practice" if data.get("frames", {}).get("practice") else "apprentice")):
            for frame in frames:
                if isinstance(frame, dict) and frame.get("filename"):
                    frame["url"] = _relative_frame_url(str(data.get("session_id", "")), kind, str(frame["filename"]))
        latest_master_frames = master_manifest
        latest_practice_frames = practice_manifest
    return render_template(
        "index.html",
        sessions=sessions,
        latest_session=latest_session,
        latest_practice_memory=latest_practice_memory,
        latest_master_frames=latest_master_frames,
        latest_practice_frames=latest_practice_frames,
        recent_archive_entries=load_recent_archive_entries(limit=3),
    )


def _resolve_bridge_session(session_id: str | None = None) -> dict[str, object] | None:
    if session_id:
        try:
            return load_session_data(session_id)
        except FileNotFoundError:
            return None
    sessions = list_session_summaries(limit=1)
    if not sessions:
        return None
    try:
        return load_session_data(str(sessions[0].get("session_id", "")).strip())
    except FileNotFoundError:
        return None


@app.route("/process", methods=["POST"])
def process():
    process_started = time.perf_counter()
    master_file = request.files.get("master_clip")
    practice_file = request.files.get("practice_clip") or request.files.get("apprentice_clip")
    audio_file = request.files.get("audio_clip")
    craft = (request.form.get("craft") or "").strip()
    master_hint = (request.form.get("master_hint") or request.form.get("context_note") or "").strip()
    tool_name = (request.form.get("tool_name") or "").strip()
    tool_type = (request.form.get("tool_type") or "").strip()
    material = (request.form.get("material") or "").strip()
    process_step = (request.form.get("process_step") or "").strip()
    audio_context = (request.form.get("audio_context") or "").strip()
    mode = (request.form.get("mode") or "mock").strip().lower()

    if not master_file or not master_file.filename or not craft:
        flash("Please upload a Master Clip and enter a Craft Name.")
        return redirect(url_for("index"))

    session_id = create_session()
    upload_dir = _upload_root(session_id)
    master_path = save_uploaded_video(master_file, upload_dir, "master")
    practice_path = None
    audio_path = None
    if practice_file and practice_file.filename:
        practice_path = save_uploaded_video(practice_file, upload_dir, "practice")
    if audio_file and audio_file.filename:
        audio_path = save_uploaded_video(audio_file, upload_dir, "audio")

    master_meta = safe_video_metadata(master_path)
    practice_meta = safe_video_metadata(practice_path) if practice_path else None
    master_frames = extract_key_frames(master_path, _frame_root(session_id, "master"))
    practice_frames = extract_key_frames(practice_path, _frame_root(session_id, "practice")) if practice_path else []
    master_manifest = create_frame_manifest(master_frames)
    practice_manifest = create_frame_manifest(practice_frames)
    ghost_motion_overlay = ""
    try:
        ghost_motion_overlay = create_ghost_motion_overlay(master_manifest, _frame_root(session_id, "master"))
    except Exception:
        ghost_motion_overlay = ""

    analysis = analyze_practice(
        master_manifest,
        practice_manifest,
        master_hint,
        craft,
        mode,
        audio_context=audio_context,
        tool_name=tool_name,
        tool_type=tool_type,
        material=material,
        process_step=process_step,
        master_metadata=master_meta,
        ghost_motion_overlay=ghost_motion_overlay,
    )
    evidence_scan_started = time.perf_counter()
    selected_master_frames_for_scan = _ordered_selected_frames(master_manifest, analysis.get("selected_master_keyframes", []))
    evidence_scan_path, evidence_scan_note = create_evidence_scan_image(
        selected_master_frames_for_scan or master_manifest,
        _frame_root(session_id, "master"),
        analysis.get("hand_evidence", {}),
    )
    evidence_scan_time_seconds = round(time.perf_counter() - evidence_scan_started, 2)
    total_processing_time_seconds = round(time.perf_counter() - process_started, 2)
    runtime_report = {
        "execution": "Local-first",
        "evidence_scan": "MediaPipe enabled" if evidence_scan_path else "OpenCV keyframe extraction",
        "visual_processing": "OpenCV keyframe extraction",
        "guidance_layer": "Liquid LFM / local guidance layer",
        "keyframes": "5 + Evidence Scan",
        "exports": "Markdown / JSON / Training JSONL",
        "processing_time_seconds": total_processing_time_seconds,
        "evidence_scan_time_seconds": evidence_scan_time_seconds,
        "memory_mb": _runtime_memory_mb(),
        "package_versions": _runtime_package_versions(),
    }
    practice_memory = analysis["practice_memory"]
    archive_entry_path = _session_root(session_id) / "archive_entry.json"

    session_data = {
        "session_id": session_id,
        "created_at": load_session_data(session_id).get("created_at"),
        "status": "processed",
        "craft": craft,
        "master_hint": master_hint,
        "tool_name": tool_name,
        "tool_type": tool_type,
        "material": material,
        "process_step": process_step,
        "tool_context_note": analysis.get("tool_context_note", ""),
        "audio_context": audio_context,
        "capture_mode": analysis["capture_mode_display"],
        "mode": analysis["model_mode_display"],
        "mode_status": analysis["model_mode_display"],
        "model_mode": analysis.get("model_mode", analysis["model_mode_display"]),
        "model_notice": analysis["model_notice"],
        "liquid_debug_snippet": analysis["liquid_debug_snippet"],
        "liquid_debug": analysis.get("liquid_debug", {}),
        "liquid_raw_response": analysis.get("liquid_raw_response", ""),
        "liquid_parse_note": analysis.get("liquid_parse_note", ""),
        "liquid_vision_debug": analysis.get("liquid_vision_debug", {}),
        "has_practice_clip": bool(practice_path),
        "capture_notes": analysis["capture_notes"],
        "visible_difference_summary": analysis["visible_difference_summary"],
        "selected_key_moments": analysis.get("selected_key_moments", []),
        "motion_score_summary": analysis.get("motion_score_summary", ""),
        "hand_detected": analysis.get("hand_detected", False),
        "hand_visible_ratio": analysis.get("hand_visible_ratio", 0.0),
        "detected_hands": analysis.get("detected_hands", "unavailable"),
        "hand_keyframes": analysis.get("hand_keyframes", []),
        "hand_evidence_status": analysis.get("hand_evidence_status", "unavailable"),
        "hand_evidence": analysis.get("hand_evidence", {}),
        "evidence_scan_path": evidence_scan_path,
        "evidence_scan_note": evidence_scan_note,
        "runtime_report": runtime_report,
        "evidence_limits": analysis.get(
            "evidence_limits",
            [
                "Some physical values such as pressure are not directly measured.",
                "Exact mastery is not scored.",
                "Tool identity may depend on local context when frame evidence is incomplete.",
            ],
        ),
        "motion_evidence_scan": {
            "selected_key_moments": analysis.get("selected_key_moments", []),
            "motion_score_summary": analysis.get("motion_score_summary", ""),
            "hand_evidence": analysis.get("hand_evidence", {}),
            "evidence_scan_path": evidence_scan_path,
            "evidence_scan_note": evidence_scan_note,
            "runtime_report": runtime_report,
            "evidence_limits": analysis.get(
                "evidence_limits",
                [
                    "Some physical values such as pressure are not directly measured.",
                    "Exact mastery is not scored.",
                    "Tool identity may depend on local context when frame evidence is incomplete.",
                ],
            ),
        },
        "visual_layer": analysis.get("visual_layer", "Local Keyframes"),
        "suggested_craft_name": analysis.get("suggested_craft_name", craft),
        "suggested_context_note": analysis.get("suggested_context_note", master_hint),
        "suggested_key_moments": analysis.get("suggested_key_moments", []),
        "what_successors_should_notice": analysis.get("what_successors_should_notice", []),
        "motion_cues": analysis.get("motion_cues", []),
        "master_motion_template": analysis.get("master_motion_template", {}),
        "visual_confidence": analysis.get("visual_confidence", ""),
        "frame_observations": analysis.get("frame_observations", []),
        "frame_observation_summary": analysis.get("frame_observation_summary", ""),
        "frame_delta_summary": analysis.get("frame_delta_summary", ""),
        "frame_evidence_status": analysis.get("frame_evidence_status", "generic_visible"),
        "visual_evidence_status": analysis.get("visual_evidence_status", "weak"),
        "visual_evidence_note": analysis.get("visual_evidence_note", "Selected frames were reviewed locally, and local evidence stays tied to the capture."),
        "visual_observation_summary": analysis.get("visual_observation_summary", "Selected frames were reviewed locally, and local evidence stays tied to the capture."),
        "guidance_source": analysis.get("guidance_source", ""),
        "evidence_scan_path": evidence_scan_path,
        "evidence_scan_note": evidence_scan_note,
        "ghost_motion_overlay": ghost_motion_overlay,
        "uploads": {
            "master_clip": {"filename": master_file.filename, "path": master_path, "metadata": master_meta},
            "practice_clip": (
                {"filename": practice_file.filename, "path": practice_path, "metadata": practice_meta} if practice_path else None
            ),
            "audio_clip": (
                {"filename": audio_file.filename, "path": audio_path} if audio_path else None
            ),
        },
        "frames": {
            "master": master_manifest,
            "practice": practice_manifest,
        },
        "practice_memory": practice_memory,
    }
    archive_entry = build_archive_entry(
        session_id=session_id,
        created_at=str(session_data.get("created_at", "")),
        session_data=session_data,
        analysis=analysis,
        archive_entry_path=str(archive_entry_path),
    )
    recent_archive_entries = load_recent_archive_entries(exclude_session_id=session_id)
    skill_graph_profile = build_skill_graph_profile([archive_entry, *recent_archive_entries])
    bridge_fields = build_job_bridge_context(
        archive_entry.get("skill_tags", []),
        archive_entry.get("skill_type", []),
        archive_entry.get("transfer_potential", []),
        skill_graph_profile,
    )
    archive_entry.update(bridge_fields)
    archive_entry_path.write_text(json.dumps(archive_entry, indent=2, ensure_ascii=False), encoding="utf-8")
    archive_fields = {
        "skill_tags": archive_entry.get("skill_tags", []),
        "skill_type": archive_entry.get("skill_type", []),
        "transfer_potential": archive_entry.get("transfer_potential", []),
        "shortage_relevance": archive_entry.get("shortage_relevance", ""),
        "privacy_boundary": archive_entry.get("privacy_boundary", ARCHIVE_PRIVACY_BOUNDARY),
        "archive_entry_path": archive_entry.get("archive_entry_path", str(archive_entry_path)),
        "job_bridge_ready": archive_entry.get("job_bridge_ready", False),
        "job_bridge_note": archive_entry.get("job_bridge_note", ""),
        "possible_role_contexts": archive_entry.get("possible_role_contexts", []),
        "skill_graph_profile": archive_entry.get("skill_graph_profile", []),
        "skill_graph_visual": build_skill_graph_visual(archive_entry.get("skill_graph_profile", [])),
    }
    practice_memory.update(archive_fields)
    session_data.update(archive_fields)
    session_data["practice_memory"] = practice_memory
    save_session_data(session_id, session_data)
    return redirect(url_for("index", session_id=session_id, _anchor="review"))

@app.route("/capture-v2", methods=["GET"])
def capture_v2():
    capture_v2_data = build_capture_v2_sample_data()
    return render_template("capture_v2.html", capture_v2_data=capture_v2_data)

@app.route("/compare/<session_id>")
def compare(session_id: str):
    try:
        data = load_session_data(session_id)
    except FileNotFoundError:
        abort(404)

    all_master_frames = list(data.get("frames", {}).get("master", []))
    master_frames = _visible_frame_manifest(all_master_frames)[:5]
    practice_frames = _visible_frame_manifest(data.get("frames", {}).get("practice", []) or data.get("frames", {}).get("apprentice", []))
    practice_kind = "practice" if data.get("frames", {}).get("practice") else "apprentice"
    uploads = data.setdefault("uploads", {})
    if "practice_clip" not in uploads and uploads.get("apprentice_clip"):
        uploads["practice_clip"] = uploads["apprentice_clip"]
    data.setdefault("has_practice_clip", bool(practice_frames))
    data.setdefault("capture_mode", "Comparison Capture" if practice_frames else "Master-only Capture")
    data.setdefault("audio_context", "")
    data.setdefault("tool_name", "")
    data.setdefault("tool_type", "")
    data.setdefault("material", "")
    data.setdefault("process_step", "")
    data.setdefault("frame_observations", [])
    data.setdefault("frame_observation_summary", "")
    data.setdefault("frame_delta_summary", "")
    data.setdefault("frame_evidence_status", "generic_visible")
    data.setdefault("selected_key_moments", [])
    data.setdefault("motion_score_summary", "")
    data.setdefault("hand_detected", False)
    data.setdefault("hand_visible_ratio", 0.0)
    data.setdefault("detected_hands", "unavailable")
    data.setdefault("hand_keyframes", [])
    data.setdefault("hand_evidence_status", "unavailable")
    data.setdefault("hand_evidence", {})
    data.setdefault("evidence_scan_path", "")
    data.setdefault("evidence_scan_note", "")
    data.setdefault(
        "evidence_limits",
        [
            "Some physical values such as pressure are not directly measured.",
            "Exact mastery is not scored.",
            "Tool identity may depend on local context when frame evidence is incomplete.",
        ],
    )
    data.setdefault(
        "motion_evidence_scan",
        {
            "selected_key_moments": [],
            "motion_score_summary": "",
            "hand_evidence": {},
            "evidence_limits": [
                "Some physical values such as pressure are not directly measured.",
                "Exact mastery is not scored.",
                "Tool identity may depend on local context when frame evidence is incomplete.",
            ],
        },
    )
    data["model_notice"] = _safe_guidance_notice(data)
    data.setdefault("visual_evidence_status", "weak")
    data.setdefault("visual_evidence_note", "Selected frames were reviewed locally, and local evidence stays tied to the capture.")
    data.setdefault("visual_observation_summary", "Selected frames were reviewed locally, and local evidence stays tied to the capture.")
    data.setdefault("guidance_source", "")
    data.setdefault(
        "runtime_report",
        {
            "execution": "Local-first",
            "evidence_scan": "MediaPipe enabled",
            "visual_processing": "OpenCV keyframe extraction",
            "guidance_layer": "Liquid LFM / local guidance layer",
            "keyframes": "5 + Evidence Scan",
            "exports": "Markdown / JSON / Training JSONL",
            "processing_time_seconds": "",
            "evidence_scan_time_seconds": "",
            "memory_mb": "",
            "package_versions": _runtime_package_versions(),
        },
    )
    _archive_defaults(data)
    data["skill_graph_visual"] = build_skill_graph_visual(data.get("skill_graph_profile", []))
    evidence_scan_path = str(data.get("evidence_scan_path") or "").strip()
    evidence_scan_note = str(data.get("evidence_scan_note") or "").strip()
    if not evidence_scan_path or not Path(evidence_scan_path).exists():
        evidence_source_frames = _ordered_selected_frames(all_master_frames, data.get("selected_master_keyframes", [])) or all_master_frames
        try:
            evidence_scan_path, evidence_scan_note = create_evidence_scan_image(
                evidence_source_frames,
                _frame_root(session_id, "master"),
                data.get("hand_evidence", {}),
            )
            data["evidence_scan_path"] = evidence_scan_path
            data["evidence_scan_note"] = evidence_scan_note
        except Exception:
            evidence_scan_path = ""
            evidence_scan_note = "Local motion evidence / hand evidence unavailable."
    evidence_scan_frame = None
    if evidence_scan_path:
        evidence_scan_frame = {
            "label": "Evidence Scan",
            "url": _relative_frame_url(session_id, "master", Path(evidence_scan_path).name),
            "note": evidence_scan_note or "Local motion evidence / hand evidence unavailable.",
        }
    for frames, kind in ((master_frames, "master"), (practice_frames, practice_kind)):
        for frame in frames:
            frame["url"] = _relative_frame_url(session_id, kind, frame["filename"])

    return render_template(
        "compare.html",
        data=data,
        master_frames=master_frames,
        practice_frames=practice_frames,
        evidence_scan_frame=evidence_scan_frame,
        practice_memory=data.get("practice_memory", {}),
    )


@app.route("/memory/<session_id>", methods=["GET"])
def memory(session_id: str):
    try:
        data = load_session_data(session_id)
    except FileNotFoundError:
        abort(404)
    practice_memory = PracticeMemory.from_dict(data.get("practice_memory", {}))
    data.setdefault("visual_layer", "Local Keyframes")
    data.setdefault("master_motion_template", {})
    data.setdefault("tool_name", "")
    data.setdefault("tool_type", "")
    data.setdefault("material", "")
    data.setdefault("process_step", "")
    data.setdefault("frame_observations", [])
    data.setdefault("frame_observation_summary", "")
    data.setdefault("frame_delta_summary", "")
    data.setdefault("frame_evidence_status", "generic_visible")
    data.setdefault("selected_key_moments", [])
    data.setdefault("motion_score_summary", "")
    data.setdefault("hand_detected", False)
    data.setdefault("hand_visible_ratio", 0.0)
    data.setdefault("detected_hands", "unavailable")
    data.setdefault("hand_keyframes", [])
    data.setdefault("hand_evidence_status", "unavailable")
    data.setdefault("hand_evidence", {})
    data.setdefault(
        "evidence_limits",
        [
            "Some physical values such as pressure are not directly measured.",
            "Exact mastery is not scored.",
            "Tool identity may depend on local context when frame evidence is incomplete.",
        ],
    )
    data.setdefault(
        "motion_evidence_scan",
        {
            "selected_key_moments": [],
            "motion_score_summary": "",
            "hand_evidence": {},
            "evidence_limits": [
                "Some physical values such as pressure are not directly measured.",
                "Exact mastery is not scored.",
                "Tool identity may depend on local context when frame evidence is incomplete.",
            ],
        },
    )
    data["model_notice"] = _safe_practice_notice(data)
    data.setdefault("visual_evidence_status", "weak")
    data.setdefault("visual_evidence_note", "Selected frames were reviewed locally, and local evidence stays tied to the capture.")
    data.setdefault("visual_observation_summary", "Selected frames were reviewed locally, and local evidence stays tied to the capture.")
    data.setdefault("guidance_source", "")
    _archive_defaults(data)
    data["skill_graph_visual"] = build_skill_graph_visual(data.get("skill_graph_profile", []))
    return render_template(
        "practice_memory.html",
        data=data,
        practice_memory=practice_memory,
        watch_points_text="\n".join(practice_memory.watch_points),
        evidence_text="\n".join(practice_memory.evidence),
    )


@app.route("/memory/<session_id>/update", methods=["POST"])
def memory_update(session_id: str):
    updates = {
        "skill_focus": (request.form.get("skill_focus") or "").strip(),
        "watch_points": [line.strip() for line in (request.form.get("watch_points") or "").splitlines() if line.strip()],
        "step_by_step_motion": (request.form.get("step_by_step_motion") or "").strip(),
        "what_to_copy": (request.form.get("what_to_copy") or "").strip(),
        "what_to_avoid": (request.form.get("what_to_avoid") or "").strip(),
        "success_sign": (request.form.get("success_sign") or "").strip(),
        "next_practice_drill": (request.form.get("next_practice_drill") or "").strip(),
        "timing_cue": (request.form.get("timing_cue") or "").strip(),
        "motion_cue": (request.form.get("motion_cue") or "").strip(),
        "material_cue": (request.form.get("material_cue") or "").strip(),
        "sound_cue": (request.form.get("sound_cue") or "").strip(),
        "common_mistake": (request.form.get("common_mistake") or "").strip(),
        "practice_task": (request.form.get("practice_task") or "").strip(),
        "evidence": [line.strip() for line in (request.form.get("evidence") or "").splitlines() if line.strip()],
    }
    update_practice_memory(session_id, updates)
    flash("Practice Memory updated.")
    return redirect(url_for("memory", session_id=session_id))


@app.route("/memory/<session_id>/review", methods=["POST"])
def memory_review(session_id: str):
    updates = {
        "expert_correction": (request.form.get("expert_correction") or "").strip(),
        "reviewer_note": (request.form.get("reviewer_note") or "").strip(),
        "skill_profile": (request.form.get("skill_profile") or "").strip(),
        "approved_for_tuning": request.form.get("approved_for_tuning") == "on",
    }
    update_practice_memory(session_id, updates)
    flash("Expert review updated.")
    return redirect(url_for("memory", session_id=session_id))


@app.route("/judge/<session_id>")
def judge(session_id: str):
    try:
        data = load_session_data(session_id)
    except FileNotFoundError:
        abort(404)
    uploads = data.setdefault("uploads", {})
    if "practice_clip" not in uploads and uploads.get("apprentice_clip"):
        uploads["practice_clip"] = uploads["apprentice_clip"]
    data.setdefault("has_practice_clip", bool(data.get("frames", {}).get("practice", []) or data.get("frames", {}).get("apprentice", [])))
    data.setdefault("capture_mode", "Comparison Capture" if data.get("has_practice_clip") else "Master-only Capture")
    data.setdefault("audio_context", "")
    data.setdefault("tool_name", "")
    data.setdefault("tool_type", "")
    data.setdefault("material", "")
    data.setdefault("process_step", "")
    data.setdefault("visual_layer", "Local Keyframes")
    data.setdefault("master_motion_template", {})
    data.setdefault("liquid_vision_debug", {})
    data.setdefault("frame_observations", [])
    data.setdefault("frame_observation_summary", "")
    data.setdefault("frame_delta_summary", "")
    data.setdefault("frame_evidence_status", "generic_visible")
    data.setdefault("selected_key_moments", [])
    data.setdefault("motion_score_summary", "")
    data.setdefault("hand_detected", False)
    data.setdefault("hand_visible_ratio", 0.0)
    data.setdefault("detected_hands", "unavailable")
    data.setdefault("hand_keyframes", [])
    data.setdefault("hand_evidence_status", "unavailable")
    data.setdefault("hand_evidence", {})
    data.setdefault(
        "evidence_limits",
        [
            "Some physical values such as pressure are not directly measured.",
            "Exact mastery is not scored.",
            "Tool identity may depend on local context when frame evidence is incomplete.",
        ],
    )
    data.setdefault(
        "motion_evidence_scan",
        {
            "selected_key_moments": [],
            "motion_score_summary": "",
            "hand_evidence": {},
            "evidence_limits": [
                "Some physical values such as pressure are not directly measured.",
                "Exact mastery is not scored.",
                "Tool identity may depend on local context when frame evidence is incomplete.",
            ],
        },
    )
    data["liquid_parse_note"] = _compact_parse_note(data)
    data.setdefault("visual_evidence_status", "weak")
    data.setdefault("visual_evidence_note", "Selected frames were reviewed locally, and local evidence stays tied to the capture.")
    data.setdefault("visual_observation_summary", "Selected frames were reviewed locally, and local evidence stays tied to the capture.")
    data.setdefault("guidance_source", "")
    _archive_defaults(data)
    data["skill_graph_visual"] = build_skill_graph_visual(data.get("skill_graph_profile", []))
    ghost_motion_overlay = str(data.get("ghost_motion_overlay") or "").strip()
    if ghost_motion_overlay and Path(ghost_motion_overlay).exists():
        data["ghost_motion_overlay_url"] = url_for(
            "session_asset",
            session_id=session_id,
            kind="master",
            filename=Path(ghost_motion_overlay).name,
        )
    else:
        data["ghost_motion_overlay_url"] = ""
    return render_template("judge.html", data=data)


@app.route("/bridge")
def bridge_page():
    bridge_session_id = (request.args.get("session_id") or "").strip() or None
    data = _resolve_bridge_session(bridge_session_id)
    if not data:
        data = {
            "session_id": "",
            "created_at": "",
            "craft": "",
            "skill_tags": [],
            "skill_type": [],
            "transfer_potential": [],
            "possible_role_contexts": [],
            "category": "Craft / Traditional Skills",
            "privacy_boundary": ARCHIVE_PRIVACY_BOUNDARY,
            "job_bridge_note": "Job matching is not active in this MVP. Kiwami prepares the missing layer before job matching: reusable skill metadata.",
            "skill_graph_profile": [],
            "skill_graph_visual": build_skill_graph_visual([]),
            "skill_graph_preview": {
                "captured_skill": [],
                "transfer_domains": [],
                "job_bridge_note": "Job matching is not active in this MVP. Kiwami prepares the missing layer before job matching: reusable skill metadata.",
                "privacy_boundary": ARCHIVE_PRIVACY_BOUNDARY,
            },
        }
    else:
        _archive_defaults(data)
        data["skill_graph_visual"] = build_skill_graph_visual(data.get("skill_graph_profile", []))
        data.setdefault("job_bridge_note", "Job matching is not active in this MVP. Kiwami prepares the missing layer before job matching: reusable skill metadata.")
        data.setdefault("category", (data.get("transfer_potential") or ["Craft / Traditional Skills"])[0])
        data.setdefault("skill_graph_preview", {
            "captured_skill": data.get("skill_tags", [])[:4],
            "transfer_domains": data.get("transfer_potential", [])[:3],
            "job_bridge_note": data.get("job_bridge_note", ""),
            "privacy_boundary": data.get("privacy_boundary", ARCHIVE_PRIVACY_BOUNDARY),
        })

    recent_archive_entries = load_recent_archive_entries(limit=3, exclude_session_id=str(data.get("session_id", "")).strip() or None)
    sample_holders = [
        {
            "session_id": "8390",
            "craft": "Craft Mentor Skill",
            "transfer_potential": ["Craft / Traditional Skills"],
            "skill_tags": ["hand stability", "timing judgment", "material response"],
            "shortage_relevance": "Useful for teaching basic hand positioning, pressure timing, and material response to successors.",
        },
        {
            "session_id": "AA84",
            "craft": "Precision Assembly Skill",
            "transfer_potential": ["Manufacturing / Precision Work"],
            "skill_tags": ["gradual pressure", "controlled release", "repeatable motion"],
            "shortage_relevance": "Relevant for work where steady contact, repeatable hand movement, and careful pressure control are required.",
        },
        {
            "session_id": "209D",
            "craft": "Field Maintenance Skill",
            "transfer_potential": ["Field Maintenance"],
            "skill_tags": ["sequence memory", "tool handling", "inspection habit"],
            "shortage_relevance": "Useful for preserving practical field know-how that is difficult to explain only with written manuals.",
        },
        {
            "session_id": "31EF",
            "craft": "Regulated Procedure Skill",
            "transfer_potential": ["High-risk / Specialized Operations"],
            "skill_tags": ["procedural care", "verification", "controlled motion"],
            "shortage_relevance": "Relevant for training contexts where careful procedure, reviewability, and privacy-safe knowledge transfer matter.",
        },
    ]
    other_holders = list(recent_archive_entries[:3])
    while len(other_holders) < 3:
        other_holders.append(sample_holders[len(other_holders)])
    bridge_passed = ["skill_tags", "skill_type", "transfer_potential", "possible_role_contexts", "category"]
    bridge_never_passed = ["raw video", "face", "name", "audio", "private site details"]
    role_contexts = data.get("possible_role_contexts") or ["Craft mentor", "Precision assembly trainer", "Field maintenance skill transfer", "Cultural heritage restoration support", "Education / Succession"]
    role_cards = [
        {
            "title": "Craft Mentor",
            "category": "Craft / Traditional Skills",
            "summary": "For teaching basic hand positioning, timing, and material response to successors.",
            "tags": ["hand stability", "timing judgment", "material response"],
        },
        {
            "title": "Cultural Heritage Restoration Support",
            "category": "Craft / Traditional Skills",
            "summary": "For connecting preserved hand skills to restoration, teaching, and cultural succession contexts.",
            "tags": ["craft memory", "material response", "careful handling"],
        },
        {
            "title": "Precision Assembly Trainer",
            "category": "Manufacturing / Precision Work",
            "summary": "For transferring steady contact, pressure control, and repeatable hand movement.",
            "tags": ["gradual pressure", "controlled release", "repeatable motion"],
        },
        {
            "title": "Field Maintenance Skill Transfer",
            "category": "Field Maintenance",
            "summary": "For preserving practical field know-how that is difficult to explain only with manuals.",
            "tags": ["sequence memory", "timing", "safety habit"],
        },
    ]
    latest_review_url = url_for("compare", session_id=str(data.get("session_id", "")).strip()) if str(data.get("session_id", "")).strip() else url_for("index", _anchor="review")
    return render_template(
        "bridge.html",
        data=data,
        bridge_passed=bridge_passed,
        bridge_never_passed=bridge_never_passed,
        bridge_other_holders=other_holders,
        role_cards=role_cards,
        role_contexts=role_contexts,
        latest_review_url=latest_review_url,
    )


@app.route("/export/<session_id>/markdown")
def export_markdown_route(session_id: str):
    try:
        data = load_session_data(session_id)
    except FileNotFoundError:
        abort(404)
    output_path = _export_root(session_id) / "practice_memory.md"
    export_markdown(data.get("practice_memory", {}), output_path)
    return send_file(output_path, as_attachment=True, download_name=f"{session_id}-practice-memory.md", mimetype="text/markdown")


@app.route("/export/<session_id>/json")
def export_json_route(session_id: str):
    try:
        data = load_session_data(session_id)
    except FileNotFoundError:
        abort(404)
    output_path = _export_root(session_id) / "practice_memory.json"
    export_json(data.get("practice_memory", {}), output_path)
    return send_file(output_path, as_attachment=True, download_name=f"{session_id}-practice-memory.json", mimetype="application/json")


@app.route("/export/<session_id>/training-jsonl")
def export_training_jsonl_route(session_id: str):
    try:
        data = load_session_data(session_id)
    except FileNotFoundError:
        abort(404)
    output_path = _export_root(session_id) / "training.jsonl"
    export_training_jsonl(data, output_path)
    return send_file(output_path, as_attachment=True, download_name=f"{session_id}-training.jsonl", mimetype="application/x-ndjson")


@app.route("/asset/<session_id>/<kind>/<path:filename>")
def session_asset(session_id: str, kind: str, filename: str):
    if kind not in {"master", "practice", "apprentice"}:
        abort(404)
    if kind == "apprentice":
        kind = "practice"
    path = _frame_root(session_id, kind) / filename
    if not path.exists():
        abort(404)
    return send_file(path)


@app.context_processor
def inject_globals():
    return {"brand_name": "Kiwami Capture"}


if __name__ == "__main__":
    app.run(debug=True)
