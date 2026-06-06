from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request, send_file, url_for

from src.analyzer import analyze_practice
from src.exporter import export_json, export_markdown, export_training_jsonl
from src.schema import PracticeMemory
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


@app.route("/")
def index():
    return render_template(
        "index.html",
        sessions=list_session_summaries(),
    )


@app.route("/process", methods=["POST"])
def process():
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
    practice_memory = analysis["practice_memory"]

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
        "evidence_limits": analysis.get(
            "evidence_limits",
            [
                "Pressure is not directly measured.",
                "Exact mastery is not scored.",
                "Tool identity may be user-provided if not visually confirmed.",
            ],
        ),
        "motion_evidence_scan": {
            "selected_key_moments": analysis.get("selected_key_moments", []),
            "motion_score_summary": analysis.get("motion_score_summary", ""),
            "hand_evidence": analysis.get("hand_evidence", {}),
            "evidence_limits": analysis.get(
                "evidence_limits",
                [
                    "Pressure is not directly measured.",
                    "Exact mastery is not scored.",
                    "Tool identity may be user-provided if not visually confirmed.",
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
        "visual_evidence_note": analysis.get("visual_evidence_note", "Selected frames were reviewed locally. The user-provided craft label is not visually confirmed."),
        "visual_observation_summary": analysis.get("visual_observation_summary", "Selected frames were reviewed locally. The user-provided craft label is not visually confirmed."),
        "guidance_source": analysis.get("guidance_source", ""),
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
    save_session_data(session_id, session_data)
    return redirect(url_for("compare", session_id=session_id))


@app.route("/compare/<session_id>")
def compare(session_id: str):
    try:
        data = load_session_data(session_id)
    except FileNotFoundError:
        abort(404)

    master_frames = data.get("frames", {}).get("master", [])
    practice_frames = data.get("frames", {}).get("practice", []) or data.get("frames", {}).get("apprentice", [])
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
    data.setdefault(
        "evidence_limits",
        [
            "Pressure is not directly measured.",
            "Exact mastery is not scored.",
            "Tool identity may be user-provided if not visually confirmed.",
        ],
    )
    data.setdefault(
        "motion_evidence_scan",
        {
            "selected_key_moments": [],
            "motion_score_summary": "",
            "hand_evidence": {},
            "evidence_limits": [
                "Pressure is not directly measured.",
                "Exact mastery is not scored.",
                "Tool identity may be user-provided if not visually confirmed.",
            ],
        },
    )
    data.setdefault("visual_evidence_status", "weak")
    data.setdefault("visual_evidence_note", "Selected frames were reviewed locally. The user-provided craft label is not visually confirmed.")
    data.setdefault("visual_observation_summary", "Selected frames were reviewed locally. The user-provided craft label is not visually confirmed.")
    data.setdefault("guidance_source", "")
    for frames, kind in ((master_frames, "master"), (practice_frames, practice_kind)):
        for frame in frames:
            frame["url"] = _relative_frame_url(session_id, kind, frame["filename"])

    return render_template(
        "compare.html",
        data=data,
        master_frames=master_frames,
        practice_frames=practice_frames,
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
            "Pressure is not directly measured.",
            "Exact mastery is not scored.",
            "Tool identity may be user-provided if not visually confirmed.",
        ],
    )
    data.setdefault(
        "motion_evidence_scan",
        {
            "selected_key_moments": [],
            "motion_score_summary": "",
            "hand_evidence": {},
            "evidence_limits": [
                "Pressure is not directly measured.",
                "Exact mastery is not scored.",
                "Tool identity may be user-provided if not visually confirmed.",
            ],
        },
    )
    data.setdefault("visual_evidence_status", "weak")
    data.setdefault("visual_evidence_note", "Selected frames were reviewed locally. The user-provided craft label is not visually confirmed.")
    data.setdefault("visual_observation_summary", "Selected frames were reviewed locally. The user-provided craft label is not visually confirmed.")
    data.setdefault("guidance_source", "")
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
            "Pressure is not directly measured.",
            "Exact mastery is not scored.",
            "Tool identity may be user-provided if not visually confirmed.",
        ],
    )
    data.setdefault(
        "motion_evidence_scan",
        {
            "selected_key_moments": [],
            "motion_score_summary": "",
            "hand_evidence": {},
            "evidence_limits": [
                "Pressure is not directly measured.",
                "Exact mastery is not scored.",
                "Tool identity may be user-provided if not visually confirmed.",
            ],
        },
    )
    data.setdefault("visual_evidence_status", "weak")
    data.setdefault("visual_evidence_note", "Selected frames were reviewed locally. The user-provided craft label is not visually confirmed.")
    data.setdefault("visual_observation_summary", "Selected frames were reviewed locally. The user-provided craft label is not visually confirmed.")
    data.setdefault("guidance_source", "")
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
