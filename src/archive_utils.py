from __future__ import annotations

import json
from pathlib import Path
import math
from typing import Any


ARCHIVE_PRIVACY_BOUNDARY = "Raw video stays local. Only skill metadata can be shared."
SKILL_GRAPH_DIMENSIONS = [
    ("hand_stability", "Hand"),
    ("timing_judgment", "Timing"),
    ("material_response", "Material"),
    ("tool_handling", "Tool"),
    ("pressure_control", "Pressure"),
    ("release_control", "Release"),
    ("repetition_readiness", "Repetition"),
    ("transfer_versatility", "Transfer"),
]


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _text_blob(*parts: Any) -> str:
    values: list[str] = []
    for part in parts:
        if isinstance(part, list):
            values.extend(str(item).strip() for item in part if str(item).strip())
        elif isinstance(part, dict):
            values.extend(str(item).strip() for item in part.values() if str(item).strip())
        else:
            text = str(part).strip()
            if text:
                values.append(text)
    return " ".join(values).lower()


def _contains(blob: str, terms: list[str]) -> bool:
    return any(term in blob for term in terms)


def _build_skill_tags(blob: str, hand_status: str, hand_detected: bool) -> list[str]:
    tags: list[str] = []

    def add(tag: str, condition: bool) -> None:
        if condition and tag not in tags:
            tags.append(tag)

    add("hand stability", hand_detected or hand_status == "detected" or _contains(blob, ["hand contact", "steady hands", "hand pressure"]))
    add("gradual pressure", _contains(blob, ["pressure", "light touch", "even pressure", "gentle"]))
    add("slow release", _contains(blob, ["release", "let go", "withdraw", "soft finish"]))
    add("timing judgment", _contains(blob, ["timing", "pause", "rhythm", "pace", "sequence"]))
    add("material response", _contains(blob, ["material", "clay", "surface", "texture", "wall", "rim", "shape"]))
    add("tool handling", _contains(blob, ["tool", "wheel", "brush", "knife", "spatula", "handle"]))
    add("fine motor control", _contains(blob, ["fine motor", "precision", "compact", "steady", "controlled"]))
    add("deliberate repetition", _contains(blob, ["repeat", "drill", "practice", "repeatable", "slowly"]))
    if len(tags) < 5:
        add("manual precision", True)
    if len(tags) < 5:
        add("embodied technique", True)
    if len(tags) < 5:
        add("timing judgment", True)
    if len(tags) < 5:
        add("tool handling", True)
    if len(tags) < 5:
        add("material response", True)
    return _unique(tags)[:8]


def _build_skill_types(skill_tags: list[str], blob: str) -> list[str]:
    types: list[str] = []

    def add(label: str, condition: bool) -> None:
        if condition and label not in types:
            types.append(label)

    add("manual precision", any(tag in skill_tags for tag in ("hand stability", "gradual pressure", "fine motor control", "tool handling")) or _contains(blob, ["precision", "steady", "controlled"]))
    add("material sensitivity", any(tag in skill_tags for tag in ("material response", "slow release")) or _contains(blob, ["material", "clay", "surface", "texture"]))
    add("timing control", any(tag in skill_tags for tag in ("timing judgment", "slow release", "deliberate repetition")) or _contains(blob, ["timing", "pause", "rhythm", "pace"]))
    add("tool coordination", any(tag in skill_tags for tag in ("tool handling", "hand stability")) or _contains(blob, ["tool", "wheel", "handle"]))
    if len(types) < 2:
        add("embodied technique", True)
    if len(types) < 2:
        add("field skill transfer", True)
    return _unique(types)[:4]


def _build_transfer_potential(skill_tags: list[str], blob: str) -> list[str]:
    domains: list[str] = []

    def add(label: str, condition: bool) -> None:
        if condition and label not in domains:
            domains.append(label)

    add("craft education", True)
    add("precision assembly", any(tag in skill_tags for tag in ("manual precision", "fine motor control", "tool handling")) or _contains(blob, ["precision", "tool"]))
    add("repair work", any(tag in skill_tags for tag in ("tool handling", "material response", "hand stability")) or _contains(blob, ["repair", "maintenance"]))
    add("cultural heritage restoration", any(tag in skill_tags for tag in ("material response", "embodied technique")) or _contains(blob, ["heritage", "restoration", "craft"]))
    add("food shaping", any(tag in skill_tags for tag in ("slow release", "gradual pressure", "timing judgment")) or _contains(blob, ["food", "shaping", "knead", "form"]))
    add("field maintenance training", any(tag in skill_tags for tag in ("tool handling", "timing judgment", "deliberate repetition")) or _contains(blob, ["field", "maintenance", "training"]))
    if len(domains) < 3:
        add("manual skills training", True)
    if len(domains) < 3:
        add("quality coaching", True)
    return _unique(domains)[:5]


def _shortage_relevance() -> str:
    return "This skill pattern is relevant where tacit hand judgment, timing, and material response are difficult to standardize through manuals alone."


def _dimension_presence(entry: dict[str, Any]) -> dict[str, float]:
    tags = {str(item).strip().lower() for item in entry.get("skill_tags", []) if str(item).strip()}
    types = {str(item).strip().lower() for item in entry.get("skill_type", []) if str(item).strip()}
    transfer = {str(item).strip().lower() for item in entry.get("transfer_potential", []) if str(item).strip()}
    hand = entry.get("hand_evidence") or {}
    if not isinstance(hand, dict):
        hand = {}
    hand_status = str(hand.get("hand_evidence_status") or entry.get("hand_evidence_status") or "").strip().lower()

    def has_any(*phrases: str) -> bool:
        return any(phrase in tags or phrase in types or phrase in transfer for phrase in phrases)

    values = {
        "hand_stability": 1.0 if hand_status == "detected" or has_any("hand stability", "fine motor control", "tool handling") else 0.0,
        "timing_judgment": 1.0 if has_any("timing judgment", "timing control", "deliberate repetition", "slow release") else 0.0,
        "material_response": 1.0 if has_any("material response", "material sensitivity", "craft education", "cultural heritage restoration") else 0.0,
        "tool_handling": 1.0 if has_any("tool handling", "tool coordination", "precision assembly") else 0.0,
        "pressure_control": 1.0 if has_any("gradual pressure", "manual precision") else 0.0,
        "release_control": 1.0 if has_any("slow release", "release control", "food shaping") else 0.0,
        "repetition_readiness": 1.0 if has_any("deliberate repetition", "field maintenance training", "practice") else 0.0,
        "transfer_versatility": min(1.0, len(transfer) / 5.0) if transfer else 0.0,
    }
    if not any(values.values()):
        values["hand_stability"] = 0.25
        values["timing_judgment"] = 0.25
        values["material_response"] = 0.25
        values["tool_handling"] = 0.25
    return values


def load_recent_archive_entries(limit: int = 12, exclude_session_id: str | None = None) -> list[dict[str, Any]]:
    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
    if not outputs_dir.exists():
        return []
    entries: list[dict[str, Any]] = []
    for child in outputs_dir.iterdir():
        if not child.is_dir():
            continue
        path = child / "archive_entry.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if exclude_session_id and str(payload.get("session_id", "")).strip() == exclude_session_id:
            continue
        entries.append(payload)
    entries.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return entries[:limit]


def build_skill_graph_profile(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned_entries = [entry for entry in entries if isinstance(entry, dict)]
    if not cleaned_entries:
        cleaned_entries = [{}]
    totals = {key: 0.0 for key, _ in SKILL_GRAPH_DIMENSIONS}
    for entry in cleaned_entries:
        presence = _dimension_presence(entry)
        for key, _label in SKILL_GRAPH_DIMENSIONS:
            totals[key] += presence.get(key, 0.0)
    count = float(len(cleaned_entries))
    profile: list[dict[str, Any]] = []
    for key, label in SKILL_GRAPH_DIMENSIONS:
        average = totals[key] / count if count else 0.0
        weight = int(round(max(0.0, min(5.0, average * 5.0))))
        profile.append(
            {
                "dimension": key,
                "label": label,
                "weight": weight,
                "presence": round(average, 2),
            }
        )
    return profile


def build_job_bridge_context(
    skill_tags: list[str],
    skill_types: list[str],
    transfer_potential: list[str],
    skill_graph_profile: list[dict[str, Any]],
) -> dict[str, Any]:
    captured_skill = [str(tag).strip() for tag in skill_tags[:4] if str(tag).strip()]
    transfer_domains = [str(item).strip() for item in transfer_potential[:3] if str(item).strip()]
    role_contexts = [
        "craft mentor",
        "precision assembly trainer",
        "cultural heritage restoration support",
        "repair technician training" if "repair work" in transfer_domains or "tool coordination" in skill_types else "",
        "field maintenance skill transfer" if "field maintenance training" in transfer_domains or "timing control" in skill_types else "",
        "hands-on education support",
    ]
    role_contexts = _unique([item for item in role_contexts if item])
    job_bridge_note = (
        "Job matching is not active in this MVP. Kiwami prepares the missing layer before job matching: reusable skill metadata."
    )
    return {
        "job_bridge_ready": True,
        "job_bridge_note": job_bridge_note,
        "possible_role_contexts": role_contexts,
        "skill_graph_profile": skill_graph_profile,
        "skill_graph_preview": {
            "captured_skill": captured_skill,
            "transfer_domains": transfer_domains,
            "job_bridge_note": job_bridge_note,
            "privacy_boundary": ARCHIVE_PRIVACY_BOUNDARY,
        },
    }


def build_skill_graph_visual(skill_graph_profile: list[dict[str, Any]]) -> dict[str, Any]:
    center = 150
    outer_radius = 88
    label_radius = 108
    axes: list[dict[str, Any]] = []
    polygon_points: list[str] = []
    count = max(1, len(SKILL_GRAPH_DIMENSIONS))
    ordered = {str(item.get("dimension", "")).strip(): item for item in skill_graph_profile if isinstance(item, dict)}
    for index, (key, label) in enumerate(SKILL_GRAPH_DIMENSIONS):
        angle = -math.pi / 2 + (2 * math.pi * index / count)
        weight = int(ordered.get(key, {}).get("weight", 0) or 0)
        weight = max(0, min(5, weight))
        ratio = weight / 5.0
        x = center + math.cos(angle) * outer_radius * ratio
        y = center + math.sin(angle) * outer_radius * ratio
        polygon_points.append(f"{x:.1f},{y:.1f}")
        axis_x = center + math.cos(angle) * outer_radius
        axis_y = center + math.sin(angle) * outer_radius
        label_x = center + math.cos(angle) * label_radius
        label_y = center + math.sin(angle) * label_radius
        anchors = "middle"
        if -math.pi / 2 < angle < math.pi / 2:
            anchors = "start"
        elif angle > math.pi / 2 or angle < -math.pi / 2:
            anchors = "end"
        axes.append(
            {
                "label": label,
                "weight": weight,
                "axis_x": axis_x,
                "axis_y": axis_y,
                "label_x": label_x,
                "label_y": label_y,
                "text_anchor": anchors,
            }
        )
    return {
        "center": center,
        "rings": [20, 40, 60, 80, 98],
        "axes": axes,
        "polygon_points": " ".join(polygon_points),
        "empty": not any(item.get("weight", 0) for item in skill_graph_profile),
    }


def build_archive_entry(
    session_id: str,
    created_at: str,
    session_data: dict[str, Any],
    analysis: dict[str, Any],
    archive_entry_path: str,
) -> dict[str, Any]:
    practice_memory = session_data.get("practice_memory") or analysis.get("practice_memory") or {}
    if not isinstance(practice_memory, dict):
        practice_memory = {}
    hand_evidence = session_data.get("hand_evidence") or analysis.get("hand_evidence") or {}
    if not isinstance(hand_evidence, dict):
        hand_evidence = {}
    blob = _text_blob(
        session_data.get("craft", ""),
        session_data.get("tool_name", ""),
        session_data.get("tool_type", ""),
        session_data.get("material", ""),
        session_data.get("process_step", ""),
        session_data.get("master_hint", ""),
        session_data.get("motion_score_summary", ""),
        session_data.get("frame_observation_summary", ""),
        session_data.get("visual_observation_summary", ""),
        session_data.get("selected_key_moments", []),
        practice_memory,
        hand_evidence,
    )
    hand_status = str(session_data.get("hand_evidence_status") or hand_evidence.get("hand_evidence_status") or "unavailable").strip()
    hand_detected = bool(session_data.get("hand_detected") or hand_evidence.get("hand_detected"))
    skill_tags = _build_skill_tags(blob, hand_status, hand_detected)
    skill_types = _build_skill_types(skill_tags, blob)
    transfer_potential = _build_transfer_potential(skill_tags, blob)
    shortage_relevance = _shortage_relevance()
    privacy_boundary = ARCHIVE_PRIVACY_BOUNDARY
    evidence_status = str(session_data.get("visual_evidence_status") or session_data.get("frame_evidence_status") or "unknown").strip()
    motion_evidence_scan = session_data.get("motion_evidence_scan") or {
        "selected_key_moments": session_data.get("selected_key_moments", []),
        "motion_score_summary": session_data.get("motion_score_summary", ""),
        "hand_evidence": hand_evidence,
        "evidence_limits": session_data.get("evidence_limits", []),
    }
    if not isinstance(motion_evidence_scan, dict):
        motion_evidence_scan = {}
    return {
        "session_id": session_id,
        "created_at": created_at,
        "craft_name": str(session_data.get("craft", "")).strip(),
        "tool_name": str(session_data.get("tool_name", "")).strip(),
        "tool_type": str(session_data.get("tool_type", "")).strip(),
        "material": str(session_data.get("material", "")).strip(),
        "process_step": str(session_data.get("process_step", "")).strip(),
        "master_hint": str(session_data.get("master_hint", "")).strip(),
        "model_mode": str(session_data.get("model_mode") or session_data.get("mode") or "").strip(),
        "guidance_source": str(session_data.get("guidance_source", "")).strip(),
        "evidence_status": evidence_status,
        "motion_evidence_scan": motion_evidence_scan,
        "frame_observations": session_data.get("frame_observations", []),
        "practice_memory": practice_memory,
        "skill_tags": skill_tags,
        "skill_type": skill_types,
        "transfer_potential": transfer_potential,
        "shortage_relevance": shortage_relevance,
        "privacy_boundary": privacy_boundary,
        "archive_entry_path": archive_entry_path,
    }
