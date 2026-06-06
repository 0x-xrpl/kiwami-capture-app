from __future__ import annotations

from typing import Any


ARCHIVE_PRIVACY_BOUNDARY = "Raw video stays local. Only skill metadata can be shared."


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
