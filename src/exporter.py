from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import PracticeMemory


def _coerce_memory(practice_memory: PracticeMemory | dict[str, Any]) -> PracticeMemory:
    if isinstance(practice_memory, PracticeMemory):
        return practice_memory
    return PracticeMemory.from_dict(practice_memory)


def _render_bullets(title: str, items: list[str]) -> list[str]:
    lines = [title]
    if items:
        lines.extend(f"- {item}" for item in items)
    else:
        lines.append("- None")
    return lines


def export_markdown(practice_memory: PracticeMemory | dict[str, Any], output_path: str | Path) -> str:
    memory = _coerce_memory(practice_memory)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Kiwami Practice Memory",
        "",
        f"## Craft\n{memory.craft}",
        f"## Skill Focus\n{memory.skill_focus}",
        *_render_bullets("## Watch Points", memory.watch_points),
        f"## Step-by-Step Motion\n{memory.step_by_step_motion}",
        f"## What to Copy\n{memory.what_to_copy}",
        f"## What to Avoid\n{memory.what_to_avoid}",
        f"## Success Sign\n{memory.success_sign}",
        f"## Next Practice Drill\n{memory.next_practice_drill}",
        f"## Guidance Source\n{memory.guidance_source or 'Unspecified'}",
        f"## Timing Cue\n{memory.timing_cue}",
        f"## Motion Cue\n{memory.motion_cue}",
        f"## Material Cue\n{memory.material_cue}",
        f"## Sound Cue\n{memory.sound_cue}",
        f"## Common Mistake\n{memory.common_mistake}",
        f"## Master Hint\n{memory.master_hint}",
        f"## Practice Task\n{memory.practice_task}",
        *_render_bullets("## Evidence", memory.evidence),
        f"## Expert Correction\n{memory.expert_correction or 'None'}",
        f"## Reviewer Note\n{memory.reviewer_note or 'None'}",
        f"## Skill Profile\n{memory.skill_profile or memory.craft}",
        f"## Skill Tags\n{', '.join(memory.skill_tags) if memory.skill_tags else 'None'}",
        f"## Skill Type\n{', '.join(memory.skill_type) if memory.skill_type else 'None'}",
        f"## Transfer Potential\n{', '.join(memory.transfer_potential) if memory.transfer_potential else 'None'}",
        f"## Shortage Relevance\n{memory.shortage_relevance or 'None'}",
        f"## Privacy Boundary\n{memory.privacy_boundary or 'None'}",
        f"## Archive Entry Path\n{memory.archive_entry_path or 'None'}",
        f"## Approved for Tuning\n{'Yes' if memory.approved_for_tuning else 'No'}",
        f"## Privacy Mode\n{memory.privacy_mode}",
        f"## Model Mode\n{memory.model_mode}",
        f"## Shareable\n{'Yes' if memory.shareable else 'No'}",
        "",
    ]
    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    return str(output_path)


def export_json(practice_memory: PracticeMemory | dict[str, Any], output_path: str | Path) -> str:
    memory = _coerce_memory(practice_memory)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(memory.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return str(output_path)


def export_training_jsonl(session_data: dict[str, Any], output_path: str | Path) -> str:
    memory = _coerce_memory(session_data.get("practice_memory", {}))
    record = {
        "instruction": "Generate Practice Memory from expert demonstration observations.",
        "input": {
            "craft": session_data.get("craft", ""),
            "context_note": session_data.get("master_hint", ""),
            "audio_context": session_data.get("audio_context", ""),
            "capture_mode": session_data.get("capture_mode", ""),
            "frame_observations": session_data.get("frame_observations", []),
            "frame_observation_summary": session_data.get("frame_observation_summary", ""),
            "frame_delta_summary": session_data.get("frame_delta_summary", ""),
            "selected_key_moments": session_data.get("selected_key_moments", []),
            "motion_score_summary": session_data.get("motion_score_summary", ""),
            "hand_evidence": session_data.get("hand_evidence", {}),
            "hand_evidence_status": session_data.get("hand_evidence_status", ""),
            "hand_visible_ratio": session_data.get("hand_visible_ratio", 0.0),
            "detected_hands": session_data.get("detected_hands", ""),
            "hand_keyframes": session_data.get("hand_keyframes", []),
            "evidence_limits": session_data.get("evidence_limits", []),
            "skill_tags": session_data.get("skill_tags", []),
            "skill_type": session_data.get("skill_type", []),
            "transfer_potential": session_data.get("transfer_potential", []),
            "shortage_relevance": session_data.get("shortage_relevance", ""),
            "privacy_boundary": session_data.get("privacy_boundary", ""),
            "archive_entry_path": session_data.get("archive_entry_path", ""),
            "job_bridge_ready": session_data.get("job_bridge_ready", False),
            "job_bridge_note": session_data.get("job_bridge_note", ""),
            "possible_role_contexts": session_data.get("possible_role_contexts", []),
            "skill_graph_profile": session_data.get("skill_graph_profile", []),
            "visual_layer": session_data.get("visual_layer", ""),
            "visual_evidence_status": session_data.get("visual_evidence_status", ""),
            "evidence_status": session_data.get("visual_evidence_status", ""),
            "guidance_source": session_data.get("guidance_source", ""),
            "tool_context_note": session_data.get("tool_context_note", ""),
            "timing_cue": memory.timing_cue,
            "motion_cue": memory.motion_cue,
            "material_cue": memory.material_cue,
            "sound_cue": memory.sound_cue,
            "evidence": memory.evidence,
        },
        "output": {
            "skill_focus": memory.skill_focus,
            "watch_points": memory.watch_points,
            "step_by_step_motion": memory.step_by_step_motion,
            "what_to_copy": memory.what_to_copy,
            "what_to_avoid": memory.what_to_avoid,
            "success_sign": memory.success_sign,
            "next_practice_drill": memory.next_practice_drill,
            "guidance_source": memory.guidance_source,
            "common_mistake": memory.common_mistake,
            "practice_task": memory.practice_task,
            "expert_correction": memory.expert_correction,
            "skill_tags": memory.skill_tags,
            "skill_type": memory.skill_type,
            "transfer_potential": memory.transfer_potential,
            "shortage_relevance": memory.shortage_relevance,
            "privacy_boundary": memory.privacy_boundary,
            "archive_entry_path": memory.archive_entry_path,
            "job_bridge_ready": session_data.get("job_bridge_ready", False),
            "job_bridge_note": session_data.get("job_bridge_note", ""),
            "possible_role_contexts": session_data.get("possible_role_contexts", []),
            "skill_graph_profile": session_data.get("skill_graph_profile", []),
        },
        "metadata": {
            "privacy_mode": memory.privacy_mode,
            "model_mode": memory.model_mode,
            "skill_profile": memory.skill_profile or session_data.get("craft", ""),
            "approved_for_tuning": bool(memory.approved_for_tuning),
            "archive_entry_path": session_data.get("archive_entry_path", ""),
            "source": "kiwami_capture_session",
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(output_path)
