from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _clean_list(values: Any) -> list[str]:
    if isinstance(values, list):
        return [str(item).strip() for item in values if str(item).strip()]
    if isinstance(values, str):
        return [line.strip() for line in values.splitlines() if line.strip()]
    return []


@dataclass
class PracticeMemory:
    craft: str
    skill_focus: str
    watch_points: list[str] = field(default_factory=list)
    step_by_step_motion: str = ""
    what_to_copy: str = ""
    what_to_avoid: str = ""
    success_sign: str = ""
    next_practice_drill: str = ""
    guidance_source: str = ""
    timing_cue: str = ""
    motion_cue: str = ""
    material_cue: str = ""
    sound_cue: str = ""
    common_mistake: str = ""
    master_hint: str = ""
    practice_task: str = ""
    evidence: list[str] = field(default_factory=list)
    privacy_mode: str = "local_only"
    model_mode: str = "mock"
    shareable: bool = True
    expert_correction: str = ""
    reviewer_note: str = ""
    approved_for_tuning: bool = False
    skill_profile: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PracticeMemory":
        return cls(
            craft=str(data.get("craft", "")).strip(),
            skill_focus=str(data.get("skill_focus", "")).strip(),
            watch_points=_clean_list(data.get("watch_points", [])),
            step_by_step_motion=str(data.get("step_by_step_motion", "")).strip(),
            what_to_copy=str(data.get("what_to_copy", "")).strip(),
            what_to_avoid=str(data.get("what_to_avoid", "")).strip(),
            success_sign=str(data.get("success_sign", "")).strip(),
            next_practice_drill=str(data.get("next_practice_drill", "")).strip(),
            guidance_source=str(data.get("guidance_source", "")).strip(),
            timing_cue=str(data.get("timing_cue", "")).strip(),
            motion_cue=str(data.get("motion_cue", "")).strip(),
            material_cue=str(data.get("material_cue", "")).strip(),
            sound_cue=str(data.get("sound_cue", "")).strip(),
            common_mistake=str(data.get("common_mistake", "")).strip(),
            master_hint=str(data.get("master_hint", "")).strip(),
            practice_task=str(data.get("practice_task", "")).strip(),
            evidence=_clean_list(data.get("evidence", [])),
            privacy_mode=str(data.get("privacy_mode", "local_only")).strip() or "local_only",
            model_mode=str(data.get("model_mode", "mock")).strip() or "mock",
            shareable=bool(data.get("shareable", True)),
            expert_correction=str(data.get("expert_correction", "")).strip(),
            reviewer_note=str(data.get("reviewer_note", "")).strip(),
            approved_for_tuning=bool(data.get("approved_for_tuning", False)),
            skill_profile=str(data.get("skill_profile", "")).strip() or str(data.get("craft", "")).strip(),
        )
