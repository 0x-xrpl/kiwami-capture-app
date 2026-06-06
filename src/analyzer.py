from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from .adapters.liquid_vision_adapter import generate_visual_layer
from .adapters.liquid_text_adapter import LiquidResponseError, generate_practice_memory
from .model_status import liquid_config, liquid_server_reachable, mode_display_label, model_notice_for_failure
from .schema import PracticeMemory

try:
    import mediapipe as mp
except Exception:
    mp = None


def _craft_slug(craft: str) -> str:
    return (craft or "").strip().lower()


def _default_watch_points(craft: str) -> list[str]:
    if "pottery" in _craft_slug(craft) or "center" in _craft_slug(craft):
        return ["hand pressure", "center wobble", "water amount", "pause timing"]
    return ["tool angle", "pressure", "timing", "material response"]


def _clean_list(values: Any) -> list[str]:
    if isinstance(values, list):
        return [str(item).strip() for item in values if str(item).strip()]
    if isinstance(values, str):
        return [line.strip() for line in values.splitlines() if line.strip()]
    return []


def _default_master_motion_template(craft: str) -> dict[str, str]:
    if "pottery" in _craft_slug(craft) or "center" in _craft_slug(craft):
        return {
            "start": "Set both hands lightly on the clay as the wheel begins.",
            "stabilize": "Apply even pressure and keep the clay centered.",
            "pause": "Ease off briefly when the center settles.",
            "release": "Release pressure without letting the clay lean.",
        }
    return {
        "start": "Enter the motion smoothly and keep the setup compact.",
        "stabilize": "Hold a steady rhythm until the motion becomes stable.",
        "pause": "Pause only long enough for the material or tool to settle.",
        "release": "Finish the motion without adding extra force.",
    }


def _select_master_keyframes(master_frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not master_frames:
        return []
    index = len(master_frames) // 2
    frame = master_frames[index]
    path = str(frame.get("path", "")).strip()
    if path:
        return [frame]
    for candidate in master_frames:
        if str(candidate.get("path", "")).strip():
            return [candidate]
    return []


def _selected_key_moments(selected_frames: list[dict[str, Any]]) -> list[str]:
    moments: list[str] = []
    for index, frame in enumerate(selected_frames, start=1):
        label = str(frame.get("label") or frame.get("filename") or f"Key frame {index}").strip()
        if label:
            moments.append(label)
    return moments


def _is_fallback_frame(path: str) -> bool:
    return Path(path).name.startswith("fallback_")


def _scan_hand_evidence(selected_frames: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "hand_detected": False,
        "hand_visible_ratio": 0.0,
        "detected_hands": "unavailable" if mp is None else "unknown",
        "hand_keyframes": [],
        "hand_evidence_status": "unavailable" if mp is None else "not_detected",
    }
    if mp is None or not selected_frames:
        return result

    try:
        hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception:
        result["hand_evidence_status"] = "error"
        result["detected_hands"] = "unknown"
        return result

    detected_frames = 0
    processed_frames = 0
    hand_labels: set[str] = set()
    hand_keyframes: list[str] = []

    with hands:
        for index, frame in enumerate(selected_frames, start=1):
            path = str(frame.get("path", "")).strip()
            if not path or _is_fallback_frame(path) or not Path(path).exists():
                continue
            image = cv2.imread(path)
            if image is None:
                continue
            processed_frames += 1
            try:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
            except Exception:
                continue
            if not results.multi_hand_landmarks:
                continue
            detected_frames += 1
            frame_name = str(frame.get("filename") or frame.get("label") or f"Key frame {index}").strip()
            if frame_name:
                hand_keyframes.append(frame_name)
            handedness = getattr(results, "multi_handedness", None) or []
            for entry in handedness:
                try:
                    label = str(entry.classification[0].label).strip().lower()
                except Exception:
                    label = ""
                if label in {"left", "right"}:
                    hand_labels.add(label)

    if detected_frames > 0:
        if "left" in hand_labels and "right" in hand_labels:
            detected_hands = "both"
        elif "left" in hand_labels:
            detected_hands = "left"
        elif "right" in hand_labels:
            detected_hands = "right"
        else:
            detected_hands = "unknown"
        result.update(
            {
                "hand_detected": True,
                "hand_visible_ratio": round(detected_frames / float(processed_frames or len(selected_frames) or 1), 3),
                "detected_hands": detected_hands,
                "hand_keyframes": list(dict.fromkeys(hand_keyframes)),
                "hand_evidence_status": "detected",
            }
        )
        return result

    result["detected_hands"] = "unknown"
    result["hand_evidence_status"] = "not_detected"
    return result


def _describe_frame(path: str, index: int) -> tuple[str, bool, float]:
    file_path = Path(path)
    if not path or not file_path.exists() or _is_fallback_frame(path):
        return f"Frame {index} is a fallback placeholder frame.", False, 0.0

    image = cv2.imread(path)
    if image is None:
        return f"Frame {index} could not be read locally.", False, 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean, stddev = cv2.meanStdDev(gray)
    mean_value = float(mean[0][0])
    std_value = float(stddev[0][0])
    small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 60, 160)
    edge_density = float(cv2.countNonZero(edges)) / float(edges.size or 1)

    motion_level = "static"
    if std_value > 42 or edge_density > 0.08:
        motion_level = "strong"
    elif std_value > 28 or edge_density > 0.05:
        motion_level = "moderate"
    elif std_value > 18 or edge_density > 0.03:
        motion_level = "slight"

    if mean_value < 24 and std_value < 10:
        return f"Frame {index} is dim and visually unclear.", False, edge_density

    color_note = "warm" if mean_value > 150 else "dark" if mean_value < 90 else "neutral"
    if edge_density > 0.08:
        shape_note = "distinct shapes and edges"
    elif edge_density > 0.04:
        shape_note = "some visible structure"
    else:
        shape_note = "soft or minimal structure"

    visible = True
    return (
        f"Frame {index} shows a {color_note} scene with {shape_note}; motion level is {motion_level}.",
        visible,
        edge_density,
    )


def _frame_observations(selected_frames: list[dict[str, Any]]) -> tuple[list[str], str, str, str]:
    observations: list[str] = []
    motion_samples: list[float] = []
    previous_small: cv2.Mat | None = None
    visible_count = 0
    frame_notes: list[str] = []

    for index, frame in enumerate(selected_frames, start=1):
        path = str(frame.get("path", "")).strip()
        observation, visible, _ = _describe_frame(path, index)
        observations.append(observation)
        frame_notes.append(observation)
        if visible:
            visible_count += 1
        image = cv2.imread(path) if path and Path(path).exists() and not _is_fallback_frame(path) else None
        if image is None:
            previous_small = None
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        if previous_small is not None:
            diff = cv2.absdiff(previous_small, small)
            motion_samples.append(float(cv2.mean(diff)[0]) / 255.0)
        previous_small = small

    if not selected_frames:
        frame_status = "visual_uncertain"
    elif visible_count == 0:
        frame_status = "visual_uncertain"
    elif len(observations) == 1:
        frame_status = "generic_visible"
    else:
        frame_status = "generic_visible"

    if motion_samples and sum(motion_samples) / len(motion_samples) > 0.04:
        motion_summary = "Visible change across the selected frames is present."
    else:
        motion_summary = "Little visible change across the selected frames."

    visible_scene = observations[0] if observations else "Selected frames were reviewed locally."
    if len(observations) > 1:
        visible_scene = f"{observations[0]} {observations[1]}"

    if len(selected_frames) >= 2:
        first_path = str(selected_frames[0].get("path", "")).strip()
        last_path = str(selected_frames[-1].get("path", "")).strip()
        first_img = cv2.imread(first_path) if first_path and Path(first_path).exists() and not _is_fallback_frame(first_path) else None
        last_img = cv2.imread(last_path) if last_path and Path(last_path).exists() and not _is_fallback_frame(last_path) else None
        if first_img is not None and last_img is not None:
            first_gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
            last_gray = cv2.cvtColor(last_img, cv2.COLOR_BGR2GRAY)
            first_small = cv2.resize(first_gray, (160, 90), interpolation=cv2.INTER_AREA)
            last_small = cv2.resize(last_gray, (160, 90), interpolation=cv2.INTER_AREA)
            brightness_delta = float(cv2.mean(last_small)[0] - cv2.mean(first_small)[0])
            color_first = cv2.mean(first_img)
            color_last = cv2.mean(last_img)
            color_shift = abs(float(color_last[0] - color_first[0])) + abs(float(color_last[1] - color_first[1])) + abs(float(color_last[2] - color_first[2]))
            diff_amount = float(cv2.mean(cv2.absdiff(first_small, last_small))[0]) / 255.0
            if diff_amount < 0.025:
                delta_note = "The scene is mostly static across the selected frames."
            elif diff_amount < 0.05:
                delta_note = "The scene changes slightly across the selected frames."
            else:
                delta_note = "The scene changes visibly across the selected frames."
            frame_summary = f"Brightness change is {brightness_delta:+.1f}; color shift is {color_shift:.1f}; frame difference amount is {diff_amount:.3f}. {delta_note}"
        else:
            frame_summary = "Frame delta could not be measured locally."
    else:
        frame_summary = "Frame delta could not be measured locally."

    observation_summary = f"{visible_scene} {motion_summary} {frame_summary}".strip()
    return observations, frame_status, observation_summary, frame_summary


def _local_visual_layer(craft: str, master_hint: str, selected_frames: list[dict[str, Any]]) -> dict[str, Any]:
    frame_labels = [str(frame.get("label") or f"Key frame {index + 1}").strip() for index, frame in enumerate(selected_frames)]
    watch_points = _default_watch_points(craft)
    motion_cues = [f"Keep the motion steady around {watch_points[0]}"] if watch_points else ["Keep the motion steady and compact."]
    return {
        "visual_layer": "Local Keyframes",
        "suggested_craft_name": craft,
        "suggested_context_note": master_hint,
        "suggested_key_moments": frame_labels or ["Early keyframe", "Middle keyframe"],
        "what_successors_should_notice": watch_points,
        "motion_cues": motion_cues,
        "master_motion_template": _default_master_motion_template(craft),
        "visual_confidence": "local",
    }


def _merge_visual_layer(base: dict[str, Any], vision: dict[str, Any]) -> dict[str, Any]:
    template = dict(base.get("master_motion_template", {}))
    vision_template = vision.get("master_motion_template")
    if isinstance(vision_template, dict):
        for key in ("start", "stabilize", "pause", "release"):
            value = str(vision_template.get(key, "")).strip()
            if value:
                template[key] = value
    return {
        "visual_layer": "Liquid Vision",
        "suggested_craft_name": str(vision.get("suggested_craft_name") or base.get("suggested_craft_name") or "").strip() or base.get("suggested_craft_name", ""),
        "suggested_context_note": str(vision.get("suggested_context_note") or base.get("suggested_context_note") or "").strip() or base.get("suggested_context_note", ""),
        "suggested_key_moments": _clean_list(vision.get("suggested_key_moments")) or base.get("suggested_key_moments", []),
        "what_successors_should_notice": _clean_list(vision.get("what_successors_should_notice")) or base.get("what_successors_should_notice", []),
        "motion_cues": _clean_list(vision.get("motion_cues")) or base.get("motion_cues", []),
        "master_motion_template": template,
        "visual_confidence": str(vision.get("visual_confidence") or base.get("visual_confidence") or "high").strip() or "high",
    }


def _visual_evidence_assessment(
    visual_layer_data: dict[str, Any],
    liquid_vision_debug: dict[str, Any],
    frame_status: str,
    frame_summary: str,
) -> tuple[str, str]:
    if visual_layer_data.get("visual_layer") == "Liquid Vision":
        repaired = str(liquid_vision_debug.get("liquid_error") or "").startswith("Repaired")
        if bool(liquid_vision_debug.get("parsed_json_success")) and not repaired and str(visual_layer_data.get("visual_confidence", "")).lower() in {"medium", "high", "strong"} and frame_status != "visual_uncertain":
            return "confirmed_craft", "Craft label visually confirmed from the selected frames."
        if frame_status == "visual_uncertain":
            return "visual_uncertain", "Selected frames are too unclear to confirm craft-specific guidance."
        return "mismatch_possible", f"{frame_summary} The user-provided craft label is not visually confirmed."
    if frame_status == "visual_uncertain":
        return "visual_uncertain", "Selected frames are too unclear to confirm craft-specific guidance."
    return "generic_visible", f"{frame_summary} The user-provided craft label is not visually confirmed."


def _visual_observation_summary(status: str, frame_summary: str, frame_observations: list[str]) -> str:
    if frame_summary:
        base = frame_summary
    else:
        base = "Selected frames were reviewed locally."
    if frame_observations:
        base = frame_observations[0]
    if status == "confirmed_craft":
        return f"{base} Craft-specific motion is visually confirmed from the selected frames."
    if status == "visual_uncertain":
        return f"{base} Craft-specific guidance is not visually confirmed."
    return f"{base} The user-provided craft label is not visually confirmed."


def _hand_evidence_summary(hand_evidence: dict[str, Any]) -> str:
    status = str(hand_evidence.get("hand_evidence_status", "unavailable"))
    if status == "detected":
        count = len(hand_evidence.get("hand_keyframes", []))
        hand_side = str(hand_evidence.get("detected_hands", "unknown"))
        if hand_side == "both":
            return f"Both hands were detected in {count} selected keyframe(s)."
        elif hand_side in {"left", "right"}:
            hand_text = f"the {hand_side} hand"
        else:
            hand_text = "hand contact"
        return f"{hand_text.capitalize()} was detected in {count} selected keyframe(s)."
    if status == "not_detected":
        return "No hands were detected in the selected keyframes."
    if status == "error":
        return "Hand evidence scan failed, so the app fell back to OpenCV-only evidence."
    return "Hand evidence scan was unavailable, so the app used OpenCV-only evidence."


def _motion_score_summary(frame_observation_summary: str, hand_evidence: dict[str, Any]) -> str:
    parts = [frame_observation_summary or "Selected frames were reviewed locally."]
    parts.append(_hand_evidence_summary(hand_evidence))
    return " ".join(part for part in parts if part).strip()


def _hand_guided_practice(memory: PracticeMemory, hand_evidence: dict[str, Any]) -> PracticeMemory:
    if str(hand_evidence.get("hand_evidence_status", "")).strip() != "detected":
        return memory

    hand_keyframes = [str(item).strip() for item in hand_evidence.get("hand_keyframes", []) if str(item).strip()]
    if hand_keyframes:
        evidence_line = f"Hand evidence was visible in {', '.join(hand_keyframes[:2])}."
        if evidence_line not in memory.evidence:
            memory.evidence.insert(0, evidence_line)

    if not memory.motion_cue or "hand" not in memory.motion_cue.lower():
        memory.motion_cue = "Watch where the hands contact the material and keep the release slow."
    if not memory.what_to_copy or "hand" not in memory.what_to_copy.lower():
        memory.what_to_copy = "Copy the steady contact and slow release visible in the hands."
    if not memory.what_to_avoid or "hand" not in memory.what_to_avoid.lower():
        memory.what_to_avoid = "Avoid sudden one-sided pressure or a rushed release."
    if not memory.practice_task or "hand" not in memory.practice_task.lower():
        memory.practice_task = memory.practice_task or "Watch where the hands contact the material, then repeat the steady contact and slow release."
    return memory


def _apply_visual_evidence_guard(memory: PracticeMemory, status: str, frame_summary: str, context_guided: bool = False) -> PracticeMemory:
    if status == "confirmed_craft":
        return memory
    if context_guided and status != "visual_uncertain":
        memory.master_hint = "User-provided hint was used as context, not visual truth."
        if not memory.evidence:
            memory.evidence = [frame_summary or "Selected frames were reviewed locally."]
        if memory.evidence and memory.evidence[0] != "Context-guided fallback after Liquid LFM attempt":
            memory.evidence.insert(0, "Context-guided fallback after Liquid LFM attempt")
        return memory
    memory.skill_focus = frame_summary or "Selected frames do not visually confirm the craft label."
    memory.watch_points = ["selected frames", "visible scene", "visible motion" if "motion" in frame_summary.lower() else "frame detail"]
    memory.master_hint = "User-provided hint exists, but it was not applied because the frames do not confirm the craft label."
    if status == "visual_uncertain":
        memory.timing_cue = "Visible timing can be described only at the scene level; no craft-specific timing cue is confirmed."
        memory.motion_cue = "No craft-specific hand or tool motion is visually confirmed from these frames."
        memory.material_cue = "No craft-specific material response is visually confirmed from these frames."
        memory.sound_cue = "No reliable craft-specific sound cue is visually confirmed from these frames."
        memory.common_mistake = "No craft-specific mistake can be identified from these frames."
        memory.practice_task = "Review frames with clearer visible action before refining the practice guidance."
    else:
        memory.timing_cue = "Visible timing can be described only at the scene level; no craft-specific timing cue is confirmed."
        memory.motion_cue = "No craft-specific hand or tool motion is visually confirmed from these frames."
        memory.material_cue = "No craft-specific material response is visually confirmed from these frames."
        memory.sound_cue = "No reliable craft-specific sound cue is visually confirmed from these frames."
        memory.common_mistake = "No craft-specific mistake can be identified from these frames."
        memory.practice_task = "Review the selected frames and confirm the craft label before refining the practice guidance."
    memory.evidence = [
        frame_summary or "Selected frames were reviewed locally.",
        "The user-provided craft label is not visually confirmed.",
    ]
    return memory


def _apply_actionable_guidance(memory: PracticeMemory, status: str, frame_observations: list[str], frame_summary: str) -> PracticeMemory:
    if status == "confirmed_craft":
        memory.step_by_step_motion = memory.step_by_step_motion or (
            "1. Start from the visible setup in the selected frames.\n"
            "2. Keep the motion compact and steady through the middle frame.\n"
            "3. Finish with the controlled release or settle shown at the end."
        )
        memory.what_to_copy = memory.what_to_copy or "Copy the steadier setup, the compact rhythm, and the controlled finish visible in the selected frames."
        memory.what_to_avoid = memory.what_to_avoid or "Avoid adding extra force, widening the motion, or changing the timing beyond what the frames show."
        memory.success_sign = memory.success_sign or "The motion stays controlled across the selected frames and the visible change remains consistent."
        memory.next_practice_drill = memory.next_practice_drill or "Repeat the motion once at half speed, then compare your result against the selected frames."
    else:
        memory.step_by_step_motion = memory.step_by_step_motion or (
            "1. Review the selected frames and note only the visible setup.\n"
            "2. Reproduce the confirmed motion in a short repeat.\n"
            "3. Stop if you need to guess details that are not visible."
        )
        memory.what_to_copy = memory.what_to_copy or "Copy only the visible setup, spacing, and steadier motion shown in the selected frames."
        memory.what_to_avoid = memory.what_to_avoid or "Avoid adding craft-specific details that are not supported by the selected frames."
        memory.success_sign = memory.success_sign or "The next attempt matches the visible scene more closely without adding unsupported assumptions."
        memory.next_practice_drill = memory.next_practice_drill or "Rewatch the selected frames, then repeat a short drill using only the confirmed visible cues."
    if not memory.practice_task:
        memory.practice_task = memory.next_practice_drill or "Review the selected frames before repeating the motion."
    if not memory.evidence:
        memory.evidence = [frame_summary or "Selected frames were reviewed locally."]
    if frame_observations and not memory.evidence[0].startswith(frame_observations[0]):
        memory.evidence.insert(0, frame_observations[0])
    return memory


def _context_text(*parts: str) -> str:
    return " ".join(part.strip().lower() for part in parts if part and part.strip())


def _has_pottery_context(*parts: str) -> bool:
    text = _context_text(*parts)
    if not text:
        return False
    keywords = (
        "pottery",
        "ceramic",
        "ceramics",
        "clay",
        "wheel",
        "centering",
        "center",
        "rim",
        "wall",
        "vessel",
        "throwing",
        "trim",
        "trimming",
        "rib",
        "shaping",
        "kiln",
        "turntable",
    )
    return any(keyword in text for keyword in keywords)


def _pottery_context_memory(craft: str, master_hint: str, mode: str, guidance_source: str) -> PracticeMemory:
    return PracticeMemory(
        craft=craft,
        skill_focus="Keep the vessel stable while you shape the wall and rim",
        watch_points=["hand position", "steady contact", "wall stability", "rim wobble"],
        step_by_step_motion=(
            "1. Set both hands close to the vessel before adding pressure.\n"
            "2. Support the outside wall while the other hand steadies the form.\n"
            "3. Keep contact smooth through the shaping phase.\n"
            "4. Release gradually and check whether the rim and wall remain stable."
        ),
        what_to_copy="Copy the close hand position, steady contact, gradual pressure, controlled release, and the way the rim and wall are watched for wobble.",
        what_to_avoid="Avoid sudden squeezing, one-sided pressure, pulling away too quickly, or chasing wobble before the form is stable.",
        success_sign="The vessel stays upright, the rim stays round, the wall does not lean, and the motion remains controlled after release.",
        next_practice_drill="Repeat a 20-second contact-and-release drill. Focus only on steady hand contact and slow release.",
        timing_cue="when both hands settle into contact and the wall begins to stabilize",
        motion_cue="keep both hands close and use steady, even contact",
        material_cue="watch whether the wall stays round and the vessel remains upright",
        sound_cue="look for a steady working rhythm without abrupt scraping or correction",
        common_mistake="sudden squeezing or pulling away before the wall and rim are stable",
        master_hint=master_hint,
        practice_task="Repeat a 20-second contact-and-release drill. Keep your hands steady and release slowly if the wall starts to wobble.",
        evidence=[
            f"{guidance_source or 'Context-guided fallback'}",
            "Based on selected frames and user-provided craft context, this is a lesson-style pottery guide.",
        ],
        privacy_mode="local_only",
        model_mode=mode,
        shareable=True,
        guidance_source=guidance_source,
    )


def _build_common_memory(craft: str, master_hint: str, mode: str) -> PracticeMemory:
    if "pottery" in _craft_slug(craft) or "center" in _craft_slug(craft):
        return PracticeMemory(
            craft=craft,
            skill_focus="Stabilize the material before shaping",
            watch_points=["hand pressure", "center wobble", "water amount", "pause timing"],
            step_by_step_motion="1. Start with light contact in the visible setup.\n2. Keep the motion steady through the middle of the action.\n3. Release without adding extra force.",
            what_to_copy="Copy the steadier setup, the controlled rhythm, and the lighter touch that are visible in the selected frames.",
            what_to_avoid="Avoid forcing the motion, widening the contact, or adding pressure that is not visible in the frames.",
            success_sign="The motion stays centered, steady, and controlled across the selected frames.",
            next_practice_drill="Repeat a short drill at half speed and compare each attempt against the selected frames.",
            timing_cue="the first 5 seconds after the wheel starts",
            motion_cue="hands stay close and apply light, even pressure",
            material_cue="the clay should rise smoothly without leaning",
            sound_cue="steady wheel sound with minimal scraping",
            common_mistake="pressing too hard before the clay is centered",
            master_hint=master_hint,
            practice_task="Repeat a 20-second centering drill. Stop if the clay starts leaning, then reset hand pressure.",
            evidence=[
                "Master clip shows stable hand pressure",
                "Practice clip shows visible center wobble",
                "Practice clip applies pressure before the clay stabilizes",
            ],
            privacy_mode="local_only",
            model_mode=mode,
            shareable=True,
        )

    focus = f"Identify the stable motion that makes {craft or 'the task'} reliable"
    return PracticeMemory(
        craft=craft,
        skill_focus=focus,
        watch_points=_default_watch_points(craft),
        step_by_step_motion="1. Start from the visible setup shown in the selected frames.\n2. Keep the motion compact and steady through the middle frame.\n3. Finish without adding extra force or unnecessary correction.",
        what_to_copy="Copy the stable setup, the compact rhythm, and the visible control shown in the selected frames.",
        what_to_avoid="Avoid adding details, force, or timing that are not visible in the selected frames.",
        success_sign="The next attempt matches the visible rhythm and stays controlled across the selected frames.",
        next_practice_drill="Repeat the motion slowly, then compare your result to the selected frames before speeding up.",
        timing_cue="the first stable moment after the task begins",
        motion_cue="keep the core motion compact, steady, and repeatable",
        material_cue="the material or tool should remain controlled, not forced",
        sound_cue="look for a steady, calm rhythm with fewer corrective noises",
        common_mistake="acting too early or using excess force before the motion settles",
        master_hint=master_hint,
        practice_task=f"Repeat the core motion slowly and stop at the first sign of instability in {craft or 'the task'}.",
        evidence=[
            f"Master clip shows a more stable version of {craft or 'the motion'}",
            f"Practice clip shows the difference learners usually miss",
            "Key frames were extracted locally for review",
        ],
        privacy_mode="local_only",
        model_mode=mode,
        shareable=True,
    )


def mock_analysis(craft: str, master_hint: str) -> PracticeMemory:
    memory = _build_common_memory(craft, master_hint, "mock")
    memory.skill_focus = "Stabilize the clay before shaping"
    memory.practice_task = "Repeat a 20-second centering drill. Keep your hands steady and reset if the clay leans."
    return memory


def rule_based_analysis(craft: str, master_hint: str) -> PracticeMemory:
    return _build_common_memory(craft, master_hint, "rule")


def _comparison_notes(master_frames: list[dict[str, Any]], apprentice_frames: list[dict[str, Any]], craft: str) -> list[str]:
    return [
        f"Master frames reviewed locally: {len(master_frames)}",
        f"Practice frames reviewed locally: {len(apprentice_frames)}",
        f"Craft context: {craft or 'unspecified'}",
    ]


def _master_only_notes(master_frames: list[dict[str, Any]], craft: str, audio_context: str) -> list[str]:
    notes = [
        f"Master frames reviewed locally: {len(master_frames)}",
        f"Craft context: {craft or 'unspecified'}",
    ]
    if audio_context:
        notes.append(f"Audio context provided: {audio_context}")
    return notes


def _difference_summary(memory: PracticeMemory, mode: str, visual_status: str, frame_summary: str) -> str:
    if visual_status != "confirmed_craft":
        return frame_summary or "Selected frames were reviewed locally."
    if mode == "mock":
        return "Expert motion stays steady and light while the learner over-presses and lets the center wobble."
    if mode == "liquid lfm":
        return "Liquid LFM generated Practice Memory from local structured observations and the master hint."
    return memory.skill_focus


def _fill_missing_fields(memory: PracticeMemory, base_memory: PracticeMemory, craft: str, master_hint: str) -> PracticeMemory:
    memory.craft = memory.craft or craft
    memory.skill_focus = memory.skill_focus or base_memory.skill_focus
    memory.watch_points = memory.watch_points or base_memory.watch_points
    memory.step_by_step_motion = memory.step_by_step_motion or base_memory.step_by_step_motion
    memory.what_to_copy = memory.what_to_copy or base_memory.what_to_copy
    memory.what_to_avoid = memory.what_to_avoid or base_memory.what_to_avoid
    memory.success_sign = memory.success_sign or base_memory.success_sign
    memory.next_practice_drill = memory.next_practice_drill or base_memory.next_practice_drill
    memory.guidance_source = memory.guidance_source or base_memory.guidance_source
    memory.timing_cue = memory.timing_cue or base_memory.timing_cue
    memory.motion_cue = memory.motion_cue or base_memory.motion_cue
    memory.material_cue = memory.material_cue or base_memory.material_cue
    memory.sound_cue = memory.sound_cue or base_memory.sound_cue
    memory.common_mistake = memory.common_mistake or base_memory.common_mistake
    memory.master_hint = memory.master_hint or master_hint
    memory.practice_task = memory.practice_task or base_memory.practice_task
    memory.evidence = memory.evidence or base_memory.evidence
    memory.privacy_mode = memory.privacy_mode or "local_only"
    memory.model_mode = memory.model_mode or "Liquid LFM"
    memory.shareable = True if memory.shareable is None else bool(memory.shareable)
    memory.expert_correction = memory.expert_correction or ""
    memory.reviewer_note = memory.reviewer_note or ""
    memory.approved_for_tuning = bool(memory.approved_for_tuning)
    memory.skill_profile = memory.skill_profile or craft
    return memory


def analyze_practice(
    master_frames: list[dict[str, Any]],
    apprentice_frames: list[dict[str, Any]],
    master_hint: str,
    craft: str,
    mode: str,
    audio_context: str = "",
    tool_name: str = "",
    tool_type: str = "",
    material: str = "",
    process_step: str = "",
    master_metadata: dict[str, Any] | None = None,
    ghost_motion_overlay: str = "",
) -> dict[str, Any]:
    mode = (mode or "mock").strip().lower().replace("_", " ")
    liquid_debug = {
        "liquid_request_url": "",
        "liquid_model": "",
        "liquid_http_status": None,
        "liquid_raw_snippet": "",
        "liquid_error": "",
        "parsed_json_success": False,
    }
    liquid_vision_debug = {
        "liquid_request_url": "",
        "liquid_model": "",
        "liquid_http_status": None,
        "liquid_raw_snippet": "",
        "liquid_error": "",
        "parsed_json_success": False,
    }
    has_practice_clip = bool(apprentice_frames)
    capture_mode_display = "Comparison Capture" if has_practice_clip else "Master-only Capture"
    pottery_context = _has_pottery_context(craft, master_hint, tool_name, tool_type, material, process_step, audio_context)
    selected_master_frames = _select_master_keyframes(master_frames)
    selected_key_moments = _selected_key_moments(selected_master_frames)
    base_visual_layer = _local_visual_layer(craft, master_hint, selected_master_frames)
    frame_observations, frame_status, frame_observation_summary, frame_delta_summary = _frame_observations(selected_master_frames)
    hand_evidence = _scan_hand_evidence(selected_master_frames)
    motion_score_summary = _motion_score_summary(frame_observation_summary, hand_evidence)
    if mode == "rule":
        memory = _pottery_context_memory(craft, master_hint, "rule", "Frame-confirmed guidance") if pottery_context else rule_based_analysis(craft, master_hint)
        model_mode_display = "Rule"
        model_notice = ""
        debug_snippet = ""
        summary_mode = "rule"
        capture_notes = _comparison_notes(master_frames, apprentice_frames, craft) if has_practice_clip else _master_only_notes(master_frames, craft, audio_context)
        visual_layer_data = base_visual_layer
    elif mode in {"liquid lfm", "liquid text", "local liquid lfm", "liquid"}:
        base_memory = _pottery_context_memory(craft, master_hint, "rule", "Frame-confirmed guidance") if pottery_context else rule_based_analysis(craft, master_hint)
        if has_practice_clip:
            comparison_notes = _comparison_notes(master_frames, apprentice_frames, craft)
            focus_notes = comparison_notes
        else:
            comparison_notes = []
            focus_notes = _master_only_notes(master_frames, craft, audio_context)
        visual_layer_data = base_visual_layer
        if selected_master_frames:
            try:
                vision_data, liquid_vision_debug = generate_visual_layer(craft, master_hint, selected_master_frames)
                visual_layer_data = _merge_visual_layer(base_visual_layer, vision_data)
                visual_layer_data["visual_layer"] = "Liquid Vision"
            except LiquidResponseError as exc:
                liquid_vision_debug = {
                    "liquid_request_url": exc.request_url,
                    "liquid_model": exc.model,
                    "liquid_http_status": exc.http_status,
                    "liquid_raw_snippet": (exc.raw_snippet or exc.snippet or "")[:1000],
                    "liquid_error": str(exc),
                    "parsed_json_success": bool(exc.parsed_json_success),
                }
            except Exception as exc:
                liquid_vision_debug = {
                    "liquid_request_url": "",
                    "liquid_model": "",
                    "liquid_http_status": None,
                    "liquid_raw_snippet": "",
                    "liquid_error": str(exc),
                    "parsed_json_success": False,
                }
        else:
            liquid_vision_debug = {
                "liquid_request_url": "",
                "liquid_model": "",
                "liquid_http_status": None,
                "liquid_raw_snippet": "",
                "liquid_error": "No selected master keyframes available.",
                "parsed_json_success": False,
            }
    visual_evidence_status, visual_evidence_note = _visual_evidence_assessment(visual_layer_data, liquid_vision_debug, frame_status, frame_delta_summary)
    visual_observation_summary = _visual_observation_summary(visual_evidence_status, frame_observation_summary, frame_observations)
    guidance_source = "Frame-confirmed guidance" if visual_evidence_status == "confirmed_craft" else ("Context-guided fallback after Liquid LFM attempt" if pottery_context else "User-provided label not visually confirmed")
    observation_data = {
        "craft": craft,
        "master_hint": master_hint,
        "tool_name": tool_name,
        "tool_type": tool_type,
        "material": material,
        "process_step": process_step,
        "context_note": master_hint,
        "tool_context_note": " / ".join([value for value in [tool_name, tool_type, material, process_step] if value]),
        "audio_context": audio_context,
        "master_clip_filename": Path(str((master_metadata or {}).get("path", ""))).name,
        "master_clip_duration_seconds": (master_metadata or {}).get("duration_seconds", ""),
        "selected_master_frame_labels": [frame.get("label", "") for frame in selected_master_frames],
        "frame_observations": frame_observations,
        "frame_observation_summary": frame_observation_summary,
        "frame_delta_summary": frame_delta_summary,
        "frame_evidence_status": frame_status,
        "selected_key_moments": selected_key_moments,
        "motion_score_summary": motion_score_summary,
        "hand_detected": hand_evidence["hand_detected"],
        "hand_visible_ratio": hand_evidence["hand_visible_ratio"],
        "detected_hands": hand_evidence["detected_hands"],
        "hand_keyframes": hand_evidence["hand_keyframes"],
        "hand_evidence_status": hand_evidence["hand_evidence_status"],
        "hand_evidence": hand_evidence,
        "evidence_limits": [
            "Pressure is not directly measured.",
            "Exact mastery is not scored.",
            "Tool identity may be user-provided if not visually confirmed.",
        ],
        "visual_layer": visual_layer_data["visual_layer"],
        "visual_observation_summary": visual_observation_summary,
        "ghost_motion_overlay_present": bool(ghost_motion_overlay),
        "visual_evidence_status": visual_evidence_status,
        "visual_evidence_note": visual_evidence_note,
        "guidance_source": guidance_source,
        "capture_mode": capture_mode_display,
        "what_successors_should_notice": visual_layer_data["what_successors_should_notice"],
        "tool_or_material_cues": [base_memory.material_cue, base_memory.sound_cue],
        "practice_task": base_memory.practice_task,
        "evidence_from_master_frames": focus_notes,
        "watch_points": base_memory.watch_points,
        "timing_cue": base_memory.timing_cue,
        "motion_cue": base_memory.motion_cue,
        "material_cue": base_memory.material_cue,
        "sound_cue": base_memory.sound_cue,
        "common_mistake": base_memory.common_mistake,
        "evidence": base_memory.evidence,
        "comparison_notes": comparison_notes,
        "suggested_craft_name": visual_layer_data["suggested_craft_name"],
        "suggested_context_note": visual_layer_data["suggested_context_note"],
        "suggested_key_moments": visual_layer_data["suggested_key_moments"],
        "motion_cues": visual_layer_data["motion_cues"],
        "master_motion_template": visual_layer_data["master_motion_template"],
        "visual_confidence": visual_layer_data["visual_confidence"],
        "frame_observations": frame_observations,
        "frame_observation_summary": frame_observation_summary,
        "frame_delta_summary": frame_delta_summary,
        "frame_evidence_status": frame_status,
        "selected_master_keyframes": [
            {
                "index": frame.get("index"),
                "label": frame.get("label", ""),
                "filename": frame.get("filename", ""),
            }
            for frame in selected_master_frames
        ],
        "model_mode": "Liquid LFM",
    }
    try:
        liquid_memory, liquid_debug = generate_practice_memory(observation_data)
        memory = _fill_missing_fields(PracticeMemory.from_dict(liquid_memory), base_memory, craft, master_hint)
        memory.model_mode = "Liquid LFM"
        model_mode_display = "Liquid LFM"
        model_notice = ""
        debug_snippet = liquid_debug.get("liquid_raw_snippet", "")[:1000]
        summary_mode = "liquid lfm"
        capture_notes = _comparison_notes(master_frames, apprentice_frames, craft) if has_practice_clip else _master_only_notes(master_frames, craft, audio_context)
    except LiquidResponseError as exc:
        reason = "unreachable" if not liquid_server_reachable(liquid_config()[0]) else "parse_error"
        memory = base_memory
        model_mode_display = mode_display_label("liquid lfm", fallback=True)
        model_notice = model_notice_for_failure(reason)
        memory.model_mode = model_mode_display
        memory.evidence.append("Liquid LFM attempt fell back to Rule mode.")
        liquid_debug = {
            "liquid_request_url": exc.request_url or f"{liquid_config()[0]}/chat/completions",
            "liquid_model": exc.model or liquid_config()[1],
            "liquid_http_status": exc.http_status,
            "liquid_raw_snippet": (exc.raw_snippet or exc.snippet or "")[:1000],
            "liquid_error": str(exc),
            "parsed_json_success": bool(exc.parsed_json_success),
        }
        debug_snippet = liquid_debug["liquid_raw_snippet"][:1000]
        summary_mode = "rule"
        capture_notes = _comparison_notes(master_frames, apprentice_frames, craft) if has_practice_clip else _master_only_notes(master_frames, craft, audio_context)
    except Exception:
        reason = "unreachable" if not liquid_server_reachable(liquid_config()[0]) else "parse_error"
        memory = base_memory
        model_mode_display = mode_display_label("liquid lfm", fallback=True)
        model_notice = model_notice_for_failure(reason)
        memory.model_mode = model_mode_display
        memory.evidence.append("Liquid LFM attempt fell back to Rule mode.")
        liquid_debug = {
            "liquid_request_url": f"{liquid_config()[0]}/chat/completions",
            "liquid_model": liquid_config()[1],
            "liquid_http_status": None,
            "liquid_raw_snippet": "",
            "liquid_error": "unexpected Liquid LFM exception",
            "parsed_json_success": False,
        }
        debug_snippet = ""
        summary_mode = "rule"
        capture_notes = _comparison_notes(master_frames, apprentice_frames, craft) if has_practice_clip else _master_only_notes(master_frames, craft, audio_context)
    else:
        memory = _pottery_context_memory(craft, master_hint, "mock", "Frame-confirmed guidance") if pottery_context else mock_analysis(craft, master_hint)
        model_mode_display = "Mock"
        model_notice = ""
        debug_snippet = ""
        liquid_debug = {
            "liquid_request_url": "",
            "liquid_model": "",
            "liquid_http_status": None,
            "liquid_raw_snippet": "",
            "liquid_error": "",
            "parsed_json_success": False,
        }
        summary_mode = "mock"
        capture_notes = _comparison_notes(master_frames, apprentice_frames, craft) if has_practice_clip else _master_only_notes(master_frames, craft, audio_context)
        visual_layer_data = base_visual_layer

    visual_evidence_status, visual_evidence_note = _visual_evidence_assessment(visual_layer_data, liquid_vision_debug, frame_status, frame_delta_summary)
    memory = _apply_visual_evidence_guard(memory, visual_evidence_status, frame_observation_summary, context_guided=pottery_context)
    memory = _apply_actionable_guidance(memory, visual_evidence_status, frame_observations, frame_observation_summary)
    if pottery_context and visual_evidence_status != "confirmed_craft":
        context_memory = _pottery_context_memory(craft, master_hint, memory.model_mode or model_mode_display, "Context-guided fallback after Liquid LFM attempt")
        memory = _fill_missing_fields(context_memory, memory, craft, master_hint)
        memory.model_mode = memory.model_mode or model_mode_display
        memory.evidence = [
            "Context-guided fallback after Liquid LFM attempt",
            f"Selected frames were reviewed locally: {frame_observation_summary or 'Selected frames were reviewed locally.'}",
            "User-provided craft context supported pottery-style practice guidance.",
        ]
    memory = _hand_guided_practice(memory, hand_evidence)
    memory.guidance_source = guidance_source

    summary = _difference_summary(memory, summary_mode, visual_evidence_status, frame_observation_summary)
    if has_practice_clip:
        summary = f"{summary} Key frames reviewed locally: {len(master_frames)} master / {len(apprentice_frames)} practice."
    else:
        summary = f"{summary} Master-only capture reviewed locally: {len(master_frames)} master frames."

    return {
        "practice_memory": memory.to_dict(),
        "visible_difference_summary": summary,
        "model_mode": memory.model_mode,
        "model_mode_display": model_mode_display,
        "model_notice": model_notice,
        "liquid_debug_snippet": debug_snippet,
        "liquid_debug": liquid_debug,
        "liquid_vision_debug": liquid_vision_debug,
        "visual_layer": visual_layer_data["visual_layer"],
        "suggested_craft_name": visual_layer_data["suggested_craft_name"],
        "suggested_context_note": visual_layer_data["suggested_context_note"],
        "suggested_key_moments": visual_layer_data["suggested_key_moments"],
        "what_successors_should_notice": visual_layer_data["what_successors_should_notice"],
        "motion_cues": visual_layer_data["motion_cues"],
        "master_motion_template": visual_layer_data["master_motion_template"],
        "visual_confidence": visual_layer_data["visual_confidence"],
        "frame_observations": frame_observations,
        "frame_observation_summary": frame_observation_summary,
        "frame_delta_summary": frame_delta_summary,
        "frame_evidence_status": frame_status,
        "selected_key_moments": selected_key_moments,
        "motion_score_summary": motion_score_summary,
        "hand_detected": hand_evidence["hand_detected"],
        "hand_visible_ratio": hand_evidence["hand_visible_ratio"],
        "detected_hands": hand_evidence["detected_hands"],
        "hand_keyframes": hand_evidence["hand_keyframes"],
        "hand_evidence_status": hand_evidence["hand_evidence_status"],
        "hand_evidence": hand_evidence,
        "evidence_limits": [
            "Pressure is not directly measured.",
            "Exact mastery is not scored.",
            "Tool identity may be user-provided if not visually confirmed.",
        ],
        "visual_evidence_status": visual_evidence_status,
        "visual_evidence_note": visual_evidence_note,
        "visual_observation_summary": visual_observation_summary,
        "guidance_source": guidance_source,
        "capture_mode_display": capture_mode_display,
        "capture_notes": capture_notes,
    }
