from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import requests

from ..model_status import liquid_config, liquid_timeout_seconds


JSON_FIELDS = [
    "craft",
    "skill_focus",
    "watch_points",
    "step_by_step_motion",
    "what_to_copy",
    "what_to_avoid",
    "success_sign",
    "next_practice_drill",
    "guidance_source",
    "timing_cue",
    "motion_cue",
    "material_cue",
    "sound_cue",
    "common_mistake",
    "master_hint",
    "practice_task",
    "evidence",
    "privacy_mode",
    "model_mode",
    "shareable",
]


@dataclass
class LiquidResponseError(Exception):
    message: str
    snippet: str = ""
    request_url: str = ""
    model: str = ""
    http_status: int | None = None
    raw_response: str = ""
    raw_snippet: str = ""
    parsed_json_success: bool = False

    def __str__(self) -> str:
        return self.message


def _normalize_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_json_object(text: str) -> str:
    text = _normalize_text(text)
    if not text:
        raise LiquidResponseError("Empty Liquid response.")
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)

    raise LiquidResponseError("No valid JSON object found in Liquid response.", snippet=text[:400])


def _extract_json_text(text: str) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        raise LiquidResponseError("Empty Liquid response.")
    try:
        json.loads(normalized)
        return normalized
    except json.JSONDecodeError:
        pass
    return _extract_json_object(normalized)


def _content_from_payload(payload: dict[str, Any]) -> str:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = choice.get("text")
    if isinstance(content, str) and content.strip():
        return content
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    if content is None:
        direct = payload.get("content")
        if isinstance(direct, str):
            return direct
        if isinstance(direct, dict):
            return json.dumps(direct, ensure_ascii=False)
        if isinstance(payload, str):
            return payload
        return ""
    return str(content)


def _extract_content(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return _content_from_payload(payload)
        if isinstance(payload, str):
            return payload
    except json.JSONDecodeError:
        return text
    return text


def _response_content_from_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        return _content_from_payload(payload)
    if isinstance(payload, str):
        return _extract_content(payload)
    return str(payload or "").strip()


def _normalize_liquid_content(content: str, observation_data: dict[str, Any]) -> tuple[dict[str, Any], str]:
    try:
        data = json.loads(_extract_json_text(content))
    except LiquidResponseError:
        if str(content).strip():
            return _derive_text_guidance(content, observation_data), "text"
        raise
    except json.JSONDecodeError:
        if str(content).strip():
            return _derive_text_guidance(content, observation_data), "text"
        raise
    if not isinstance(data, dict):
        raise LiquidResponseError("Liquid response was not a JSON object.", snippet=str(content)[:400])
    return data, "json"


def _split_lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _summarize_text(text: str, limit: int = 180) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def _pick_instruction_lines(text: str) -> list[str]:
    lines = _split_lines(text)
    instruction_lines: list[str] = []
    for line in lines:
        normalized = line.lstrip("-•*0123456789. )(").strip()
        if re.match(r"^(step|first|then|next|finally|copy|avoid|watch|look|repeat|keep|use|focus)\b", normalized.lower()):
            instruction_lines.append(normalized)
        elif any(token in normalized.lower() for token in ("do ", "don't", "avoid", "copy", "watch", "keep", "repeat")):
            instruction_lines.append(normalized)
    return instruction_lines


def _derive_text_guidance(text: str, observation_data: dict[str, Any]) -> dict[str, Any]:
    raw_text = (text or "").strip()
    summary = _summarize_text(raw_text)
    instruction_lines = _pick_instruction_lines(raw_text)
    evidence_lines = instruction_lines[:3] or ([summary] if summary else [])
    base_skill_focus = str(observation_data.get("skill_focus") or observation_data.get("craft") or "").strip()
    base_watch_points = observation_data.get("watch_points")
    if not isinstance(base_watch_points, list):
        base_watch_points = []
    base_step = str(observation_data.get("step_by_step_motion") or "").strip()
    base_copy = str(observation_data.get("what_to_copy") or "").strip()
    base_avoid = str(observation_data.get("what_to_avoid") or "").strip()
    base_success = str(observation_data.get("success_sign") or "").strip()
    base_drill = str(observation_data.get("next_practice_drill") or "").strip()
    if instruction_lines:
        first_instruction = instruction_lines[0]
        if not base_skill_focus:
            base_skill_focus = _summarize_text(first_instruction, 90)
        if not base_step:
            base_step = "\n".join(f"{index + 1}. {line}" for index, line in enumerate(instruction_lines[:4]))
        if not base_copy:
            base_copy = _summarize_text(" ".join(instruction_lines[:2]), 180)
        if not base_avoid:
            base_avoid = "Avoid adding unsupported assumptions beyond the selected frames and local context."
        if not base_success:
            base_success = "The motion stays consistent with the selected frames and the guidance reads clearly."
        if not base_drill:
            base_drill = "Repeat the motion slowly using the local Liquid guidance and selected frame context."
    return {
        "craft": str(observation_data.get("craft", "")).strip(),
        "skill_focus": base_skill_focus or _summarize_text(raw_text, 90),
        "watch_points": base_watch_points or ([summary] if summary else []),
        "step_by_step_motion": base_step or raw_text,
        "what_to_copy": base_copy or summary or raw_text,
        "what_to_avoid": base_avoid or "Avoid adding unsupported assumptions beyond the selected frames.",
        "success_sign": base_success or "The motion stays consistent with the selected frames.",
        "next_practice_drill": base_drill or "Repeat the motion slowly using the local Liquid guidance.",
        "guidance_source": "Liquid LFM text guidance",
        "timing_cue": str(observation_data.get("timing_cue", "")).strip(),
        "motion_cue": str(observation_data.get("motion_cue", "")).strip() or summary,
        "material_cue": str(observation_data.get("material_cue", "")).strip(),
        "sound_cue": str(observation_data.get("sound_cue", "")).strip(),
        "common_mistake": str(observation_data.get("common_mistake", "")).strip()
        or "Avoid adding unsupported assumptions beyond the selected frames.",
        "master_hint": str(observation_data.get("master_hint", "")).strip(),
        "practice_task": raw_text if raw_text else str(observation_data.get("practice_task", "")).strip(),
        "evidence": evidence_lines,
        "privacy_mode": str(observation_data.get("privacy_mode", "local_only")).strip() or "local_only",
        "model_mode": "Liquid LFM",
        "shareable": bool(observation_data.get("shareable", True)),
    }


def _prompt(observation_data: dict[str, Any]) -> list[dict[str, str]]:
    schema = {
                "craft": "string",
                "skill_focus": "string",
                "watch_points": ["string"],
                "step_by_step_motion": "string",
                "what_to_copy": "string",
                "what_to_avoid": "string",
                "success_sign": "string",
                "next_practice_drill": "string",
                "guidance_source": "string",
                "timing_cue": "string",
                "motion_cue": "string",
                "material_cue": "string",
                "sound_cue": "string",
        "common_mistake": "string",
        "master_hint": "string",
        "practice_task": "string",
        "evidence": ["string"],
        "privacy_mode": "local_only",
        "model_mode": "Liquid LFM",
        "shareable": True,
    }
    return [
        {
            "role": "system",
            "content": (
                "Return only one valid JSON object.\n"
                "Do not use Markdown.\n"
                "Do not wrap the output in ```json or ```.\n"
                "Do not add explanation.\n"
                "Use double quotes for all keys and string values.\n"
                "Use arrays where required.\n"
                "Use true or false for booleans.\n"
                "Frame observations are the source of truth.\n"
                "Treat Craft Name, Master Hint, Tool Name, Tool Type, Material, and Process Step as user-provided context only.\n"
                "Base Practice Memory on the selected frame observations first.\n"
                "Convert the visible frame evidence into practice guidance. Do not stop at description.\n"
                "Produce what the learner should copy, avoid, check, and repeat.\n"
                "If the craft context is pottery-like and the visible evidence does not contradict it, write lesson-style guidance for wheel, clay, rim, wall, and hand contact.\n"
                "Make motion_cue and material_cue specific to the selected master frames. Avoid craft-specific wording unless directly supported. If uncertain, stay honest.\n"
                "Include all required Practice Memory fields."
            ),
        },
        {
            "role": "user",
            "content": (
                "Create Practice Memory JSON from these observations.\n"
                f"Required fields: {', '.join(JSON_FIELDS)}\n"
                f"Schema example: {json.dumps(schema, ensure_ascii=False)}\n"
                f"Observations: {json.dumps(observation_data, ensure_ascii=False)}"
            ),
        },
    ]


def _build_request_body(observation_data: dict[str, Any], model: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": _prompt(observation_data),
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }


def _build_native_prompt(observation_data: dict[str, Any]) -> str:
    return (
        "Return a short Practice Memory for pottery centering with: skill_focus, what_to_notice, "
        "what_to_copy, what_to_avoid, success_sign, next_practice_drill."
    )


def _build_native_body(observation_data: dict[str, Any], n_predict: int = 160) -> dict[str, Any]:
    return {
        "prompt": _build_native_prompt(observation_data),
        "temperature": 0.2,
        "n_predict": max(16, min(160, int(n_predict or 160))),
    }


def _native_completion_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return f"{normalized}/completion"


def _request_json(request_url: str, payload: dict[str, Any], timeout: float) -> requests.Response:
    return requests.post(request_url, json=payload, timeout=timeout)


def _extract_completion_content(payload: Any) -> str:
    if isinstance(payload, dict):
        choice = (payload.get("choices") or [{}])[0]
        text = choice.get("text")
        if isinstance(text, str) and text.strip():
            return text
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)
        content = payload.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)
    if isinstance(payload, str):
        return payload
    return ""


def generate_practice_memory(
    observation_data: dict[str, Any],
    timeout: float | None = None,
    max_tokens: int = 700,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_url, model = liquid_config()
    timeout = liquid_timeout_seconds() if timeout is None else timeout
    chat_request_url = f"{base_url}/chat/completions"
    native_request_url = _native_completion_url(base_url)
    debug: dict[str, Any] = {
        "liquid_request_url": chat_request_url,
        "liquid_model": model,
        "liquid_http_status": None,
        "liquid_raw_response": "",
        "liquid_raw_snippet": "",
        "liquid_error": "",
        "parsed_json_success": False,
        "liquid_response_mode": "unknown",
    }
    chat_payload = _build_request_body(observation_data, model, max_tokens)
    native_payload = _build_native_body(observation_data, max_tokens)
    chat_timeout = timeout if timeout > 25 else 0.5
    native_timeout = timeout if timeout > 25 else max(2.0, timeout - chat_timeout)

    def _attempt(
        request_url: str,
        payload: dict[str, Any],
        *,
        allow_text: bool,
    ) -> dict[str, Any]:
        try:
            request_timeout = chat_timeout if request_url == chat_request_url else native_timeout
            response = _request_json(request_url, payload, request_timeout)
            debug["liquid_request_url"] = request_url
            debug["liquid_http_status"] = response.status_code
            raw_text = response.text or ""
            debug["liquid_raw_response"] = raw_text
            debug["liquid_raw_snippet"] = raw_text[:1000]
            response.raise_for_status()
            try:
                response_payload = response.json()
            except ValueError:
                response_payload = raw_text
            content = _response_content_from_payload(response_payload)
            if not content.strip():
                raise LiquidResponseError(
                    "Liquid response was empty after normalization.",
                    snippet=raw_text[:400],
                    request_url=request_url,
                    model=model,
                    http_status=response.status_code,
                    raw_response=raw_text,
                    raw_snippet=raw_text[:1000],
                    parsed_json_success=False,
                )
            try:
                data, response_mode = _normalize_liquid_content(content, observation_data)
            except LiquidResponseError as exc:
                if allow_text and str(content).strip():
                    data = _derive_text_guidance(content, observation_data)
                    response_mode = "text"
                else:
                    raise LiquidResponseError(
                        str(exc),
                        snippet=(exc.snippet or str(content)[:400]),
                        request_url=request_url,
                        model=model,
                        http_status=response.status_code,
                        raw_response=raw_text,
                        raw_snippet=raw_text[:1000],
                        parsed_json_success=False,
                    ) from exc
            debug["parsed_json_success"] = response_mode == "json"
            debug["liquid_response_mode"] = response_mode
            return data
        except requests.RequestException as exc:
            raise LiquidResponseError(
                str(exc),
                snippet=debug["liquid_raw_snippet"][:400],
                request_url=request_url,
                model=model,
                http_status=debug["liquid_http_status"],
                raw_response=debug["liquid_raw_response"],
                raw_snippet=debug["liquid_raw_response"][:1000] or debug["liquid_raw_snippet"],
                parsed_json_success=False,
            ) from exc

    try:
        try:
            data = _attempt(chat_request_url, chat_payload, allow_text=True)
            return data, debug
        except LiquidResponseError:
            data = _attempt(native_request_url, native_payload, allow_text=True)
            return data, debug
    except LiquidResponseError:
        raise
    except Exception as exc:
        raise LiquidResponseError(
            str(exc),
            snippet=debug["liquid_raw_snippet"][:400],
            request_url=debug["liquid_request_url"],
            model=model,
            http_status=debug["liquid_http_status"],
            raw_response=debug["liquid_raw_response"],
            raw_snippet=debug["liquid_raw_snippet"],
            parsed_json_success=False,
        ) from exc
