from __future__ import annotations

import json
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
    raw_snippet: str = ""
    parsed_json_success: bool = False

    def __str__(self) -> str:
        return self.message


def _normalize_text(text: str) -> str:
    text = (text or "").strip()
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


def generate_practice_memory(observation_data: dict[str, Any], timeout: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    base_url, model = liquid_config()
    timeout = liquid_timeout_seconds() if timeout is None else timeout
    request_url = f"{base_url}/chat/completions"
    debug: dict[str, Any] = {
        "liquid_request_url": request_url,
        "liquid_model": model,
        "liquid_http_status": None,
        "liquid_raw_snippet": "",
        "liquid_error": "",
        "parsed_json_success": False,
    }
    try:
        response = requests.post(
            request_url,
            json={
                "model": model,
                "messages": _prompt(observation_data),
                "temperature": 0.2,
                "max_tokens": 700,
            },
            timeout=timeout,
        )
        debug["liquid_http_status"] = response.status_code
        raw_text = response.text or ""
        debug["liquid_raw_snippet"] = raw_text[:1000]
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError:
            payload = raw_text
        if isinstance(payload, dict):
            content = _content_from_payload(payload)
        elif isinstance(payload, str):
            content = _extract_content(payload)
        else:
            content = str(payload)
        data = json.loads(_extract_json_object(content))
        if not isinstance(data, dict):
            raise LiquidResponseError(
                "Liquid response was not a JSON object.",
                snippet=str(content)[:400],
                request_url=request_url,
                model=model,
                http_status=response.status_code,
                raw_snippet=raw_text[:1000],
                parsed_json_success=False,
            )
        debug["parsed_json_success"] = True
        debug["liquid_raw_snippet"] = debug["liquid_raw_snippet"] or str(content)[:1000]
        return data, debug
    except LiquidResponseError:
        raise
    except requests.RequestException as exc:
        raise LiquidResponseError(
            str(exc),
            snippet=debug["liquid_raw_snippet"][:400],
            request_url=request_url,
            model=model,
            http_status=debug["liquid_http_status"],
            raw_snippet=debug["liquid_raw_snippet"],
            parsed_json_success=False,
        ) from exc
    except Exception as exc:
        raise LiquidResponseError(
            str(exc),
            snippet=debug["liquid_raw_snippet"][:400],
            request_url=request_url,
            model=model,
            http_status=debug["liquid_http_status"],
            raw_snippet=debug["liquid_raw_snippet"],
            parsed_json_success=False,
        ) from exc
