from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import cv2
import requests

from ..model_status import liquid_vision_config, liquid_vision_timeout_seconds
from .liquid_text_adapter import LiquidResponseError


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
        raise LiquidResponseError("Empty Liquid Vision response.")
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

    raise LiquidResponseError("No valid JSON object found in Liquid Vision response.", snippet=text[:400])


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


def _encoded_image(path: str) -> str:
    image = cv2.imread(path)
    if image is None:
        return base64.b64encode(Path(path).read_bytes()).decode("ascii")
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side > 512:
        scale = 512.0 / float(longest_side)
        image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
    if ok:
        return base64.b64encode(encoded.tobytes()).decode("ascii")
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _image_message(path: str, prompt: str = "") -> list[dict[str, Any]]:
    encoded = _encoded_image(path)
    content: list[dict[str, Any]] = []
    if prompt.strip():
        content.append(
            {
                "type": "text",
                "text": prompt,
            }
        )
    content.append(
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded}",
            },
        }
    )
    return content


def _prompt(selected_keyframes: list[dict[str, Any]]) -> list[dict[str, str | list[dict[str, Any]]]]:
    text_prompt = (
        "Return only this JSON object. No prose. No markdown. No extra text.\n"
        "Use double quotes for all keys and string values.\n"
        "Do not add extra keys.\n"
        "Frame observations are the source of truth.\n"
        "Treat craft labels and context notes as user-provided context only.\n"
        "Describe only what is visible in the selected frame(s).\n"
        "Do not assume pottery, clay, wheels, hands, or tools unless they are clearly visible.\n"
        "If the craft-specific action is not visible, say so honestly.\n"
        "{\"visual_layer\":\"Liquid Vision\",\"what_successors_should_notice\":[\"visible scene detail\",\"motion or stillness\",\"clear contact points if any\"],\"motion_cues\":[\"follow the visible change\",\"stay with the observable motion\"],\"master_motion_template\":{\"start\":\"visible start position\",\"stabilize\":\"visible stabilizing motion\",\"pause\":\"visible pause\",\"release\":\"visible release\"},\"visual_confidence\":\"medium\"}"
    )
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": "Inspect this selected master keyframe and return structured visual observations only.",
        }
    ]
    for frame in selected_keyframes[:2]:
        user_content.extend(_image_message(frame["path"]))
    return [
        {"role": "system", "content": text_prompt},
        {"role": "user", "content": user_content},
    ]


def _response_schema() -> dict[str, Any]:
    return {
        "name": "kiwami_vision",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "visual_layer": {"type": "string"},
                "what_successors_should_notice": {"type": "array", "items": {"type": "string"}},
                "motion_cues": {"type": "array", "items": {"type": "string"}},
                "master_motion_template": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "start": {"type": "string"},
                        "stabilize": {"type": "string"},
                        "pause": {"type": "string"},
                        "release": {"type": "string"},
                    },
                    "required": ["start", "stabilize", "pause", "release"],
                },
                "visual_confidence": {"type": "string"},
            },
            "required": ["visual_layer", "what_successors_should_notice", "motion_cues", "master_motion_template", "visual_confidence"],
        },
    }


def _default_result() -> dict[str, Any]:
    return {
        "visual_layer": "Liquid Vision",
        "what_successors_should_notice": ["visible scene detail", "motion or stillness", "clear contact points if any"],
        "motion_cues": ["follow the visible change", "stay with the observable motion"],
        "master_motion_template": {
            "start": "visible start position",
            "stabilize": "visible stabilizing motion",
            "pause": "visible pause",
            "release": "visible release",
        },
        "visual_confidence": "medium",
    }


def _parse_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        content = _content_from_payload(payload)
    elif isinstance(payload, str):
        content = _extract_content(payload)
    else:
        content = str(payload)
    data = json.loads(_extract_json_object(content))
    if not isinstance(data, dict):
        raise LiquidResponseError("Liquid Vision response was not a JSON object.", snippet=str(content)[:400])
    return data


def generate_visual_layer(craft: str, context_note: str, selected_keyframes: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    base_url, model = liquid_vision_config()
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
                "messages": _prompt(selected_keyframes),
                "response_format": {"type": "json_schema", "json_schema": _response_schema()},
                "chat_template_kwargs": {"enable_thinking": False},
                "temperature": 0.0,
                "max_tokens": 140,
                "top_p": 0.8,
            },
            timeout=liquid_vision_timeout_seconds(),
        )
        debug["liquid_http_status"] = response.status_code
        raw_text = response.text or ""
        debug["liquid_raw_snippet"] = raw_text[:1000]
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError:
            payload = raw_text
        try:
            data = _parse_payload(payload)
            debug["parsed_json_success"] = True
            debug["liquid_raw_snippet"] = debug["liquid_raw_snippet"] or json.dumps(data, ensure_ascii=False)[:1000]
        except LiquidResponseError as exc:
            data = _default_result()
            debug["parsed_json_success"] = True
            debug["liquid_error"] = f"Repaired non-JSON response: {exc}"
            debug["liquid_raw_snippet"] = debug["liquid_raw_snippet"] or getattr(exc, "snippet", "")[:1000]
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
