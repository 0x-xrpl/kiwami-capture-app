from __future__ import annotations

import os

import requests


DEFAULT_LIQUID_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LIQUID_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_K_M"
DEFAULT_LIQUID_TIMEOUT_SECONDS = 60.0
DEFAULT_LIQUID_VISION_BASE_URL = "http://127.0.0.1:8081/v1"
DEFAULT_LIQUID_VISION_MODEL = "LFM2.5-VL-1.6B-BF16.gguf"
DEFAULT_LIQUID_VISION_TIMEOUT_SECONDS = 180.0


def liquid_config() -> tuple[str, str]:
    base_url = (os.getenv("LIQUID_LFM_BASE_URL") or DEFAULT_LIQUID_BASE_URL).rstrip("/")
    model = os.getenv("LIQUID_LFM_MODEL") or DEFAULT_LIQUID_MODEL
    return base_url, model


def liquid_vision_config() -> tuple[str, str]:
    base_url = (os.getenv("LIQUID_VISION_BASE_URL") or DEFAULT_LIQUID_VISION_BASE_URL).rstrip("/")
    model = os.getenv("LIQUID_VISION_MODEL") or DEFAULT_LIQUID_VISION_MODEL
    return base_url, model


def liquid_timeout_seconds() -> float:
    raw = os.getenv("LIQUID_LFM_TIMEOUT_SECONDS")
    if not raw:
        return DEFAULT_LIQUID_TIMEOUT_SECONDS
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_LIQUID_TIMEOUT_SECONDS


def liquid_vision_timeout_seconds() -> float:
    raw = os.getenv("LIQUID_VISION_TIMEOUT_SECONDS")
    if not raw:
        return DEFAULT_LIQUID_VISION_TIMEOUT_SECONDS
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_LIQUID_VISION_TIMEOUT_SECONDS


def liquid_server_reachable(base_url: str | None = None, timeout: float = 2.0) -> bool:
    base_url = (base_url or liquid_config()[0]).rstrip("/")
    try:
        response = requests.get(f"{base_url}/models", timeout=timeout)
        return response.ok
    except requests.RequestException:
        return False


def normalize_mode(mode: str) -> str:
    value = (mode or "").strip().lower().replace("_", " ")
    if value in {"liquid lfm", "liquid text", "local liquid lfm", "liquid"}:
        return "liquid lfm"
    if value == "rule":
        return "rule"
    return "mock"


def mode_display_label(mode: str, fallback: bool = False) -> str:
    normalized = normalize_mode(mode)
    if normalized == "liquid lfm":
        return "Rule fallback after Liquid LFM attempt" if fallback else "Liquid LFM"
    return "Rule" if normalized == "rule" else "Mock"


def model_notice_for_failure(reason: str | None) -> str:
    if reason == "unreachable":
        return "Local Liquid LFM server is not reachable. Rule fallback was used."
    if reason == "parse_error":
        return "Liquid LFM response could not be parsed. Rule fallback was used."
    return ""
