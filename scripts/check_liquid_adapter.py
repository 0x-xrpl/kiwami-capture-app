from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adapters.liquid_text_adapter import generate_practice_memory  # noqa: E402
from src.model_status import liquid_server_reachable  # noqa: E402


def _base_url_from_chat_url(chat_url: str) -> str:
    parsed = urlparse(chat_url.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Expected a full Liquid chat completions URL.")
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]
    base_path = path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{base_path}"


def _prompt() -> dict[str, object]:
    return {
        "craft": "Pottery",
        "master_hint": "Watch steady hand contact and slow release.",
        "tool_name": "Hand",
        "tool_type": "Contact",
        "material": "Clay",
        "process_step": "Centering",
        "skill_focus": "Keep the clay centered with light, steady hand contact.",
        "watch_points": ["hand pressure", "center wobble", "water amount"],
        "step_by_step_motion": "1. Set both hands lightly.\n2. Keep pressure even.\n3. Release slowly.",
        "what_to_copy": "Copy the steady contact and controlled release.",
        "what_to_avoid": "Avoid sudden one-sided pressure.",
        "success_sign": "The clay stays centered and stable.",
        "next_practice_drill": "Repeat a short centering drill at slow speed.",
        "timing_cue": "when both hands settle into contact",
        "motion_cue": "keep the hands close and even",
        "material_cue": "watch the clay stay round",
        "sound_cue": "steady wheel sound",
        "common_mistake": "pressing too hard too early",
        "practice_task": "Repeat a 20-second centering drill.",
        "privacy_mode": "local_only",
        "model_mode": "Liquid LFM",
        "shareable": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether the Liquid adapter returns usable guidance.")
    parser.add_argument("--url", default="", help="Optional Liquid chat completions URL.")
    args = parser.parse_args()

    base_url = None
    if args.url:
        base_url = _base_url_from_chat_url(args.url)
        os.environ["LIQUID_LFM_BASE_URL"] = base_url

    if base_url is None:
        from src.model_status import liquid_config  # noqa: E402

        base_url = liquid_config()[0]

    server_ok = liquid_server_reachable(base_url)
    print(f"server_reachable: {'yes' if server_ok else 'no'}")
    if not server_ok:
        print("liquid_response_received: no")
        print("parse_mode: fallback")
        print("guidance_source: Context-guided fallback")
        print("normalized_output: ")
        return 1

    try:
        memory, debug = generate_practice_memory(_prompt(), timeout=20, max_tokens=8)
    except Exception as exc:
        print("liquid_response_received: no")
        print("parse_mode: error")
        print("guidance_source: Context-guided fallback")
        print(f"normalized_output: {str(exc)[:500]}")
        return 1

    response_mode = str(debug.get("liquid_response_mode") or "").strip()
    received = bool(str(debug.get("liquid_raw_response") or "").strip())
    parse_mode = "error"
    if response_mode == "json" and debug.get("parsed_json_success"):
        parse_mode = "structured_json"
    elif response_mode == "text":
        parse_mode = "text_guidance"
    elif received:
        parse_mode = "text_guidance"
    elif not received:
        parse_mode = "fallback"

    guidance_source = str(memory.get("guidance_source") or "").strip() or "unknown"
    normalized_output = str(memory.get("step_by_step_motion") or memory.get("practice_task") or memory.get("skill_focus") or "")[:500]

    print(f"liquid_response_received: {'yes' if received else 'no'}")
    print(f"parse_mode: {parse_mode}")
    print(f"guidance_source: {guidance_source}")
    print(f"normalized_output: {normalized_output}")

    if parse_mode in {"structured_json", "text_guidance"} and received:
        return 0
    if not received:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
