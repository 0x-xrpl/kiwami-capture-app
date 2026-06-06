from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
UPLOADS_DIR = BASE_DIR / "uploads"


def _session_dir(session_id: str) -> Path:
    return OUTPUTS_DIR / session_id


def _session_path(session_id: str) -> Path:
    return _session_dir(session_id) / "session.json"


def _ensure_dirs(session_id: str) -> None:
    (_session_dir(session_id)).mkdir(parents=True, exist_ok=True)
    (UPLOADS_DIR / session_id).mkdir(parents=True, exist_ok=True)
    (_session_dir(session_id) / "frames" / "master").mkdir(parents=True, exist_ok=True)
    (_session_dir(session_id) / "frames" / "apprentice").mkdir(parents=True, exist_ok=True)
    (_session_dir(session_id) / "frames" / "practice").mkdir(parents=True, exist_ok=True)
    (_session_dir(session_id) / "exports").mkdir(parents=True, exist_ok=True)


def create_session() -> str:
    session_id = uuid.uuid4().hex[:12]
    _ensure_dirs(session_id)
    payload = {
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "created",
    }
    save_session_data(session_id, payload)
    return session_id


def save_session_data(session_id: str, data: dict[str, Any]) -> None:
    _ensure_dirs(session_id)
    path = _session_path(session_id)
    temp_path = path.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)


def load_session_data(session_id: str) -> dict[str, Any]:
    path = _session_path(session_id)
    if not path.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def update_practice_memory(session_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    data = load_session_data(session_id)
    practice_memory = data.get("practice_memory", {})
    practice_memory.update(updates)
    data["practice_memory"] = practice_memory
    data["status"] = "updated"
    save_session_data(session_id, data)
    return data


def list_session_summaries(limit: int = 12) -> list[dict[str, Any]]:
    if not OUTPUTS_DIR.exists():
        return []
    sessions: list[dict[str, Any]] = []
    for child in OUTPUTS_DIR.iterdir():
        if not child.is_dir():
            continue
        path = child / "session.json"
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            sessions.append(
                {
                    "session_id": data.get("session_id", child.name),
                    "created_at": data.get("created_at", ""),
                    "craft": data.get("craft", ""),
                    "mode": data.get("mode", ""),
                    "status": data.get("status", ""),
                }
            )
        except Exception:
            continue
    sessions.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return sessions[:limit]
