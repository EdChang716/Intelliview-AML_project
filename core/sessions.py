# core/sessions.py
from typing import Optional, List, Dict, Any, Set
import json
import datetime
from pathlib import Path

from .config import SESSIONS_DIR

def _session_path(profile_id: str) -> Path:
    return SESSIONS_DIR / f"{profile_id}.json"


def load_session(profile_id: str) -> dict:
    path = _session_path(profile_id)
    if not path.exists():
        return {"profile_id": profile_id, "turns": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_session(profile_id: str, data: dict):
    path = _session_path(profile_id)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_asked_questions(
    profile_id: str,
    mode: Optional[str] = None,
    behavioral_type: Optional[str] = None,
    entry_key: Optional[str] = None,
) -> Set[str]:
    """
    從 session 裡抓出「使用者有回答並存檔過的主題目」，用來避免重複。
    追問 (is_followup=True) 不會被拿來避免重複主題問題。
    """
    data = load_session(profile_id)
    questions: set[str] = set()

    for t in data.get("turns", []):
        if t.get("is_followup"):
            continue

        if mode and t.get("mode") != mode:
            continue
        if behavioral_type and t.get("behavioral_type") != behavioral_type:
            continue
        if entry_key and t.get("entry_key") != entry_key:
            continue

        q = t.get("question")
        if q:
            questions.add(q.strip())

    return questions


def log_practice_turn(
    profile_id: str,
    question: str,
    sample_answer: Optional[str],
    bullets: List[Dict[str, Any]],
    mode: str,
    behavioral_type: Optional[str] = None,
    entry_key: Optional[str] = None,
    user_answer: Optional[str] = None,
    score: Optional[int] = None,
    strengths: Optional[str] = None,
    improvements: Optional[str] = None,
    thread_id: Optional[str] = None,
    is_followup: bool = False,
    # 🔥 新增
    media_type: Optional[str] = None,       # "audio" / "video"
    media_filename: Optional[str] = None,   # 相對路徑或檔名
    media_duration_ms: Optional[int] = None,
):
    data = load_session(profile_id)
    turns = data.get("turns", [])

    now = datetime.datetime.utcnow().isoformat() + "Z"

    # 給舊資料 / 其他地方使用的巢狀結構
    media_obj: Optional[dict] = None
    if media_type or media_filename or media_duration_ms is not None:
        media_obj = {
            "type": media_type,
            "filename": media_filename,
            "duration_ms": media_duration_ms,
        }

    turns.append(
        {
            "timestamp": now,
            "mode": mode,
            "behavioral_type": behavioral_type,
            "entry_key": entry_key,
            "question": question,
            "user_answer": user_answer,
            "sample_answer": sample_answer,
            "bullets": bullets,
            "score": score,
            "strengths": strengths,
            "improvements": improvements,
            "thread_id": thread_id,
            "is_followup": is_followup,
            # ⭐ 新增平鋪欄位，給 history.html 用
            "media_type": media_type,
            "media_filename": media_filename,
            "media_duration_ms": media_duration_ms,
            # ⭐ 同時保留原本的巢狀 media 結構
            "media": media_obj,
        }
    )

    data["turns"] = turns
    save_session(profile_id, data)


def log_asked_question(
    profile_id: str,
    question: str,
    mode: str,
    behavioral_type: Optional[str] = None,
    entry_key: Optional[str] = None,
):
    data = load_session(profile_id)
    asked = data.get("asked_questions", [])
    now = datetime.datetime.utcnow().isoformat() + "Z"
    asked.append(
        {
            "timestamp": now,
            "mode": mode,
            "behavioral_type": behavioral_type,
            "entry_key": entry_key,
            "question": question,
        }
    )
    data["asked_questions"] = asked
    save_session(profile_id, data)


def get_practice_stats(profile_id: str) -> dict:
    data = load_session(profile_id)
    turns = data.get("turns", [])

    stats = {
        "total": len(turns),
        "by_mode": {
            "auto": 0,
            "behavioral": 0,
            "project": 0,
            "custom": 0,
        },
        "behavioral_by_type": {},
        "project_by_entry": {},
    }

    for t in turns:
        mode = t.get("mode")
        if mode in stats["by_mode"]:
            stats["by_mode"][mode] += 1

        if mode == "behavioral":
            bt = t.get("behavioral_type") or "unspecified"
            stats["behavioral_by_type"][bt] = stats["behavioral_by_type"].get(bt, 0) + 1

        if mode == "project":
            ek = t.get("entry_key") or "unspecified"
            stats["project_by_entry"][ek] = stats["project_by_entry"].get(ek, 0) + 1

    return stats
