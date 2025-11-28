# core/profiles.py
from .config import USER_DATA_DIR
from pathlib import Path
import json
from typing import List, Dict, Any

JOB_PROFILES_PATH = USER_DATA_DIR / "job_profiles.json"

def load_job_profiles() -> List[Dict[str, Any]]:
    if not JOB_PROFILES_PATH.exists():
        return []
    data = json.loads(JOB_PROFILES_PATH.read_text(encoding="utf-8"))
    return data.get("profiles", [])

def save_job_profiles(profiles: List[Dict[str, Any]]) -> None:
    JOB_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOB_PROFILES_PATH.write_text(
        json.dumps({"profiles": profiles}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def load_all_profiles() -> List[Dict[str, Any]]:
    # 目前就等於 load_job_profiles，方便 main 使用
    return load_job_profiles()
