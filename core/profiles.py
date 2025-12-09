from pathlib import Path
import json
from .config import USER_DATA_DIR

JOB_PROFILES_PATH = USER_DATA_DIR / "job_profiles.json"

def load_job_profiles() -> list[dict]:
    if not JOB_PROFILES_PATH.exists():
        return []
    try:
        data = json.loads(JOB_PROFILES_PATH.read_text(encoding="utf-8"))
        return data.get("profiles", [])
    except Exception:
        return []

def save_job_profiles(profiles: list[dict]) -> None:
    JOB_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOB_PROFILES_PATH.write_text(
        json.dumps({"profiles": profiles}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def load_all_profiles() -> list[dict]:
    # Alias for load_job_profiles, used by some routers
    return load_job_profiles()
