# core/config.py
from pathlib import Path

# 專案根目錄：Intelliview_coach/
BASE_DIR = Path(__file__).resolve().parents[1]

USER_DATA_DIR = BASE_DIR / "user_data"
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = BASE_DIR / "models" / "jdq_bullet_finetuned"

SESSIONS_DIR = USER_DATA_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

SESSION_MEDIA_DIR = USER_DATA_DIR / "session_media"
SESSION_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
