# core/transcription.py
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import requests

from .llm_client import client

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ========== 現有：整段檔案一次轉錄（備用/非 realtime） ==========

def transcribe_media(
    path: Union[str, Path],
    *,
    language: str = "en",
    prompt: Optional[str] = None,
) -> str:
    """
    使用 OpenAI Whisper-1 把 audio/video 檔案轉成文字。

    - path: 檔案實際路徑（可以是 audio 或 video，Whisper 會自動抽出音軌）
    - language: 語言（如果你都是英文回答，可以固定 "en"，會比較穩）
    - prompt: 可選的小提示（例如 "This is an interview answer..."）

    回傳：轉錄後的純文字（失敗的話回傳空字串）
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Media file not found: {p}")

    with p.open("rb") as f:
        try:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=language,
                response_format="json",
                temperature=0,
                prompt=prompt,
            )
        except Exception as e:
            # 避免整個 API 爆掉，留個 log 就好
            print(f"[transcribe_media] Error during transcription: {e}")
            return ""

    # 新版 SDK 會有 resp.text
    text = getattr(resp, "text", None)
    if not text:
        return ""
    return text.strip()


def transcribe_media_with_segments(
    path: Path,
    language: str = "en",
    prompt: str = "",
) -> Dict[str, Any]:
    """
    回傳：
    {
      "text": "整段全文 transcript",
      "segments": [
        {"start": 0.0, "end": 3.2, "text": "..."},
        ...
      ]
    }

    已改為使用 gpt-4o-mini-transcribe
    - 速度快
    - 支援逐段 segments
    """
    with path.open("rb") as f:
        resp = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",   # ← 使用 4o-mini 模型
            file=f,
            language=language,
            response_format="verbose_json",   # ← 仍可用 verbose_json 拿 segments
            prompt=prompt or None,
        )

    # 兼容不同模型 response 格式
    full_text = getattr(resp, "text", "") or ""

    raw_segments = []
    # OpenAI 可能回傳:
    # resp.segments 或 resp["segments"] 或 resp["output"]["segments"]
    if hasattr(resp, "segments"):
        raw_segments = resp.segments or []
    elif isinstance(resp, dict):
        raw_segments = resp.get("segments", []) or resp.get("output", {}).get("segments", [])
    else:
        raw_segments = []

    segments_out: List[Dict[str, Any]] = []
    for seg in raw_segments:
        # dict or object 都支援
        if isinstance(seg, dict):
            start_sec = float(seg.get("start", 0.0))
            end_sec = float(seg.get("end", 0.0))
            text = (seg.get("text", "") or "").strip()
        else:
            start_sec = float(getattr(seg, "start", 0.0))
            end_sec = float(getattr(seg, "end", 0.0))
            text = (getattr(seg, "text", "") or "").strip()

        segments_out.append(
            {
                "start": start_sec,
                "end": end_sec,
                "text": text,
            }
        )

    return {
        "text": full_text,
        "segments": segments_out,
    }