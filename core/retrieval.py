# core/retrieval.py
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import USER_DATA_DIR, MODEL_DIR

_retriever_model = None

def get_retriever_model():
    global _retriever_model
    if _retriever_model is None:
        _retriever_model = SentenceTransformer(str(MODEL_DIR), device="cpu")
    return _retriever_model

def load_job_profiles():
    path = USER_DATA_DIR / "job_profiles.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("profiles", [])

def get_profile(profile_id: str):
    profiles = load_job_profiles()
    for p in profiles:
        if p.get("profile_id") == profile_id:
            return p
    raise ValueError(f"Profile {profile_id} not found")


def load_resume_entries_and_embs(resume_id: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """讀取該履歷版本的 bullets + 對應 embeddings"""
    parsed_dir = USER_DATA_DIR / "parsed" / resume_id
    edited_path = parsed_dir / "experience_entries_edited.json"
    raw_path = parsed_dir / "experience_entries.json"

    if edited_path.exists():
        entries = json.loads(edited_path.read_text(encoding="utf-8"))
    else:
        entries = json.loads(raw_path.read_text(encoding="utf-8"))

    emb_dir = USER_DATA_DIR / "embeddings" / resume_id
    emb_path = emb_dir / "resume_bullets.npy"
    if not emb_path.exists():
        raise ValueError(f"Embeddings not found for resume {resume_id}")

    embs = np.load(emb_path)
    return entries, embs


def retrieve_bullets_for_profile(
    profile_or_id,
    question: str,
    resume_id: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    傳進來可以是 profile_id (str) 或 profile dict。
    - 如果是 str：會幫你用 get_profile() 抓 profile
    - 如果是 dict：直接用
    - resume_id: 必須明確傳入
    """

    # --- 1. 把 profile_or_id 統一變成 profile dict ---
    if isinstance(profile_or_id, str):
        profile = get_profile(profile_or_id)
    elif isinstance(profile_or_id, dict):
        profile = profile_or_id
    else:
        raise TypeError(
            f"retrieve_bullets_for_profile expects a profile_id (str) or profile dict, "
            f"got {type(profile_or_id)}"
        )

    jd_text = profile.get("jd_text", "")

    # --- 2. 下面跟你原本的邏輯一樣 ---
    entries, emb_matrix = load_resume_entries_and_embs(resume_id)
    model = get_retriever_model()

    # 分開 encode
    q_emb_question = model.encode([question])[0]

    # JD 太長就截短一點，避免淹沒 question（可調）
    jd_snippet = jd_text[:600] if jd_text else ""
    if jd_snippet:
        q_emb_jd = model.encode([jd_snippet])[0]
        # 可自行調整權重
        q_emb = 0.8 * q_emb_question + 0.2 * q_emb_jd
    else:
        q_emb = q_emb_question

    norms = np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(q_emb)
    sims = emb_matrix @ q_emb / (norms + 1e-8)
    top_idx = np.argsort(-sims)[:top_k]

    print("RAG question:", question)
    print("Top idx:", top_idx)
    print("Top sims:", sims[top_idx])

    bullets: List[Dict[str, Any]] = []
    for i in top_idx:
        e = entries[int(i)]
        bullets.append(
            {
                "section": e.get("section"),
                "entry": e.get("entry"),
                "text": e.get("text"),
            }
        )
    return bullets


def get_bullets_for_entry(resume_id: str, entry_key: str) -> List[Dict[str, Any]]:
    entries, _ = load_resume_entries_and_embs(resume_id)
    section, entry_title = entry_key.split("||", 1)
    results = []
    for e in entries:
        if e.get("section") == section and (e.get("entry") or "") == entry_title:
            results.append(e)
    return results
