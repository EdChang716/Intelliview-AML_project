# core/embeddings.py
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[1]
USER_DATA_DIR = ROOT_DIR / "user_data"
MODEL_DIR = ROOT_DIR / "models" / "jdq_bullet_finetuned"
DEVICE = "cpu"  # or "cuda"

retriever_model = SentenceTransformer(str(MODEL_DIR), device=DEVICE)

def load_edited_bullets(project_id: str):
    """讀 user_data/parsed/{project_id}/experience_entries_edited.json"""
    path = USER_DATA_DIR / "parsed" / project_id / "experience_entries_edited.json"
    
    # Fallback to non-edited version if edited doesn't exist
    if not path.exists():
        path = USER_DATA_DIR / "parsed" / project_id / "experience_entries.json"
    
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    # 每一筆 entry: {section, entry, text}
    texts = []
    meta = []
    for e in entries:
        combined = f"{e.get('entry', '')} — {e.get('text', '')}"
        texts.append(combined)
        meta.append(e)

    return texts, meta

def build_resume_embeddings(project_id: str, normalize: bool = True):
    """為特定 project_id 建 embeddings 並存到 user_data/embeddings/{project_id}/"""
    texts, meta = load_edited_bullets(project_id)
    if not texts:
        raise ValueError(f"No bullets for project {project_id}")

    emb = retriever_model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )

    out_dir = USER_DATA_DIR / "embeddings" / project_id
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "resume_bullets.npy", emb)
    with open(out_dir / "resume_bullets_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return emb, meta
