# core/retriever.py
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer

from .embeddings import ROOT_DIR, USER_DATA_DIR, MODEL_DIR, DEVICE

retriever_model = SentenceTransformer(str(MODEL_DIR), device=DEVICE)

class ResumeRetriever:
    def __init__(self, project_id: str):
        self.project_id = project_id
        emb_dir = USER_DATA_DIR / "embeddings" / project_id
        emb_path = emb_dir / "resume_bullets.npy"
        meta_path = emb_dir / "resume_bullets_meta.json"

        if not emb_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found for project {project_id}. "
                f"Run build_resume_embeddings first."
            )

        self.embeddings = np.load(emb_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def retrieve_bullets(self, jd_text: str, question: str, k: int = 5):
        """
        用 (JD + Question) 當 query，回 top-k bullets
        """
        query = f"Job description:\n{jd_text}\n\nInterview question:\n{question}"
        q_emb = retriever_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        # cosine similarity (embedding 已 normalize)
        scores = np.dot(self.embeddings, q_emb)
        idx = np.argsort(-scores)[:k]

        results = []
        for i in idx:
            item = dict(self.meta[i])
            item["score"] = float(scores[i])
            results.append(item)
        return results
