# scripts/ingest_resume.py
import argparse
from core.embeddings import build_resume_embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", required=True)
    args = parser.parse_args()

    emb, meta = build_resume_embeddings(args.project_id)
    print(f"Built embeddings for {args.project_id}: {emb.shape[0]} bullets.")

if __name__ == "__main__":
    main()
