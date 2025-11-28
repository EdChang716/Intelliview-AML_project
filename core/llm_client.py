# core/llm_client.py
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment or .env")

client = OpenAI(api_key=api_key)


def generate_sample_answer(question: str, jd_text: str, bullets: list[str]) -> str:
    """
    bullets: top-k bullet strings
    """
    prompt = f"""You are an interview coach.

Job description:
{jd_text}

Relevant experience bullet points:
- """ + "\n- ".join(bullets) + f"""

Interview question:
{question}

Based on ONLY the bullet points above (do not invent new jobs), draft a strong,
concise answer the candidate could say in an interview.
"""

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        max_output_tokens=400,
    )
    return resp.output[0].content[0].text
