# core/answers.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re
import subprocess

import numpy as np
import librosa
from fastapi import HTTPException

from .llm_client import client


# ========== 1. 產生範例答案（給使用者看的 sample answer） ==========

def call_llm_for_answer(
    question: str,
    jd_text: str,
    bullets: List[Dict[str, Any]],
) -> str:
    ctx_lines = []
    for b in bullets:
        entry = b.get("entry") or "Unknown entry"
        text = b.get("text") or ""
        ctx_lines.append(f"- [{entry}] {text}")
    ctx = "\n".join(ctx_lines)

    prompt = (
        "You are an interview coach helping a candidate practice.\n\n"
        "Use ONLY the resume bullets below to write a clear, conversational answer "
        "to the interview question. Answer in first person (\"I ...\"), 2–3 short "
        "paragraphs, and implicitly follow a STAR-style structure (situation, task, "
        "actions, results) but don't label the sections.\n\n"
        f"Job description (for context):\n{jd_text}\n\n"
        f"Resume bullets (with entries):\n{ctx}\n\n"
        f"Question: {question}\n\n"
        "Write the answer in a natural spoken tone that would sound normal in a live interview."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def call_llm_for_sample_answer(
    question: str,
    jd_text: str,
    bullets: List[Dict[str, Any]],
    user_answer: Optional[str] = None,
) -> dict:
    clean_bullets: List[Dict[str, Any]] = []
    for b in bullets:
        if isinstance(b, dict):
            clean_bullets.append(b)
        else:
            clean_bullets.append({"entry": "Auto", "text": str(b)})
    bullets = clean_bullets

    bullet_lines = []
    for b in bullets:
        entry = b.get("entry") or "Unknown entry"
        text = b.get("text") or ""
        bullet_lines.append(f"- [{entry}] {text}")
    bullet_block = "\n".join(bullet_lines) if bullet_lines else "None."

    prompt = (
        "You are an interview coach helping a candidate prepare for data/ML/analytics roles.\n"
        "Given the job description, an interview question, and some resume bullets, do THREE things:\n"
        "1) Write a strong sample answer in first person, 2–3 paragraphs, using a STAR-like structure.\n"
        "2) Write a short 'hint' (2–3 sentences) describing how to structure a good answer.\n"
        "3) Write a short 'rationale' (3–5 sentences) explaining WHY this sample answer is effective.\n\n"
        "Return ONLY JSON with keys: answer, hint, rationale.\n\n"
        f"Job description:\n{jd_text}\n\n"
        f"Resume bullets:\n{bullet_block}\n\n"
        f"Interview question:\n{question}\n\n"
        f"User's draft answer:\n{user_answer or '(none)'}\n\n"
        "Now respond strictly in JSON."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    raw = resp.choices[0].message.content.strip()

    # 清掉 ```json ... ``` 之類的 wrapper
    cleaned = raw
    if cleaned.startswith("```"):
        cleaned = cleaned.strip()
        cleaned = cleaned[3:].lstrip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
        if "```" in cleaned:
            cleaned = cleaned.split("```", 1)[0].strip()

    data = None
    try:
        data = json.loads(cleaned)
    except Exception:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None

    if not isinstance(data, dict):
        return {
            "answer": raw,
            "hint": "",
            "rationale": "",
        }

    answer = data.get("answer", "")
    hint = data.get("hint", "")
    rationale = data.get("rationale", "")

    return {
        "answer": str(answer).strip(),
        "hint": str(hint).strip(),
        "rationale": str(rationale).strip(),
    }


# ========== 2. Audio feature 抽取 ==========

FILLER_WORDS = {"um", "uh", "like", "you know", "well"}


def extract_audio_features(wav_path: str | Path, transcript: str) -> Dict[str, Any]:
    """
    給一個 .wav 檔路徑 + transcript，回傳一些基本的 audio features：
    - duration_sec
    - wpm
    - silence_ratio
    - avg_volume
    - volume_std
    - pitch_range
    - filler_per_min
    """
    wav_path = Path(wav_path)
    y, sr = librosa.load(wav_path, sr=16000)

    # 1) 長度
    duration = librosa.get_duration(y=y, sr=sr)

    # 2) 音量 (RMS)
    rms = librosa.feature.rms(y=y)[0]
    avg_volume = float(np.mean(rms))
    volume_std = float(np.std(rms))

    # 3) 靜音比例（簡單用能量門檻判）
    silence_frames = np.sum(rms < 0.02)
    silence_ratio = float(silence_frames / len(rms))

    # 4) pitch range（非常粗略但夠用）
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    valid = magnitudes > np.median(magnitudes)
    pitch_values = pitches[valid]
    if pitch_values.size > 0:
        pitch_range = float(np.max(pitch_values) - np.min(pitch_values))
    else:
        pitch_range = 0.0

    # 5) 語速 & filler
    words = transcript.split()
    wpm = len(words) / (duration / 60 + 1e-6)

    filler_count = sum(w.lower() in FILLER_WORDS for w in words)
    filler_per_min = filler_count / (duration / 60 + 1e-6)

    return {
        "duration_sec": duration,
        "wpm": wpm,
        "silence_ratio": silence_ratio,
        "avg_volume": avg_volume,
        "volume_std": volume_std,
        "pitch_range": pitch_range,
        "filler_per_min": filler_per_min,
    }


# ========== 3. 核心：文字 +（可選）audio 評估 ==========

def evaluate_answer(
    question: str,
    jd_text: str,
    bullets: List[Dict[str, Any]],
    user_answer: str,
    audio_wav_path: Optional[str | Path] = None,
) -> dict:
    """
    用 LLM 評估面試回答（文字 transcript），輸出：
    - overall_score: 1–10
    - subscores: 各維度分數
    - strengths: 優點摘要
    - improvements_overview: 需要加強的總結
    - improvement_items: 每個面向的具體建議
    - sample_answer: 參考範例回答

    如果有提供 audio_wav_path，會另外回傳：
    - audio: {
        "has_audio": bool,
        "features": {... 原始數值特徵 ...},
        "delivery_score": 1–10,
        "delivery_comment": "..."
      }
    """

    # ===== 先組 resume context =====
    ctx_lines = []
    for b in bullets:
        entry = b.get("entry") or "Unknown entry"
        text = b.get("text") or ""
        ctx_lines.append(f"- [{entry}] {text}")
    ctx = "\n".join(ctx_lines)

    # ===== 如果有音檔，先算 audio features =====
    audio_features: Optional[Dict[str, Any]] = None
    audio_features_json = ""
    if audio_wav_path is not None:
        try:
            audio_features = extract_audio_features(audio_wav_path, user_answer)
            audio_features_json = json.dumps(audio_features, indent=2)
        except Exception:
            # 不讓整個評分爆掉，失敗就當沒音檔處理
            audio_features = None
            audio_features_json = ""

    # ===== 組 prompt：有 / 沒 audio 兩種版本 =====
    if audio_features is None:
        # 🔹 純文字版本
        prompt = f"""
You are an interview coach evaluating a TEXT transcript of a candidate's answer.

CRITICAL: First determine if the answer is substantive or not.
- If the answer is gibberish, random characters, "I don't know", or has no meaningful content related to the question, set overall_score to 1-2 and leave "strengths" as an empty string "". In "improvements_overview", state clearly that no meaningful content was provided.
- ONLY provide positive feedback in "strengths" if there are actual positive elements worth noting.

**RELEVANCE IS CRITICAL**: Even well-structured answers with good details must score LOW if they don't directly address what the question asks. Read the question carefully and penalize answers that miss key aspects.
Focus ONLY on the content, wording, and structure of the answer (not voice, tone, or body language), grammar is not important if its not affecting how you understand the content.

**FEEDBACK MUST BE ULTRA-SPECIFIC AND ACTIONABLE**:
- Don't say "add more detail" - say WHAT details to add
- Don't say "improve structure" - say HOW to restructure (e.g., "Start with the situation, then explain your task")
- Give EXAMPLES of sentences they could add
- Point to SPECIFIC parts of the question they didn't address

Return ONLY valid JSON with this exact structure, and nothing else:

{{
  "overall_score": <integer 1-10>,
  "subscores": {{
    "relevance": <integer 1-10>,         // how well the answer addresses the question and JD, penalize missing key elements of the question.
    "structure": <integer 1-10>,         // organization, logical flow, STAR-like clarity
    "clarity": <integer 1-10>,           // clear wording, easy to follow
    "depth": <integer 1-10>,             // concrete details, actions, and results
    "conciseness": <integer 1-10>        // avoids rambling and repetition
  }},
  "strengths": "<one short paragraph summarizing the main strengths>",
  "improvements_overview": "<A DETAILED 4-6 sentence explanation of what to improve, including specific examples and concrete suggestions. Be straight to the point, concise and actionable.>",
  "improvement_items": [
    {{
      "aspect": "Relevance | Structure | Depth | Clarity",
      "issue": "PINPOINT THE EXACT PROBLEM. Quote missing question elements. Point to specific unclear parts. Be surgical, not general.",
      "suggestion": "GIVE CONCRETE REWRITES. Example format: 'Add this after your opening: \"The situation was that our ad targeting model had 30% false positives. My task was to reduce this to under 10% within Q2.\"' OR 'Replace your current opening with: \"At [Company], I debugged a critical issue in our recommendation system where...\"' Give them actual sentences or phrases to use."
    }},
    {{
      "aspect": "Another aspect",
      "issue": "Another specific issue with examples from their answer",
      "suggestion": "Another concrete rewrite suggestion with example text"
    }}
    // 2-5 items total; you may include fewer or more aspects as needed. Each suggestion should include EXAMPLE SENTENCES or PHRASES the candidate could use.
  ],
  "sample_answer": "<a strong sample answer to this question, 6-10 sentences, using clear structure (e.g., STAR), referring to the job description when helpful and directly addressing ALL parts of the question. Include specific metrics and technical details. Do NOT mention that you are an AI.>"
}}

Scoring guidelines (1-10 for all scores):
- 9-10: Addresses question fully, clear structure, concrete details, ready for real interviews
- 7-8: mostly answers question, good structure, minor gaps in detail or relevance
- 5-6: some good elements but missing key parts of question OR unclear structure OR lack of concrete details
- 3-4: major issues: doesn't fully address question OR very unclear OR lacks specific examples
- 1-2: Very weak, gibberish, no meaningful content, or completely misses the question

Job description:
{jd_text}

Relevant resume bullets:
{ctx}

Question:
{question}

User answer (transcript):
{user_answer}

Remember:
- Focus on TEXT QUALITY ONLY.
- Return JSON only, no explanations, no extra text, no code fences.
"""
    else:
        # 🔹 有 audio 的版本：多一個 audio 區塊
        prompt = f"""
You are an interview coach evaluating a candidate's interview answer.

You have:
1) The TEXT transcript of the answer.
2) The job description and relevant resume bullets.
3) Numeric AUDIO DELIVERY METRICS extracted from the recording.

Use:
- The TEXT to evaluate content and structure.
- The AUDIO METRICS ONLY to evaluate vocal delivery (pace, pauses, fillers, energy).
Do NOT infer anything about accent, gender, or background.

AUDIO_METRICS (JSON):
{audio_features_json}

Return ONLY valid JSON with this exact structure, and nothing else:

{{
  "overall_score": <integer 1-10>,
  "subscores": {{
    "relevance": <integer 1-10>,
    "structure": <integer 1-10>,
    "clarity": <integer 1-10>,
    "depth": <integer 1-10>,
    "conciseness": <integer 1-10>
  }},
  "audio": {{
    "delivery_score": <integer 1-10>,     // overall vocal delivery (pace, pauses, fillers, energy)
    "delivery_comment": "<short paragraph: what is good, what to improve, based on the metrics>"
  }},
  "strengths": "<one short paragraph summarizing the main strengths (content + structure)>",
  "improvements_overview": "<one short paragraph summarizing key areas to improve (content + delivery)>",
  "improvement_items": [
    {{
      "aspect": "Structure",
      "issue": "What is currently weak or missing",
      "suggestion": "Concrete suggestion on how to rewrite or add specific content"
    }},
    {{
      "aspect": "Relevance",
      "issue": "What is currently weak or missing",
      "suggestion": "Concrete suggestion on how to better align with the question and JD"
    }}
    // 2-5 items total; aspects may include: Structure, Relevance, Clarity, Depth, Conciseness, Vocal delivery
  ],
  "sample_answer": "<a strong sample answer to this question, 4-8 sentences, using clear structure (e.g., STAR) and referring to the job description when helpful. Do NOT mention that you are an AI.>"
}}

Job description:
{jd_text}

Relevant resume bullets:
{ctx}

Question:
{question}

User answer (transcript):
{user_answer}

Remember:
- Use AUDIO_METRICS ONLY for vocal delivery.
- Do not guess about the candidate's identity or background.
- Return JSON only, no explanations, no extra text, no code fences.
"""

    # ===== 呼叫 LLM =====
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a strict but supportive interview coach helping candidates improve their answers."
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    raw = resp.choices[0].message.content.strip()

    # 防呆：有時候 model 還是會包 ```json ... ```
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()
        raw = raw.split("```")[0].strip()

    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                parsed = None

    # ===== 預設回傳（fallback） =====
    default_result = {
        "overall_score": 5,
        "subscores": {
            "relevance": 5,
            "structure": 5,
            "clarity": 5,
            "depth": 5,
            "conciseness": 5,
        },
        "strengths": "",
        "improvements_overview": "The model returned an unparsable result. Raw output:\n" + raw,
        "improvement_items": [],
        "sample_answer": "",
        "audio": {
            "has_audio": audio_features is not None,
            "features": audio_features or {},
            "delivery_score": 5,
            "delivery_comment": "",
        },
    }

    if not parsed:
        return default_result

    # ===== 安全處理分數 =====
    def clamp_score(x, default=5):
        try:
            v = int(x)
        except Exception:
            v = default
        return max(1, min(10, v))

    overall_score = clamp_score(parsed.get("overall_score", 5))

    subs = parsed.get("subscores", {}) or {}
    subscores = {
        "relevance": clamp_score(subs.get("relevance", 5)),
        "structure": clamp_score(subs.get("structure", 5)),
        "clarity": clamp_score(subs.get("clarity", 5)),
        "depth": clamp_score(subs.get("depth", 5)),
        "conciseness": clamp_score(subs.get("conciseness", 5)),
    }

    strengths = (parsed.get("strengths") or "").strip()
    improvements_overview = (parsed.get("improvements_overview") or "").strip()

    improvement_items = parsed.get("improvement_items") or []
    normalized_items = []
    for item in improvement_items:
        if not isinstance(item, dict):
            continue
        normalized_items.append({
            "aspect": (item.get("aspect") or "").strip(),
            "issue": (item.get("issue") or "").strip(),
            "suggestion": (item.get("suggestion") or "").strip(),
        })

    sample_answer = (parsed.get("sample_answer") or "").strip()

    # ===== audio block =====
    raw_audio = parsed.get("audio") or {}
    audio_result = {
        "has_audio": audio_features is not None,
        "features": audio_features or {},
        "delivery_score": clamp_score(raw_audio.get("delivery_score", 5)) if audio_features is not None else 5,
        "delivery_comment": (raw_audio.get("delivery_comment") or "").strip() if audio_features is not None else "",
    }

    return {
        "overall_score": overall_score,
        "subscores": subscores,
        "strengths": strengths,
        "improvements_overview": improvements_overview,
        "improvement_items": normalized_items,
        "sample_answer": sample_answer,
        "audio": audio_result,
    }
