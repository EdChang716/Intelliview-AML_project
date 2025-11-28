# core/rag_pipeline.py
from pathlib import Path
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
import datetime
import random
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv



BASE_DIR = Path(__file__).resolve().parents[1]
USER_DATA_DIR = BASE_DIR / "user_data"
MODEL_DIR = BASE_DIR / "models" / "jdq_bullet_finetuned"
SESSIONS_DIR = USER_DATA_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# 讀入 .env
load_dotenv(BASE_DIR / ".env")

#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your_API_key"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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


def load_resume_entries_and_embs(resume_id: str):
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


def retrieve_bullets_for_profile(profile_id: str, question: str, top_k: int = 3):
    profile = get_profile(profile_id)
    resume_id = profile["resume_id"]
    jd_text = profile.get("jd_text", "")

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

    bullets = []
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



def call_llm_for_question(jd_text: str, mode: str, avoid: set[str] | None = None) -> str:
    """用 JD + mode 生一題題目，避免出現在 avoid 裡"""
    avoid = avoid or set()

    if mode == "behavioral":
        style = "behavioral STAR-format question focusing on teamwork, conflict, or impact"
    elif mode == "project":
        style = "deep-dive question about one of the candidate's past projects that is relevant to this job"
    else:  # auto
        style = "interview question that mixes behavioral and resume-based aspects relevant to this job"

    # 針對 auto 模式隨機指定一個「角度」，讓題目多樣化
    angle = ""
    if mode == "auto":
        angle_choices = [
            "focus on the measurable impact and metrics of a past project",
            "focus on collaboration with teammates or cross-functional stakeholders",
            "focus on how the candidate dealt with ambiguity or unclear requirements",
            "focus on a tricky debugging or failure case and what they learned",
            "focus on how they balanced trade-offs such as speed vs quality or model complexity vs reliability",
        ]
        angle = random.choice(angle_choices)

    # 告訴 model 之前問過哪些
    avoid_block = ""
    if avoid:
        joined = "\n".join(f"- {q}" for q in list(avoid)[:10])
        avoid_block = (
            "We have already asked the following questions in previous turns.\n"
            "Do NOT ask about the same topics or scenarios again. "
            "Ask something that probes a clearly different aspect of the candidate's skills "
            "(for example: a different project, different type of challenge, different metric, or different soft skill).\n"
            "Previously asked questions:\n"
            f"{joined}\n\n"
        )

    prompt = (
        "You are an interview coach. Given the job description below, write ONE "
        f"{style}. The question should be short and natural, as an interviewer would say it.\n\n"
        f"{avoid_block}"
        f"Job description:\n{jd_text}\n\n"
    )

    if angle:
        prompt += f"The question should specifically {angle}.\n\n"

    prompt += "Return only the question text."

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    text = resp.choices[0].message.content.strip()
    q = text.split("\n")[0].strip()

    # 如果 model 還是給重複題，就簡單 retry 一次
    if q in avoid:
        resp2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
        )
        text2 = resp2.choices[0].message.content.strip()
        q2 = text2.split("\n")[0].strip()
        if q2 and q2 not in avoid:
            q = q2

    return q

def call_llm_for_followup_question(
    jd_text: str,
    mode: str,
    base_question: str,
    bullets: list[dict],
    qa_history: list[dict] | None = None,
    avoid: set[str] | None = None,
) -> str:
    """
    根據某一題 (base_question) & 這題底下的 QA 歷史，產生下一個追問問題。
    qa_history: [{ "question": "...", "answer": "..." }, ...]
    avoid: 這個 thread 裡已經問過的問題（主題 + 追問）
    """
    qa_history = qa_history or []
    avoid = avoid or set()

    prev_block = ""
    if qa_history:
        lines = []
        for qa in qa_history[-3:]:
            q = qa.get("question", "")
            a = qa.get("answer", "")
            lines.append(f"Q: {q}\nA: {a}\n")
        prev_block = (
            "Here is the previous conversation about this topic:\n"
            + "\n".join(lines)
            + "\n\n"
        )

    bullet_text = "\n".join(f"- {b.get('text','')}" for b in bullets)

    style_hint = {
        "behavioral": "a deeper behavioral follow-up question (for example: emotions, conflict, reflection, or what they would do differently).",
        "project": "a deeper technical follow-up about decisions, trade-offs, metrics, or collaboration.",
        "auto": "a deeper follow-up about impact, decisions, or what they would do differently.",
        "custom": "a deeper follow-up from a different perspective on the same situation.",
    }.get(mode, "a deeper follow-up question.")

    avoid_block = ""
    if avoid:
        joined = "\n".join(f"- {q}" for q in list(avoid)[:10])
        avoid_block = (
            "Do NOT repeat or rephrase any of the following questions:\n"
            f"{joined}\n\n"
        )

    prompt = (
        "You are an interviewer asking follow-up questions in a realistic live interview.\n\n"
        f"Job description (for context):\n{jd_text}\n\n"
        f"The main question is:\n{base_question}\n\n"
        f"Relevant resume bullets:\n{bullet_text}\n\n"
        f"{prev_block}"
        f"{avoid_block}"
        f"Now ask ONE short {style_hint}\n"
        "Return only the question text, as you would say it to the candidate."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    text = resp.choices[0].message.content.strip()
    return text.split("\n")[0].strip()


def generate_followup_question(
    jd_text: str,
    mode: str,
    base_question: str,
    bullets: list[dict],
    qa_history: list[dict] | None,
    avoid: set[str],
) -> Optional[str]:
    """
    包一層，負責：
    - 避免重複（簡單 normalization + 多次嘗試）
    - 如果真的避不開，回 None
    """
    qa_history = qa_history or []

    def norm(s: str) -> str:
        return s.rstrip("?.! ").strip().lower()

    avoid_norm = {norm(q) for q in avoid if q}

    for temp in [0.7, 0.9, 1.0]:
        q = call_llm_for_followup_question(
            jd_text=jd_text,
            mode=mode,
            base_question=base_question,
            bullets=bullets,
            qa_history=qa_history,
            avoid=avoid,
        )
        q_norm = norm(q)
        if q and q_norm not in avoid_norm:
            return q

    # 真的生不出新問題，就回 None，讓上層決定要不要結束 thread
    return None

BEHAVIORAL_BANK = {
    "teamwork": [
        "Tell me about a time you worked on a team and faced a challenge.",
        "Describe a time you had to help a teammate who was struggling.",
        "Tell me about a time when you disagreed with your team and how you handled it.",
    ],
    "conflict": [
        "Tell me about a time you had a conflict with a teammate or stakeholder.",
        "Describe a time you had to deliver bad news to someone.",
    ],
    "leadership": [
        "Tell me about a time you had to take the lead on a project.",
        "Describe a situation where you motivated others to achieve a goal.",
    ],
    "failure": [
        "Tell me about a time you failed at something and what you learned.",
        "Describe a time a project did not go as planned.",
    ],
    "strengths_weaknesses": [
        "What do you consider your greatest strength, and can you give me an example?",
        "Tell me about a weakness you are actively working on.",
    ],
}

def get_behavioral_question(profile_id: str, subtype: str) -> str:
    bank = BEHAVIORAL_BANK.get(subtype)
    if not bank:
        # 沒指定 subtype 就從全部題目裡抽
        all_q = []
        for lst in BEHAVIORAL_BANK.values():
            all_q.extend(lst)
        bank = all_q

    asked = get_asked_questions(profile_id, mode="behavioral", behavioral_type=subtype)
    remaining = [q for q in bank if q not in asked]

    if remaining:
        return random.choice(remaining)
    else:
        # 全部問過了，隨機重複一題
        return random.choice(bank)

def get_bullets_for_entry(resume_id: str, entry_key: str) -> list[dict]:
    entries, _ = load_resume_entries_and_embs(resume_id)
    section, entry_title = entry_key.split("||", 1)
    results = []
    for e in entries:
        if e.get("section") == section and (e.get("entry") or "") == entry_title:
            results.append(e)
    return results


def call_llm_for_project_question(
    jd_text: str,
    entry_title: str,
    bullets: list[dict],
    previous_qas: list[dict] | None = None,
) -> str:
    """
    給某個 project/experience（entry_title + bullets）＋ 過去 QA context，
    產生一個 deep dive / follow-up 問題。
    previous_qas: [{ "question": "...", "answer": "..." }, ...]
    """
    prev_block = ""
    if previous_qas:
        lines = []
        for qa in previous_qas[-3:]:  # 只放最後 3 題
            q = qa.get("question", "")
            a = qa.get("answer", "")
            lines.append(f"Q: {q}\nA: {a}\n")
        if lines:
            prev_block = (
                "Here is the previous conversation about this project:\n"
                + "\n".join(lines)
                + "\n\n"
                "Ask a follow-up question that goes deeper into a different aspect "
                "(decisions, trade-offs, metrics, collaboration, or reflection).\n\n"
            )

    bullet_text = "\n".join(f"- {b.get('text','')}" for b in bullets)

    prompt = (
        "You are an interviewer doing a deep dive on ONE specific project from the candidate's resume.\n\n"
        f"Job description (for context):\n{jd_text}\n\n"
        f"The project/experience is:\n{entry_title}\n\n"
        f"Relevant bullets from the resume:\n{bullet_text}\n\n"
        f"{prev_block}"
        "Write ONE short follow-up question you would ask about this project. "
        "Return only the question text."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    text = resp.choices[0].message.content.strip()
    return text.split("\n")[0].strip()

def call_llm_for_answer(question: str, jd_text: str, bullets: list[dict]) -> str:
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
    """
    Call LLM to generate sample answer + hint + rationale.
    Tries hard to parse JSON even if model wraps it in ```json fences.
    """

    # -------- sanitize bullets --------
    clean_bullets = []
    for b in bullets:
        if isinstance(b, dict):
            clean_bullets.append(b)
        else:
            clean_bullets.append({
                "entry": "Auto",
                "text": str(b)
            })
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

    # ---- Clean model output: remove ```json ... ``` fences ----
    # 例如 ```json\n{ ... }\n``` 或 ```\n{ ... }\n```
    cleaned = raw
    if cleaned.startswith("```"):
        # 去掉開頭的 ``` 和可能的 "json"
        cleaned = cleaned.strip()
        # 移除第一個 ``` 
        cleaned = cleaned[3:].lstrip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
        # 再切掉最後一組 ``` 之後的內容
        if "```" in cleaned:
            cleaned = cleaned.split("```", 1)[0].strip()

    # ---- Try JSON.loads directly ----
    data = None
    try:
        data = json.loads(cleaned)
    except Exception:
        # 再試著用 regex 抽出第一個 { ... } 區塊
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None

    # ---- fallback if still broken ----
    if not isinstance(data, dict):
        # 真的 parse 不到，就整段 raw 當 answer，hint/rationale 空
        return {
            "answer": raw,
            "hint": "",
            "rationale": "",
        }

    # ---- extract fields ----
    answer = data.get("answer", "")
    hint = data.get("hint", "")
    rationale = data.get("rationale", "")

    return {
        "answer": str(answer).strip(),
        "hint": str(hint).strip(),
        "rationale": str(rationale).strip(),
    }




def generate_sample_answer_for_profile(profile_id: str, question: str):
    """整合：RAG 選 bullets + LLM 產生答案"""
    profile = get_profile(profile_id)
    jd_text = profile.get("jd_text", "")

    bullets = retrieve_bullets_for_profile(profile_id, question, top_k=5)
    answer = call_llm_for_answer(question, jd_text, bullets)
    return answer, bullets


def _session_path(profile_id: str) -> Path:
    return SESSIONS_DIR / f"{profile_id}.json"


def load_session(profile_id: str) -> dict:
    path = _session_path(profile_id)
    if not path.exists():
        return {"profile_id": profile_id, "turns": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_session(profile_id: str, data: dict):
    path = _session_path(profile_id)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_asked_questions(
    profile_id: str,
    mode: Optional[str] = None,
    behavioral_type: Optional[str] = None,
    entry_key: Optional[str] = None,
) -> set[str]:
    """
    從 session 裡抓出「使用者有回答並存檔過的主題目」，用來避免重複。
    追問 (is_followup=True) 不會被拿來避免重複主題問題。
    """
    data = load_session(profile_id)
    questions: set[str] = set()

    for t in data.get("turns", []):
        # 忽略追問，只把主題目拿來避免重複
        if t.get("is_followup"):
            continue

        if mode and t.get("mode") != mode:
            continue
        if behavioral_type and t.get("behavioral_type") != behavioral_type:
            continue
        if entry_key and t.get("entry_key") != entry_key:
            continue

        q = t.get("question")
        if q:
            questions.add(q.strip())

    return questions



def log_practice_turn(
    profile_id: str,
    question: str,
    sample_answer: Optional[str],
    bullets: List[Dict[str, Any]],
    mode: str,
    behavioral_type: Optional[str] = None,
    entry_key: Optional[str] = None,
    user_answer: Optional[str] = None,
    score: Optional[int] = None,
    strengths: Optional[str] = None,
    improvements: Optional[str] = None,
    thread_id: Optional[str] = None,
    is_followup: bool = False,
):
    """
    每次產生 / 存答案時，把這一題完整記錄下來。
    thread_id: 同一條主題 + 追問共用的 ID（前端可用 UUID 或主題目）
    is_followup: 是否為追問問題（True 表示 follow-up）
    """
    data = load_session(profile_id)
    turns = data.get("turns", [])

    now = datetime.datetime.utcnow().isoformat() + "Z"
    turns.append(
        {
            "timestamp": now,
            "mode": mode,
            "behavioral_type": behavioral_type,
            "entry_key": entry_key,
            "question": question,
            "user_answer": user_answer,
            "sample_answer": sample_answer,
            "bullets": bullets,
            "score": score,
            "strengths": strengths,
            "improvements": improvements,
            "thread_id": thread_id,
            "is_followup": is_followup,
        }
    )
    data["turns"] = turns
    save_session(profile_id, data)

def log_asked_question(
    profile_id: str,
    question: str,
    mode: str,
    behavioral_type: Optional[str] = None,
    entry_key: Optional[str] = None,
):
    """
    記錄「這個 profile 在某個 mode / subtype / entry 下被問過哪一題」，
    不管使用者有沒有存答案。這些資料只拿來避免重複，不算進 stats。
    """
    data = load_session(profile_id)
    asked = data.get("asked_questions", [])
    now = datetime.datetime.utcnow().isoformat() + "Z"
    asked.append(
        {
            "timestamp": now,
            "mode": mode,
            "behavioral_type": behavioral_type,
            "entry_key": entry_key,
            "question": question,
        }
    )
    data["asked_questions"] = asked
    save_session(profile_id, data)

def get_practice_stats(profile_id: str) -> dict:
    data = load_session(profile_id)
    turns = data.get("turns", [])

    stats = {
        "total": len(turns),
        "by_mode": {
            "auto": 0,
            "behavioral": 0,
            "project": 0,
            "custom": 0,
        },
        "behavioral_by_type": {},
        "project_by_entry": {},
    }

    for t in turns:
        mode = t.get("mode")
        if mode in stats["by_mode"]:
            stats["by_mode"][mode] += 1

        if mode == "behavioral":
            bt = t.get("behavioral_type") or "unspecified"
            stats["behavioral_by_type"][bt] = stats["behavioral_by_type"].get(bt, 0) + 1

        if mode == "project":
            ek = t.get("entry_key") or "unspecified"
            stats["project_by_entry"][ek] = stats["project_by_entry"].get(ek, 0) + 1

    return stats

# add scoring system (text)
def evaluate_answer(
    question: str,
    jd_text: str,
    bullets: list[dict],
    user_answer: str,
) -> dict:
    """
    Use LLM to evaluate the user's interview answer.
    Always returns a clean dict:
      {
        "score": int (1–10),
        "strengths": str,
        "improvements": str
      }
    """

    # ---- prepare bullet context ----
    ctx_lines = []
    for b in bullets:
        entry = b.get("entry") or "Unknown entry"
        text = b.get("text") or ""
        ctx_lines.append(f"- [{entry}] {text}")
    ctx = "\n".join(ctx_lines)

    # ---- strict JSON-only prompt ----
    prompt = f"""
You are an interview coach. Evaluate the candidate's answer to the interview question.
Return ONLY valid JSON with this structure, and nothing else:

{{
  "score": <integer 1-10>,
  "strengths": "<one short paragraph>",
  "improvements": "<one short paragraph>"
}}

NO explanations, NO extra text, NO code fences.

Job description:
{jd_text}

Relevant resume bullets:
{ctx}

Question:
{question}

User answer:
{user_answer}

Return JSON only.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    raw = resp.choices[0].message.content.strip()

    # ---- Clean model output: remove ```json ... ``` fences ----
    # e.g. ```json {...}``` or ``` {...}```
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()
        # After removing json, still may have trailing ```
        raw = raw.split("```")[0].strip()

    # ---- Try to parse JSON ----
    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        # try regex to extract {...}
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except:
                parsed = None

    # ---- fallback if still broken ----
    if not parsed:
        return {
            "score": 5,
            "strengths": "",
            "improvements": "The model returned an unparsable result. Raw output:\n" + raw
        }

    # ---- extract + sanitise ----
    score = parsed.get("score", 5)
    try:
        score = int(score)
    except:
        score = 5
    score = max(1, min(10, score))   # clamp to 1–10

    strengths = parsed.get("strengths", "")
    improvements = parsed.get("improvements", "")

    return {
        "score": score,
        "strengths": strengths.strip(),
        "improvements": improvements.strip(),
    }