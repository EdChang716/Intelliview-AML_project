from __future__ import annotations

import datetime
import json as _json
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from pathlib import Path
import subprocess
from fastapi import HTTPException

from .retrieval import retrieve_bullets_for_profile, load_resume_entries_and_embs
from .llm_client import client
from core.answers import evaluate_answer
from .transcription import transcribe_media_with_segments
from .profiles import load_job_profiles
from .questions import (
    call_llm_for_question,         # 用 JD + LLM 生 technical / auto / case 題
    call_llm_for_project_question  # 用 JD + resume entry 生 project deep dive 題
)
from core.video_features import extract_video_features

# ---- 路徑設定 ----

BASE_DIR = Path(__file__).resolve().parents[1]
USER_DATA_DIR = BASE_DIR / "user_data"

MOCK_BASE_DIR = USER_DATA_DIR / "mock"
MOCK_SESSIONS_DIR = MOCK_BASE_DIR / "sessions"
MOCK_MEDIA_DIR = MOCK_BASE_DIR / "media"
MOCK_RESULTS_DIR = MOCK_BASE_DIR / "results"

for d in [MOCK_SESSIONS_DIR, MOCK_MEDIA_DIR, MOCK_RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
#  設定：Behavioral 類別定義
# ============================================================

# 定義你要求的幾大類別
BEHAVIORAL_CATEGORIES = [
    "teamwork",
    "conflict",
    "leadership",
    "failure_mistakes",
    "strengths_weaknesses"
]

# 緊急 Fallback 用的小型題庫 (當 LLM API 失敗時才用)
FALLBACK_BEHAVIORAL_BANK = [
    {"text": "Tell me about a time you faced a challenge.", "tag": "Behavioral · General"},
    {"text": "Describe a time you worked in a team.", "tag": "Behavioral · Teamwork"},
    {"text": "Tell me about a time you failed.", "tag": "Behavioral · Failure"},
]


# ============================================================
#  Question 設計：題數 / 時間估計 & 題型規劃
# ============================================================

def _estimate_questions_for_time(minutes: int) -> int:
    return 30  # 粗估 upper bound


def _build_question_plan(
    length_type: str,
    num_questions: Optional[int],
    time_limit: Optional[int],
) -> List[Dict[str, Any]]:
    if length_type == "questions":
        total_slots = max(1, num_questions or 5)
    else:
        total_slots = max(1, _estimate_questions_for_time(time_limit or 30))

    plan: List[Dict[str, Any]] = []

    if total_slots == 1:
        plan.append({"index": 0, "type": "intro"})
        return plan

    if total_slots == 2:
        plan.append({"index": 0, "type": "intro"})
        plan.append({"index": 1, "type": "project"})
        return plan

    if total_slots == 3:
        plan.append({"index": 0, "type": "intro"})
        plan.append({"index": 1, "type": "project"})
        plan.append({"index": 2, "type": "case"})
        return plan

    if total_slots == 4:
        plan.append({"index": 0, "type": "intro"})
        plan.append({"index": 1, "type": "project"})
        plan.append({"index": 2, "type": "case"})
        plan.append({"index": 3, "type": "behavioral"})
        return plan

    # --- general case: total_slots >= 5 ---
    plan.append({"index": 0, "type": "intro"})     # Q0: intro
    plan.append({"index": 1, "type": "project"})   # Q1: project deep dive
    plan.append({"index": 2, "type": "technical"}) # Q2: technical
    plan.append({"index": 3, "type": "case"})      # Q3: case reasoning

    last_behavioral_start = max(4, total_slots - 2)

    idx = 4
    while idx < last_behavioral_start:
        q_type = "technical" if random.random() < 0.7 else "auto"
        plan.append({"index": idx, "type": q_type})
        idx += 1

    while idx < total_slots:
        plan.append({"index": idx, "type": "behavioral"})
        idx += 1

    for i, spec in enumerate(plan):
        spec["index"] = i

    return plan


# ============================================================
#  Session 管理 & project deep dive entry 選擇
# ============================================================

def _load_profile_jd_for_questions(profile_id: Optional[str]) -> str:
    if not profile_id:
        return ""
    try:
        profiles = load_job_profiles()
    except Exception:
        return ""
    profile = next((p for p in profiles if p.get("profile_id") == profile_id), None)
    if not profile:
        return ""
    return (
        profile.get("jd_text")
        or profile.get("jd")
        or profile.get("job_description")
        or ""
    )


def _pick_primary_project_entry(profile_id: str, resume_id: str) -> Optional[str]:
    jd_text = _load_profile_jd_for_questions(profile_id)
    if not jd_text.strip():
        return None

    try:
        top_bullets = retrieve_bullets_for_profile(profile_id, jd_text, resume_id, top_k=10)
    except Exception as e:
        print("[mock] _pick_primary_project_entry retrieve error:", e)
        return None

    if not top_bullets:
        return None

    stats: Dict[str, Dict[str, Any]] = {}
    for rank, b in enumerate(top_bullets):
        entry = b.get("entry")
        section = b.get("section") or "EXPERIENCE"
        if not entry:
            continue
        entry_key = f"{section}||{entry}"
        if entry_key not in stats:
            stats[entry_key] = {"count": 0, "best_rank": rank}
        stats[entry_key]["count"] += 1
        stats[entry_key]["best_rank"] = min(stats[entry_key]["best_rank"], rank)

    if not stats:
        return None

    ranked = sorted(stats.items(), key=lambda kv: (-kv[1]["count"], kv[1]["best_rank"]))
    top_k = min(3, len(ranked))
    candidate_entry_keys = [rk[0] for rk in ranked[:top_k]]

    if not candidate_entry_keys:
        return None

    try:
        all_entries, _ = load_resume_entries_and_embs(resume_id)
    except Exception as e:
        print("[mock] load_resume_entries_and_embs error:", e)
        all_entries = []

    candidate_projects: Dict[str, Dict[str, Any]] = {}
    for entry_key in candidate_entry_keys:
        candidate_projects[entry_key] = {
            "title": entry_key.split("||", 1)[1] if "||" in entry_key else entry_key,
            "bullets": [],
        }

    for e in all_entries:
        section = e.get("section") or "EXPERIENCE"
        entry = e.get("entry") or ""
        if not entry:
            continue
        key = f"{section}||{entry}"
        if key not in candidate_projects:
            continue
        text = e.get("text") or ""
        if text.strip():
            candidate_projects[key]["bullets"].append(text.strip())

    candidate_entry_keys = [k for k in candidate_entry_keys if candidate_projects.get(k, {}).get("bullets")]
    if not candidate_entry_keys:
        return None

    projects_block_lines = []
    for entry_key in candidate_entry_keys:
        proj = candidate_projects[entry_key]
        title = proj["title"]
        bullets = proj["bullets"]
        projects_block_lines.append(f"ID: {entry_key}\nTitle: {title}\nBullets:")
        for bt in bullets:
            projects_block_lines.append(f"- {bt}")
        projects_block_lines.append("")

    projects_block = "\n".join(projects_block_lines)
    
    system_msg = (
        "You are a hiring manager. Choose ONE project from the candidate's resume "
        "that is the best deep-dive topic for this Job Description."
    )
    user_msg = (
        f"JD:\n{jd_text}\n\nProjects:\n{projects_block}\n"
        "Return ONLY the ID of the chosen project."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.0,
        )
        raw_choice = (resp.choices[0].message.content or "").strip()
    except Exception:
        raw_choice = ""

    chosen = None
    for key in candidate_entry_keys:
        if key == raw_choice:
            chosen = key
            break
    if chosen is None:
        raw_norm = raw_choice.strip()
        for key in candidate_entry_keys:
            if key.strip() == raw_norm:
                chosen = key
                break
    if chosen is None:
        chosen = candidate_entry_keys[0]

    return chosen


def create_mock_session(
    profile_id: str,
    resume_id: str,
    mode: str,
    length_type: str,
    hint_level: str,
    focus_category: Optional[str] = None,
    num_questions: Optional[int] = None,
    time_limit: Optional[int] = None,
    interviewer_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    session_id = str(uuid.uuid4())
    question_plan = _build_question_plan(length_type, num_questions, time_limit)
    now = datetime.datetime.utcnow().isoformat()
    num_questions_planned = len(question_plan) if length_type == "questions" else None

    session: Dict[str, Any] = {
        "session_id": session_id,
        "profile_id": profile_id,
        "resume_id": resume_id,
        "mode": mode,
        "length_type": length_type,
        "hint_level": hint_level,
        "focus_category": focus_category,
        "num_questions": num_questions_planned,
        "time_limit": time_limit,
        "question_plan": question_plan,
        "created_at": now,
        "completed": False,
        "used_question_ids": [],
        "used_question_slugs": [],
        "used_question_texts": [],
        "interviewer_profile": interviewer_profile or {},
    }

    primary_project_entry_key = _pick_primary_project_entry(profile_id, resume_id)
    if primary_project_entry_key:
        session["primary_project_entry_key"] = primary_project_entry_key

    path = MOCK_SESSIONS_DIR / f"{session_id}.json"
    with path.open("w", encoding="utf-8") as f:
        _json.dump(session, f, ensure_ascii=False, indent=2)

    return session


def load_mock_session(session_id: str) -> Dict[str, Any]:
    path = MOCK_SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"mock session not found: {session_id}")
    with path.open("r", encoding="utf-8") as f:
        return _json.load(f)


def save_mock_session(session: Dict[str, Any]) -> None:
    session_id = session["session_id"]
    path = MOCK_SESSIONS_DIR / f"{session_id}.json"
    with path.open("w", encoding="utf-8") as f:
        _json.dump(session, f, ensure_ascii=False, indent=2)


# ============================================================
#  題庫：Constants
# ============================================================

INTRO_QUESTION = {
    "question_id": "intro_q1",
    "text": "Hi, thanks for taking the time today. To start, could you give me a brief introduction of yourself?",
    "tag": "Intro · Warm-up",
    "type": "intro",
}

# Auto 和 Follow-up 題庫保留作為 Fallback
AUTO_BANK: List[Dict[str, Any]] = [
    {"id": "auto_fallback_1", "text": "Walk me through one of your favorite projects.", "tag": "Project · Deep dive"},
    {"id": "auto_fallback_2", "text": "Tell me about a technical challenge you faced.", "tag": "Project · Challenge"},
]

FOLLOWUP_BANK: List[Dict[str, Any]] = [
    {"id": "fu_fallback_1", "text": "What was the most challenging part of that?", "tag": "Follow-up"},
    {"id": "fu_fallback_2", "text": "If you could do it again, what would you change?", "tag": "Follow-up"},
]


# ============================================================
#  Interviewer persona builder
# ============================================================

def _build_interviewer_persona(session: Dict[str, Any]) -> str:
    profile = session.get("interviewer_profile") or {}
    role_code = profile.get("role", "senior_engineer")
    style_preset = profile.get("style_preset", "balanced")
    extra = (profile.get("extra_notes") or "").strip()

    ROLE_LABELS = {
        "senior_engineer": "a senior engineer on the team.",
        "hiring_manager": "the hiring manager.",
        "recruiter": "a recruiter.",
        "peer_teammate": "a future teammate.",
        "executive": "a director or VP.",
    }
    role_sentence = ROLE_LABELS.get(role_code, "an interviewer.")

    STYLE_DESCRIPTIONS = {
        "balanced": "Your style is balanced: neutral but probing.",
        "supportive": "Your style is supportive and encouraging.",
        "direct": "Your style is direct and concise.",
        "challenging": "Your style is challenging, asking tough questions.",
        "high_pressure": "Your style is high-pressure and fast-paced.",
    }
    style_sentence = STYLE_DESCRIPTIONS.get(style_preset, "You keep a professional tone.")

    lines = [f"You are {role_sentence}", style_sentence]
    if extra:
        lines.append(f"Note: {extra}")
    return "\n".join(lines)


# ============================================================
#  生成與出題核心邏輯 (Refactored)
# ============================================================

def _generate_non_intro_question(
    session: Dict[str, Any],
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    """
    統一生成邏輯：所有題目類型都優先透過 LLM 生成。
    Behavioral 題目會根據 focus_category 決定 prompt。
    """
    hint_level = session.get("hint_level", "standard")
    used_texts = set(session.get("used_question_texts", []))
    
    q_type = spec["type"]
    profile_id = session.get("profile_id")
    
    # 取得使用者設定的 category
    raw_category = session.get("focus_category")  # e.g., "teamwork", "random", "failure"

    # Interviewer Persona
    persona_text = _build_interviewer_persona(session)
    
    # JD Helper
    def _get_jd_context() -> str:
        jd = _load_profile_jd_for_questions(profile_id)
        if persona_text:
            return f"Interviewer Persona:\n{persona_text}\n\nJob Description:\n{jd}"
        return f"Job Description:\n{jd}"

    # ★ HELPER: 通用 LLM 出題函數
    def _call_llm_gen(system_prompt: str, user_prompt: str) -> str:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[mock] LLM Generation Error ({q_type}): {e}")
            return ""

    text = ""
    tag = ""
    question_id = ""
    entry_key = None
    hints = None

    # ==========================
    # 1. BEHAVIORAL 生成邏輯
    # ==========================
    if q_type == "behavioral":
        # 決定具體的 category
        target_category = raw_category
        
        # 如果是 random 或沒選，從定義好的類別中隨機挑一個
        if not target_category or target_category == "random":
            target_category = random.choice(BEHAVIORAL_CATEGORIES)
        
        # Mapping: 前端傳來的 value -> Prompt 用的描述
        cat_prompt_map = {
            "teamwork": "Teamwork, collaboration, or handling disagreements in a team.",
            "conflict": "Conflict resolution, disagreement with stakeholders or colleagues.",
            "leadership": "Leadership, ownership, influence without authority, or mentorship.",
            "failure_mistakes": "Failure, making mistakes, learning from errors, or handling criticism.",
            "strengths_weaknesses": "Strengths, weaknesses, self-improvement, or self-reflection.",
        }
        
        # 取得 prompt 描述，預設為 General
        cat_desc = cat_prompt_map.get(target_category, "General behavioral interview topics.")

        system_msg = (
            "You are an expert interviewer. Generate ONE behavioral interview question.\n"
            f"Focus Category: {cat_desc}\n"
            "The question must be relevant to the candidate's role (Data/ML) and the provided Job Description."
        )
        user_msg = (
            f"{_get_jd_context()}\n\n"
            f"Please generate a unique question.\n"
            f"Do NOT repeat these questions:\n" + "\n".join(list(used_texts)[:8])
        )

        # 呼叫 LLM
        text = _call_llm_gen(system_msg, user_msg)
        
        # 設置 Tag
        display_cat = target_category.replace("_", " & ").capitalize()
        tag = f"Behavioral · {display_cat}"
        
        # Fallback
        if not text:
            fallback = random.choice(FALLBACK_BEHAVIORAL_BANK)
            text = fallback["text"]
            tag = fallback["tag"]
            
        question_id = f"beh_{target_category}_{uuid.uuid4().hex[:8]}"
        hints = _build_hints_for_generic(hint_level)

    # ==========================
    # 2. AUTO / TECHNICAL / CASE
    # ==========================
    elif q_type in ("auto", "technical", "case"):
        context_text = _get_jd_context()
        if context_text:
            try:
                text = call_llm_for_question(
                    jd_text=context_text,
                    mode=q_type,
                    avoid=used_texts,
                )
            except Exception:
                text = ""
        
        if not text:
            text = "Tell me about a challenging problem you solved recently."
            tag = f"{q_type.capitalize()} (Fallback)"
        else:
            tag = f"{q_type.capitalize()} · JD-Based"
            
        question_id = f"{q_type}_{uuid.uuid4().hex[:8]}"
        
        if q_type == "case":
            hints = _build_hints_for_case(hint_level)
        else:
            hints = _build_hints_for_generic(hint_level)

    # ==========================
    # 3. PROJECT DEEP DIVE
    # ==========================
    elif q_type == "project":
        entry_key = session.get("primary_project_entry_key")
        jd_text = _get_jd_context()
        
        if entry_key and jd_text:
            try:
                resume_id = session.get("resume_id")
                bullets = retrieve_bullets_for_profile(profile_id, entry_key, resume_id, top_k=8)
                entry_title = entry_key.split("||", 1)[1] if "||" in entry_key else entry_key
                text = call_llm_for_project_question(
                    jd_text=jd_text,
                    entry_title=entry_title,
                    bullets=bullets,
                    previous_qas=None
                )
            except Exception:
                text = ""
        
        if not text:
            text = "Walk me through the most significant project on your resume."
            tag = "Project · Deep Dive"
        else:
            tag = "Project · Resume-Based"
            
        question_id = f"proj_{uuid.uuid4().hex[:8]}"
        hints = _build_hints_for_generic(hint_level)

    # ==========================
    # 4. FOLLOW-UP
    # ==========================
    elif q_type == "followup":
        text = spec.get("followup_question_text", "")
        if not text:
            # Fallback
            text = "Could you elaborate more on the outcome?"
        tag = spec.get("followup_tag", "Follow-up")
        question_id = f"fu_{uuid.uuid4().hex[:8]}"
        hints = _build_hints_for_generic(hint_level)

    else:
        # Safety net
        text = "Tell me about yourself."
        tag = "General"
        question_id = f"fallback_{uuid.uuid4().hex[:8]}"
        hints = _build_hints_for_generic(hint_level)


    # --- 儲存紀錄防止重複 ---
    session.setdefault("used_question_ids", [])
    session.setdefault("used_question_texts", [])

    if question_id not in session["used_question_ids"]:
        session["used_question_ids"].append(question_id)
    
    if text and text not in session["used_question_texts"]:
        session["used_question_texts"].append(text)

    result = {
        "question_id": question_id,
        "question": text,
        "tag": tag,
        "hints": hints,
    }
    if entry_key:
        result["entry_key"] = entry_key
        
    return result


# ============================================================
#  Hints builders
# ============================================================

def _build_hints_for_intro(hint_level: str) -> Dict[str, Any]:
    if hint_level == "minimal": return {"show": False}
    return {
        "show": True,
        "bullets": ["Name & Background", "Key Experiences", "Why this role?"],
        "structure": "Past → Present → Future (60-90s)",
        "extra": "Focus on narrative, not just reading the resume."
    }

def _build_hints_for_generic(hint_level: str) -> Dict[str, Any]:
    if hint_level == "minimal": return {"show": False}
    return {
        "show": True,
        "bullets": [],
        "structure": "STAR Method: Situation, Task, Action, Result.",
        "extra": "Focus on YOUR actions and the measurable impact."
    }

def _build_hints_for_case(hint_level: str) -> Dict[str, Any]:
    if hint_level == "minimal": return {"show": False}
    return {
        "show": True,
        "bullets": ["Restate Goal", "List Assumptions", "Approach Steps", "Metrics & Validation"],
        "structure": "Top-down approach.",
        "extra": "Think aloud. Show your reasoning process."
    }


# ============================================================
#  API 呼叫入口: get_question_for_index
# ============================================================

def get_question_for_index(
    session_id: str,
    index: int,
    seconds_left: Optional[int] = None,
) -> Dict[str, Any]:
    session = load_mock_session(session_id)
    plan = session["question_plan"]

    if index < 0 or index >= len(plan):
        return {"done": True, "message": "No more questions."}

    # ★ 緩存機制：檢查是否已經生成過
    current_slot = plan[index]
    if current_slot.get("generated_data"):
        print(f"[mock] Index {index} hit CACHE.")
        return current_slot["generated_data"]

    # 讀取上一題 reaction
    prev_reaction = ""
    answers = session.get("answers") or []
    if index > 0:
        for a in answers:
            if a.get("index") == index - 1:
                prev_reaction = (a.get("reaction") or "").strip()
                break

    spec = dict(plan[index])
    profile_id = session.get("profile_id")

    # 時間模式：剩餘時間少於 5 分鐘強制轉 Behavioral
    if (
        session.get("length_type") == "time"
        and seconds_left is not None
        and seconds_left <= 300
        and spec.get("type") not in ("intro", "behavioral", "followup")
    ):
        spec["type"] = "behavioral"
        plan[index]["type"] = "behavioral"
        session["question_plan"] = plan
        save_mock_session(session)

    # --- 生成題目 ---
    result_data = {}

    if spec["type"] == "intro":
        hints = _build_hints_for_intro(session.get("hint_level", "standard"))
        text = INTRO_QUESTION["text"]
        bullets = []
        if profile_id:
            try:
                resume_id = session.get("resume_id")
                bullets = retrieve_bullets_for_profile(profile_id, text, resume_id, top_k=5)
            except Exception: pass
        
        result_data = {
            "question_id": INTRO_QUESTION["question_id"],
            "question": text,
            "tag": INTRO_QUESTION["tag"],
            "hints": hints,
            "index": index,
            "total": len(plan),
            "bullets": bullets,
            "reaction": prev_reaction,
        }
    else:
        # 呼叫重構後的生成函式
        q = _generate_non_intro_question(session, spec)
        q["index"] = index
        q["total"] = len(plan)
        
        bullets = []
        if profile_id:
            try:
                bullets = retrieve_bullets_for_profile(profile_id, q["question"], resume_id, top_k=5)
            except Exception: pass
        
        q["bullets"] = bullets
        q["reaction"] = prev_reaction
        result_data = q

    # ★ 存入緩存並寫檔
    plan[index]["generated_data"] = result_data
    session["question_plan"] = plan
    save_mock_session(session)

    return result_data


# ============================================================
#  Mock 結算 & 工具 (保持不變)
# ============================================================

def _load_profile_jd(profile_id: str) -> str:
    profile_dir = USER_DATA_DIR / "profiles"
    p = profile_dir / f"{profile_id}.json"
    if not p.exists(): return ""
    try:
        with p.open("r", encoding="utf-8") as f:
            data = _json.load(f)
    except Exception: return ""
    return data.get("jd_text") or data.get("jd") or ""

def finalize_mock_session(session_id: str) -> Dict[str, Any]:
    session = load_mock_session(session_id)
    session["completed"] = True
    session["completed_at"] = datetime.datetime.utcnow().isoformat()
    save_mock_session(session)

    profile_id = session["profile_id"]
    jd_text = _load_profile_jd(profile_id)
    answers = session.get("answers") or []
    
    report_questions = []
    scores = []
    answers_sorted = sorted(answers, key=lambda a: a.get("index", 0))

    for a in answers_sorted:
        answer_text = a.get("transcript", "") or ""
        question_text = a.get("question_text", "")
        idx = a.get("index")

        audio_block = {"has_audio": False, "features": {}, "delivery_score": None, "delivery_comment": ""}
        video_block = {"has_video": False, "features": {}}

        if not answer_text.strip():
            eval_result = {
                "overall_score": None, "subscores": None, "strengths": "",
                "improvements_overview": "No transcript captured.", "improvement_items": [],
                "sample_answer": "", "audio": audio_block
            }
        else:
            audio_wav_path = None
            if idx is not None:
                try:
                    audio_wav_path = extract_wav_from_webm(session_id, idx)
                except Exception: pass

            if idx is not None:
                video_path = MOCK_MEDIA_DIR / f"{session_id}_{idx}.webm"
                if video_path.exists():
                    try:
                        video_block = {"has_video": True, "features": extract_video_features(video_path)}
                    except Exception: pass

            try:
                eval_result = evaluate_answer(
                    question=question_text, jd_text=jd_text, bullets=[],
                    user_answer=answer_text, audio_wav_path=audio_wav_path
                )
            except Exception:
                eval_result = {
                    "overall_score": None, "subscores": None, "strengths": "",
                    "improvements_overview": "Evaluation failed.", "improvement_items": [],
                    "sample_answer": "", "audio": audio_block
                }

        q_score = eval_result.get("overall_score")
        if q_score is None and isinstance(eval_result.get("score"), int):
            q_score = eval_result.get("score")
        if isinstance(q_score, int):
            scores.append(q_score)

        report_questions.append({
            "index": idx,
            "question_id": a.get("question_id"),
            "question_text": question_text,
            "answer_text": answer_text,
            "overall_score": q_score,
            "subscores": eval_result.get("subscores"),
            "strengths": eval_result.get("strengths", ""),
            "improvements_overview": eval_result.get("improvements_overview") or eval_result.get("improvements") or "",
            "improvement_items": eval_result.get("improvement_items") or [],
            "sample_answer": eval_result.get("sample_answer") or "",
            "audio": eval_result.get("audio") or audio_block,
            "video": video_block,
            "score": q_score, # compat
            "improvements": eval_result.get("improvements_overview") or "", # compat
        })

    overall = int(round(sum(scores) / len(scores))) if scores else None
    result_obj = {
        "session_id": session_id,
        "profile_id": profile_id,
        "resume_id": session.get("resume_id"),
        "created_at": session.get("created_at"),
        "completed_at": session.get("completed_at"),
        "length_type": session.get("length_type"),
        "hint_level": session.get("hint_level"),
        "num_questions_planned": session.get("num_questions"),
        "questions": report_questions,
        "overall_score": overall,
    }

    with (MOCK_RESULTS_DIR / f"{session_id}.json").open("w", encoding="utf-8") as f:
        _json.dump(result_obj, f, ensure_ascii=False, indent=2)

    return result_obj


def list_mock_sessions_for_profile(profile_id: str) -> List[Dict[str, Any]]:
    sessions = []
    for p in MOCK_RESULTS_DIR.glob("*.json"):
        with p.open("r", encoding="utf-8") as f:
            data = _json.load(f)
        if data.get("profile_id") == profile_id:
            sessions.append({
                "session_id": data.get("session_id"),
                "created_at": data.get("created_at"),
                "overall_score": data.get("overall_score"),
                "num_questions": len(data.get("questions", [])),
                "length_type": data.get("length_type"),
                "hint_level": data.get("hint_level"),
            })
    sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return sessions


def load_mock_result(session_id: str) -> Dict[str, Any]:
    path = MOCK_RESULTS_DIR / f"{session_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"result not found: {session_id}")
    with path.open("r", encoding="utf-8") as f:
        return _json.load(f)


def _maybe_add_followup_after_answer(session_id: str, answer: Dict[str, Any]) -> None:
    try:
        session = load_mock_session(session_id)
    except FileNotFoundError: return

    plan = session.get("question_plan") or []
    idx = answer.get("index")
    if idx is None or not isinstance(idx, int) or idx < 0 or idx >= len(plan):
        return

    spec = plan[idx]
    if spec.get("type") in ("intro", "followup"):
        return

    current_a = (answer.get("transcript") or "").lower()
    clueless = ["i have no idea", "no idea", "i don't know", "unsure", "not sure"]
    if any(p in current_a for p in clueless):
        return

    # Max 3 follow-ups
    existing = [s for s in plan if s.get("type") == "followup" and s.get("followup_of") == idx]
    if len(existing) >= 3:
        return

    profile_id = session.get("profile_id")
    jd_text = _load_profile_jd_for_questions(profile_id)
    persona_text = _build_interviewer_persona(session)
    
    answers_all = session.get("answers") or []
    history = sorted([a for a in answers_all if isinstance(a.get("index"), int) and a["index"] <= idx], key=lambda a: a["index"])[-3:]
    hist_block = "\n".join([f"Q{h['index']}: {h.get('question_text')}\nA: {h.get('transcript')}" for h in history])

    system_msg = (
        f"You are an interviewer. Decide if a follow-up is needed.\n"
        f"Persona: {persona_text}\n"
    )
    user_msg = (
        f"JD: {jd_text}\n\nHistory:\n{hist_block}\n\nCurrent Q: {answer.get('question_text')}\nCurrent A: {answer.get('transcript')}\n\n"
        "Return JSON: {\"need_followup\": bool, \"question\": str, \"tag\": str}"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.2,
        )
        data = _json.loads((resp.choices[0].message.content or "").strip())
    except Exception: return

    if not data.get("need_followup"): return
    
    fu_q = data.get("question")
    if not fu_q: return

    insert_pos = idx + 1
    plan.insert(insert_pos, {
        "index": insert_pos, "type": "followup", "followup_of": idx,
        "followup_question_text": fu_q, "followup_tag": data.get("tag", "Follow-up")
    })
    for i, s in enumerate(plan): s["index"] = i
    
    session["question_plan"] = plan
    save_mock_session(session)


def generate_interviewer_reaction(question: str, answer: str, session: Optional[Dict[str, Any]] = None) -> str:
    persona_text = _build_interviewer_persona(session) if session else ""
    system_msg = f"You are an interviewer. Give a 1-sentence reaction to the answer. Persona: {persona_text}"
    user_msg = f"Q: {question}\nA: {answer}"
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.6,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception: return ""


def extract_wav_from_webm(session_id: str, index: int) -> Path:
    webm_path = MOCK_MEDIA_DIR / f"{session_id}_{index}.webm"
    if not webm_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {webm_path}")
    wav_path = MOCK_MEDIA_DIR / f"{session_id}_{index}.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(webm_path), "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(wav_path)
    ], check=True)
    return wav_path