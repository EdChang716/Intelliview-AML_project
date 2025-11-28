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
#  Question 設計：題數 / 時間估計 & 題型規劃
# ============================================================

def _estimate_questions_for_time(minutes: int) -> int:
    """
    時間制 mock 的時候，粗估一下題數。
    （只是用來先排出 question_plan，真正結束時間前端可以再用 timer 控制）
    """
    # 先給一個比較大的 upper bound，實際結束由前端 countdown 控制
    return 30  # TODO: 之後可以依 minutes 比較精準估計


def _build_question_plan(
    length_type: str,
    num_questions: Optional[int],
    time_limit: Optional[int],
) -> List[Dict[str, Any]]:
    """
    新版 question_plan：

    - 1 題：intro
    - 2 題：intro → project
    - 3 題：intro → project → case
    - 4 題：intro → project → case → behavioral
    - ≥5 題（正常完整 flow）：
        Q0: intro
        Q1: project deep dive（根據 JD + resume 挑 project）
        Q2: technical（JD-based）
        Q3: case reasoning（跟 JD 對齊的 data/ML/product case）
        中間：technical / auto 題
        最後 1–2 題：behavioral 收尾
    """
    if length_type == "questions":
        total_slots = max(1, num_questions or 5)
    else:
        total_slots = max(1, _estimate_questions_for_time(time_limit or 30))

    plan: List[Dict[str, Any]] = []

    # --- 小題數的特例 ---
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
        # 盡量維持你想要的 flow：intro → project → case → behavioral
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

    # 我們保留最後兩題給 behavioral，如果只有一題空間就留最後一題
    last_behavioral_start = max(4, total_slots - 2)

    # 中間 technical / auto 題（在 case 之後、behavioral 之前）
    idx = 4
    while idx < last_behavioral_start:
        q_type = "technical" if random.random() < 0.7 else "auto"
        plan.append({"index": idx, "type": q_type})
        idx += 1

    # 最後 1–2 題：behavioral
    while idx < total_slots:
        plan.append({"index": idx, "type": "behavioral"})
        idx += 1

    # 保險 re-index
    for i, spec in enumerate(plan):
        spec["index"] = i

    return plan


# ============================================================
#  Session 管理 & project deep dive entry 選擇
# ============================================================

def _load_profile_jd_for_questions(profile_id: Optional[str]) -> str:
    """
    給 technical / auto / project deep dive / case 出題用，從 job_profiles.json 裡找到對應 profile 的 JD 文字。
    """
    if not profile_id:
        return ""

    try:
        profiles = load_job_profiles()  # list[dict]
    except Exception:
        return ""

    profile = next(
        (p for p in profiles if p.get("profile_id") == profile_id),
        None,
    )
    if not profile:
        return ""

    return (
        profile.get("jd_text")
        or profile.get("jd")
        or profile.get("job_description")
        or ""
    )


def _pick_primary_project_entry(profile_id: str, resume_id: str) -> Optional[str]:
    """
    混合策略：
      1) 用 JD 當 query，retrieve 前 10 個最相關 bullet
      2) 從這 10 顆 bullet 裡找出最多 3 個 entry_key (section||entry) 當候選
      3) 讀取整份履歷，把這些候選 entry 底下的所有 bullet 整理出來
      4) 丟給 LLM，請它根據 JD 選出最適合做 primary deep dive 的 entry_key

    回傳：entry_key (例如 "EXPERIENCE||CAYIN Technology – ML Intern")
    """

    # 1) 讀 JD
    jd_text = _load_profile_jd_for_questions(profile_id)
    if not jd_text.strip():
        return None

    # 2) 用 JD 做一次 RAG，只要 top_k=10（前 10 顆最相關 bullet）
    try:
        top_bullets = retrieve_bullets_for_profile(profile_id, jd_text, top_k=10)
    except Exception as e:
        print("[mock] _pick_primary_project_entry retrieve error:", e)
        return None

    if not top_bullets:
        return None

    # 3) 從前 10 顆 bullet 裡統計 entry_key（最多取 3 個）
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

    # 按「出現次數多」優先，其次「best_rank 靠前」排序
    ranked = sorted(
        stats.items(),
        key=lambda kv: (-kv[1]["count"], kv[1]["best_rank"])
    )

    # 只取最多前 3 個 entry_key 當候選（如果本來就只有 1 或 2 個就照實際數量）
    top_k = min(3, len(ranked))
    candidate_entry_keys = [rk[0] for rk in ranked[:top_k]]

    if not candidate_entry_keys:
        return None

    # 4) 載入整份履歷的 bullet，整理出每個 candidate entry_key 對應的完整 bullets
    try:
        all_entries, _ = load_resume_entries_and_embs(resume_id)
    except Exception as e:
        print("[mock] load_resume_entries_and_embs error:", e)
        all_entries = []

    # 建一個 mapping: entry_key -> {"title": ..., "bullets": [..]}
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

    # 把沒有任何 bullet 的候選刪掉（理論上不太會發生）
    candidate_entry_keys = [
        k for k in candidate_entry_keys
        if candidate_projects.get(k, {}).get("bullets")
    ]
    if not candidate_entry_keys:
        return None

    # 5) 用 LLM 在這幾個候選中選出「最適合 deep dive 的 project」
    projects_block_lines = []
    for entry_key in candidate_entry_keys:
        proj = candidate_projects[entry_key]
        title = proj["title"]
        bullets = proj["bullets"]
        projects_block_lines.append(f"ID: {entry_key}\nTitle: {title}\nBullets:")
        for bt in bullets:
            projects_block_lines.append(f"- {bt}")
        projects_block_lines.append("")  # 空行分隔

    projects_block = "\n".join(projects_block_lines)

    system_msg = (
        "You are a hiring manager preparing for a data / ML interview.\n"
        "Given a job description and several projects from the candidate's resume, "
        "choose ONE project that is the best primary deep-dive topic for this interview."
    )

    user_msg = (
        "Job description:\n"
        f"{jd_text}\n\n"
        "Here are candidate projects from the resume. Each project has an ID, a title, and its bullets:\n\n"
        f"{projects_block}\n"
        "Your task:\n"
        "- Choose EXACTLY ONE project that is the best fit to deep-dive on for this job.\n"
        "- Prefer projects that (1) match the tools and responsibilities in the job, "
        "(2) show end-to-end ownership, and (3) have clear impact or measurable results.\n\n"
        "Return ONLY the ID of the chosen project (exactly one of the IDs shown above), with no explanation."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
        raw_choice = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("[mock] LLM choose primary project error:", e)
        raw_choice = ""

    # 確保回傳的是候選裡的一個；若 LLM 回傳怪怪的就 fallback 第一個
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

    print("[mock] picked primary_project_entry_key (hybrid) =", chosen)
    return chosen


def create_mock_session(
    profile_id: str,
    resume_id: str,
    mode: str,
    length_type: str,
    hint_level: str,
    num_questions: Optional[int] = None,
    time_limit: Optional[int] = None,
    interviewer_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    從 mock settings 頁建立一個新的 mock session，存成 json。

    回傳 session dict，裡面會有：
    - session_id
    - num_questions（實際會被問到的 slot 數，包括 follow-ups）
    - interviewer_profile：面試官 persona 設定
    """
    session_id = str(uuid.uuid4())

    question_plan = _build_question_plan(
        length_type=length_type,
        num_questions=num_questions,
        time_limit=time_limit,
    )

    now = datetime.datetime.utcnow().isoformat()

    # 問題模式才固定 main 題數；時間模式交給前端倒數計時控制
    if length_type == "questions":
        num_questions_planned = len(question_plan)
    else:
        num_questions_planned = None  # 不用這個欄位來截斷

    # 先建立基本 session
    session: Dict[str, Any] = {
        "session_id": session_id,
        "profile_id": profile_id,
        "resume_id": resume_id,
        "mode": mode,
        "length_type": length_type,
        "hint_level": hint_level,
        "num_questions": num_questions_planned,
        "time_limit": time_limit,          # 這個等等要丟給前端
        "question_plan": question_plan,
        "created_at": now,
        "completed": False,
        "used_question_ids": [],
        "used_question_slugs": [],
        # ★ NEW: 記錄問過的題目文字，讓 technical / auto / case 題可以避免重複
        "used_question_texts": [],
        # ★ NEW: interviewer persona
        "interviewer_profile": interviewer_profile or {},
    }

    # ★ NEW: 建立 project deep dive 用的 primary_project_entry_key
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
#  題庫：behavioral / auto / follow-up（technical/auto/case 用 LLM + JD）
# ============================================================

INTRO_QUESTION = {
    "question_id": "intro_q1",
    "slug": "intro_self_intro",
    "text": (
        "Hi, thanks for taking the time today. "
        "To start, could you give me a brief introduction of yourself?"
    ),
    "tag": "Intro · Warm-up",
    "type": "intro",
}

# ★ Behavioral 題庫
BEHAVIORAL_BANK: List[Dict[str, Any]] = [
    {
        "id": "beh_lead_team",
        "slug": "leadership_lead_team",
        "category": "leadership",
        "text": "Tell me about a time you had to take the lead on a project.",
        "tag": "Behavioral · Leadership",
    },
    {
        "id": "beh_ownership",
        "slug": "ownership_end_to_end",
        "category": "ownership",
        "text": "Describe a situation where you took end-to-end ownership of a problem or project.",
        "tag": "Behavioral · Ownership",
    },
    {
        "id": "beh_conflict_teammate",
        "slug": "conflict_teammate",
        "category": "conflict",
        "text": "Describe a time when you disagreed with a teammate. How did you handle it?",
        "tag": "Behavioral · Conflict",
    },
    {
        "id": "beh_conflict_stakeholder",
        "slug": "conflict_stakeholder",
        "category": "conflict",
        "text": "Tell me about a time you had to push back on a stakeholder or manager.",
        "tag": "Behavioral · Stakeholder management",
    },
    {
        "id": "beh_failure",
        "slug": "failure_learn",
        "category": "failure",
        "text": "Tell me about a time you failed at something important. What did you learn?",
        "tag": "Behavioral · Failure",
    },
    {
        "id": "beh_ambiguity",
        "slug": "ambiguity_unstructured",
        "category": "ambiguity",
        "text": "Describe a time you had to work through an ambiguous or poorly defined problem.",
        "tag": "Behavioral · Ambiguity",
    },
    {
        "id": "beh_teamwork",
        "slug": "teamwork_collaboration",
        "category": "teamwork",
        "text": "Tell me about a time you worked closely with others to achieve a goal.",
        "tag": "Behavioral · Teamwork",
    },
    {
        "id": "beh_time_pressure",
        "slug": "time_pressure_deadlines",
        "category": "time_management",
        "text": "Give me an example of a time you were under a lot of time pressure. How did you prioritize and execute?",
        "tag": "Behavioral · Time management",
    },
    {
        "id": "beh_feedback",
        "slug": "feedback_receive",
        "category": "feedback",
        "text": "Tell me about a time you received critical feedback. What did you do afterward?",
        "tag": "Behavioral · Feedback",
    },
    {
        "id": "beh_communication",
        "slug": "communication_non_technical",
        "category": "communication",
        "text": "Describe a time you had to explain a complex or technical topic to a non-technical audience.",
        "tag": "Behavioral · Communication",
    },
]

# auto bank 保留作為 fallback（當 LLM / JD 出問題時再用）
AUTO_BANK: List[Dict[str, Any]] = [
    {
        "id": "auto_project_favorite",
        "slug": "project_favorite_ds",
        "category": "project",
        "text": "Walk me through one of your favorite data or ML projects on your resume.",
        "tag": "Project · Deep dive",
    },
    {
        "id": "auto_project_challenging",
        "slug": "project_most_challenging",
        "category": "project",
        "text": "Tell me about the most technically challenging project you’ve worked on.",
        "tag": "Project · Challenge",
    },
    {
        "id": "auto_impact",
        "slug": "impact_business",
        "category": "impact",
        "text": "Give me an example of a project where your work had a clear impact on business or users.",
        "tag": "Impact · Results",
    },
    {
        "id": "auto_metrics",
        "slug": "metrics_evaluation",
        "category": "metrics",
        "text": "Tell me about a time you had to define or choose metrics to evaluate your solution.",
        "tag": "Analytics · Metrics",
    },
]

FOLLOWUP_BANK: List[Dict[str, Any]] = [
    {
        "id": "fu_deeper_challenge",
        "slug": "followup_challenge",
        "text": "What was the most challenging part of that situation, and how did you handle it in the moment?",
    },
    {
        "id": "fu_alt_decision",
        "slug": "followup_alternative",
        "text": "If you faced a similar situation again, what would you do differently?",
        "tag": "Follow-up · Reflection",
    },
    {
        "id": "fu_stakeholder",
        "slug": "followup_stakeholder",
        "text": "How did the other people involved react, and how did you manage those reactions?",
        "tag": "Follow-up · Stakeholders",
    },
    {
        "id": "fu_impact_details",
        "slug": "followup_impact_details",
        "text": "Can you go a bit deeper into the impact? How did you know your approach was successful?",
        "tag": "Follow-up · Impact",
    },
]


# ============================================================
#  Interviewer persona builder
# ============================================================

def _build_interviewer_persona(session: Dict[str, Any]) -> str:
    """
    從 session['interviewer_profile'] 組一段簡短 persona 描述，
    給 LLM 出題 / follow-up / reaction 用。
    """
    profile = session.get("interviewer_profile") or {}

    gender = profile.get("gender", "auto")
    role_code = profile.get("role", "senior_engineer")
    role_custom = (profile.get("role_custom") or "").strip()
    style_preset = profile.get("style_preset", "balanced")
    style_custom = (profile.get("style_custom") or "").strip()
    extra = (profile.get("extra_notes") or "").strip()

    # Role 描述
    ROLE_LABELS = {
        "senior_engineer": "a senior engineer on the team, focusing on technical depth and collaboration.",
        "hiring_manager": "the hiring manager, balancing technical depth with team fit, ownership, and impact.",
        "recruiter": "a recruiter or HR partner, focusing on communication, motivation, and culture fit.",
        "peer_teammate": "a future teammate, curious about how this candidate works with others and solves problems day-to-day.",
        "executive": "a director or VP, focusing on high-level impact, business value, and alignment with the org’s goals.",
    }

    if role_code == "custom" and role_custom:
        role_sentence = role_custom
    else:
        role_sentence = ROLE_LABELS.get(
            role_code,
            "a realistic interviewer for this role.",
        )

    # Style 描述
    STYLE_DESCRIPTIONS = {
        "balanced": "Your style is balanced: neutral but probing, professional, and fair.",
        "supportive": "Your style is supportive: encouraging, patient, and helping the candidate feel comfortable while still asking thoughtful questions.",
        "direct": "Your style is direct: concise, straightforward, and to the point, but not rude.",
        "challenging": "Your style is challenging: you push on weak spots and ask tough questions, but remain professional.",
        "high_pressure": "Your style is high-pressure: you move quickly, occasionally skeptical, and simulate a demanding interview, while still being fair.",
    }

    if style_preset == "custom" and style_custom:
        style_sentence = style_custom
    else:
        style_sentence = STYLE_DESCRIPTIONS.get(
            style_preset,
            "You keep a realistic but fair interview tone.",
        )

    # Gender / voice 只當成氛圍，不影響內容
    gender_bits = []
    if gender == "male":
        gender_bits.append("You sound like a male interviewer.")
    elif gender == "female":
        gender_bits.append("You sound like a female interviewer.")
    elif gender == "neutral":
        gender_bits.append("Your voice and style feel gender-neutral and professional.")

    if extra:
        gender_bits.append(f"Additional notes about your style: {extra}")

    lines = [
        "You are interviewing a candidate for this role.",
        f"Your interviewer persona: {role_sentence}",
        style_sentence,
    ]
    if gender_bits:
        lines.extend(gender_bits)

    return "\n".join(lines)


# ============================================================
#  題目選取：避免重複 & follow-up
# ============================================================

def _pick_from_bank(
    bank: List[Dict[str, Any]],
    used_ids: set[str],
    used_slugs: set[str],
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    從題庫裡挑一題，盡量避開已使用的 slug（避免語意近似的重複）。
    - 先找「同 category 且 slug 未使用」的
    - 若找不到，再退而求其次找「同 category 任意題」
    - 若還是沒有，再從整個 bank 任意選
    """
    candidates = [
        q for q in bank
        if (category is None or q.get("category") == category)
        and q["slug"] not in used_slugs
    ]
    if not candidates and category is not None:
        candidates = [q for q in bank if q.get("category") == category]
    if not candidates:
        candidates = bank[:]  # 最後保底：隨機一題，但仍避免重複 id
    random.shuffle(candidates)
    for q in candidates:
        if q["id"] not in used_ids:
            return q
    # 萬一全部都用過，就真的隨便選一題
    return random.choice(bank)


def _generate_non_intro_question(
    session: Dict[str, Any],
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    """
    回傳:
    {
      "question_id": "...",
      "question": "...",
      "tag": "...",
      "hints": {...},
      "entry_key": "...",   # 只有 project deep dive 會有，其它 type 可省略
    }
    """
    hint_level = session.get("hint_level", "standard")
    used_ids = set(session.get("used_question_ids", []))
    used_slugs = set(session.get("used_question_slugs", []))
    used_texts = set(session.get("used_question_texts", []))

    q_type = spec["type"]
    profile_id = session.get("profile_id")

    # interviewer persona：給 LLM 用
    persona_text = _build_interviewer_persona(session)

    def _attach_persona_to_jd(jd_text: str) -> str:
        """
        不改 call_llm_for_question 的 interface，
        改成把 persona prepend 到 jd_text 前面一起丟進去。
        """
        jd_text = jd_text or ""
        if not persona_text.strip():
            return jd_text
        return (
            "Interviewer persona:\n"
            f"{persona_text}\n\n"
            "Job description and context:\n"
            f"{jd_text}"
        )

    text = ""
    tag = ""
    question_id = ""
    entry_key: Optional[str] = None
    hints: Optional[Dict[str, Any]] = None

    if q_type == "behavioral":
        base = _pick_from_bank(
            BEHAVIORAL_BANK,
            used_ids,
            used_slugs,
            category=None,  # 之後可以看需要讓你選某一類
        )
        question_id = base["id"]
        text = base["text"]
        tag = base["tag"]
        entry_key = None
        hints = _build_hints_for_generic(hint_level)

    elif q_type == "auto":
        # auto 題：JD-based LLM 出題，fallback 才用 AUTO_BANK
        jd_text = _load_profile_jd_for_questions(profile_id)
        jd_for_llm = _attach_persona_to_jd(jd_text)
        if jd_for_llm.strip():
            try:
                question_text = call_llm_for_question(
                    jd_text=jd_for_llm,
                    mode="auto",
                    avoid=used_texts,
                )
                text = (question_text or "").strip()
            except Exception as e:
                print("[mock auto] LLM error:", e)

        if text:
            question_id = f"auto_{spec['index']}_{uuid.uuid4().hex[:8]}"
            tag = "General · JD-based"
        else:
            # fallback
            base = _pick_from_bank(AUTO_BANK, used_ids, used_slugs, category=None)
            question_id = base["id"]
            text = base["text"]
            tag = base["tag"]
        entry_key = None
        hints = _build_hints_for_generic(hint_level)

    elif q_type == "technical":
        # technical 題目用 JD + LLM 動態產生
        jd_text = _load_profile_jd_for_questions(profile_id)
        jd_for_llm = _attach_persona_to_jd(jd_text)
        print("[mock technical] profile_id=", profile_id, "JD length=", len(jd_text))

        if jd_for_llm.strip():
            try:
                question_text = call_llm_for_question(
                    jd_text=jd_for_llm,
                    mode="technical",
                    avoid=used_texts,
                )
                text = (question_text or "").strip()
            except Exception as e:
                print("[mock technical] LLM error:", e)

        if not text:
            # safety fallback：給一題 generic technical
            text = (
                "Let’s talk about a technical challenge you recently solved. "
                "Could you walk me through the problem, your approach, and the impact?"
            )
        question_id = f"tech_{spec['index']}_{uuid.uuid4().hex[:8]}"
        tag = "Technical · JD-based"
        entry_key = None
        hints = _build_hints_for_generic(hint_level)

    elif q_type == "case":
        # case reasoning 題：JD-based LLM 出題
        jd_text = _load_profile_jd_for_questions(profile_id)
        jd_for_llm = _attach_persona_to_jd(jd_text)
        print("[mock case] profile_id=", profile_id, "JD length=", len(jd_text))

        if jd_for_llm.strip():
            try:
                question_text = call_llm_for_question(
                    jd_text=jd_for_llm,
                    mode="case",
                    avoid=used_texts,
                )
                text = (question_text or "").strip()
            except Exception as e:
                print("[mock case] LLM error:", e)

        if not text:
            # safety fallback：generic case prompt
            text = (
                "Imagine you joined our team as a data scientist. "
                "How would you design an end-to-end approach to identify and prioritize opportunities "
                "to improve a key business metric? Talk through assumptions, data, modeling, and how "
                "you’d measure success."
            )

        question_id = f"case_{spec['index']}_{uuid.uuid4().hex[:8]}"
        tag = "Case · Reasoning"
        entry_key = None
        hints = _build_hints_for_case(hint_level)

    elif q_type == "project":
        # project deep dive，根據 primary_project_entry_key + JD 做 LLM 問題
        jd_text = _load_profile_jd_for_questions(profile_id)
        jd_for_llm = _attach_persona_to_jd(jd_text)
        entry_key = session.get("primary_project_entry_key")
        resume_id = session.get("resume_id")

        if jd_for_llm.strip() and entry_key and resume_id:
            try:
                # 用 entry_key 當 query，讓 RAG 找出對應 bullets
                bullets = retrieve_bullets_for_profile(profile_id, entry_key, top_k=8)
            except Exception as e:
                print("[mock project] retrieve error:", e)
                bullets = []

            # entry title 從 entry_key 拆
            if "||" in entry_key:
                entry_title = entry_key.split("||", 1)[1]
            else:
                entry_title = entry_key

            try:
                question_text = call_llm_for_project_question(
                    jd_text=jd_for_llm,
                    entry_title=entry_title,
                    bullets=bullets,
                    previous_qas=None,  # mock 模式先不帶 history
                )
                text = (question_text or "").strip()
            except Exception as e:
                print("[mock project] LLM error:", e)

        if not text:
            # fallback：用 auto 題庫的一題 project 類
            base = _pick_from_bank(AUTO_BANK, used_ids, used_slugs, category="project")
            question_id = base["id"]
            text = base["text"]
            tag = base["tag"]
            entry_key = entry_key
        else:
            question_id = f"proj_{spec['index']}_{uuid.uuid4().hex[:8]}"
            tag = "Project · Deep dive (JD-based)"
        hints = _build_hints_for_generic(hint_level)

    elif q_type == "followup":
        # 優先吃 spec 裡 LLM 產好的追問
        custom_text = spec.get("followup_question_text")
        custom_tag = spec.get("followup_tag")

        if custom_text:
            question_id = f"fu_{spec['index']}_{uuid.uuid4().hex[:8]}"
            text = custom_text.strip()
            tag = custom_tag or "Follow-up"
            entry_key = None
            # follow-up 通常不需要太多提示，這裡仍用 generic 統一格式
            hints = _build_hints_for_generic(hint_level)
        else:
            # fallback：舊的隨機追問題庫
            base = _pick_from_bank(FOLLOWUP_BANK, used_ids, used_slugs, category=None)
            question_id = base["id"]
            parent_idx = spec.get("followup_of")
            tag = "Follow-up"
            if isinstance(parent_idx, int):
                tag = f"Follow-up to Q{parent_idx + 1}"
            text = base["text"]
            entry_key = None
            hints = _build_hints_for_generic(hint_level)

    else:
        # safety net
        question_id = f"fallback_{spec['index']}"
        text = "Tell me about a time you had to solve a difficult problem."
        tag = "Behavioral · Problem solving"
        entry_key = None
        hints = _build_hints_for_generic(hint_level)

    # 更新 session 的 used_question_ids / slugs / texts
    session.setdefault("used_question_ids", [])
    session.setdefault("used_question_slugs", [])
    session.setdefault("used_question_texts", [])

    if question_id and question_id not in session["used_question_ids"]:
        session["used_question_ids"].append(question_id)

    slug = None
    for bank in (BEHAVIORAL_BANK, AUTO_BANK, FOLLOWUP_BANK):
        for q in bank:
            if q["id"] == question_id:
                slug = q.get("slug")
                break
        if slug is not None:
            break
    if slug and slug not in session["used_question_slugs"]:
        session["used_question_slugs"].append(slug)

    if text and text not in session["used_question_texts"]:
        session["used_question_texts"].append(text)

    save_mock_session(session)

    result = {
        "question_id": question_id,
        "question": text,
        "tag": tag,
        "hints": hints or _build_hints_for_generic(hint_level),
    }
    if entry_key:
        result["entry_key"] = entry_key
    return result


# ============================================================
#  Hints builders
# ============================================================

def _build_hints_for_intro(hint_level: str) -> Dict[str, Any]:
    if hint_level == "minimal":
        return {"show": False}
    bullets = [
        "Name, current program / role, and background (e.g., data science student at Columbia).",
        "1–2 key experiences relevant to this role (e.g., RA, internship, main projects).",
        "Wrap up with what you're looking for and why this role / company.",
    ]
    structure = "Think of a 60–90 second elevator pitch: past → present → future."
    extra = (
        "Avoid reading your resume line by line; focus on a clear narrative and what makes you a good fit."
    )
    return {
        "show": True,
        "bullets": bullets,
        "structure": structure,
        "extra": extra if hint_level == "full" else "",
    }


def _build_hints_for_generic(hint_level: str) -> Dict[str, Any]:
    if hint_level == "minimal":
        return {"show": False}
    structure = "Use STAR: briefly set the situation, explain your task, list 2–3 concrete actions, and end with a measurable result or takeaway."
    extra = (
        "Focus on your decisions and reasoning, not just what the team did. "
        "Tie the outcome back to impact on metrics, users, or stakeholders."
    )
    return {
        "show": True,
        "bullets": [],
        "structure": structure,
        "extra": extra if hint_level == "full" else "",
    }


def _build_hints_for_case(hint_level: str) -> Dict[str, Any]:
    if hint_level == "minimal":
        return {"show": False}

    bullets = [
        "Start by restating the goal in your own words and clarifying what success looks like.",
        "List your key assumptions explicitly (about users, data availability, constraints).",
        "Outline your approach in clear steps instead of jumping into details immediately.",
        "Call out the metrics you would monitor and how you would validate your solution.",
    ]
    structure = (
        "Use a top-down structure: goal → assumptions → high-level plan → metrics and validation → trade-offs / risks."
    )
    extra = (
        "Think aloud as you reason through the case. It's better to show your reasoning process clearly "
        "than to jump to a final answer without explaining how you got there."
    )

    return {
        "show": True,
        "bullets": bullets,
        "structure": structure,
        "extra": extra if hint_level == "full" else "",
    }


# ============================================================
#  給 API 用：依 index 取出當題題目
# ============================================================

def get_question_for_index(
    session_id: str,
    index: int,
    seconds_left: Optional[int] = None,  # 時間模式：剩餘秒數
) -> Dict[str, Any]:
    session = load_mock_session(session_id)
    plan = session["question_plan"]

    if index < 0 or index >= len(plan):
        return {
            "done": True,
            "message": "No more questions in this mock interview.",
        }

    # ---------- 讀上一題的 reaction ----------
    prev_reaction = ""
    answers = session.get("answers") or []
    if index > 0:
        for a in answers:
            if a.get("index") == index - 1:
                prev_reaction = (a.get("reaction") or "").strip()
                break

    # 先拿原本腳本裡的 spec
    spec = dict(plan[index])  # 做一份 copy 不直接動原 dict
    profile_id = session.get("profile_id")

    # ===== 時間模式專用邏輯：剩下 < 5 分鐘 → 強制 behavioral =====
    if (
        session.get("length_type") == "time"
        and seconds_left is not None
        and seconds_left <= 300  # 5 分鐘 = 300 秒
        and spec.get("type") not in ("intro", "behavioral", "followup")
    ):
        spec["type"] = "behavioral"
        plan[index]["type"] = "behavioral"
        session["question_plan"] = plan
        save_mock_session(session)

    # ===== 根據 type 出題 =====
    if spec["type"] == "intro":
        hints = _build_hints_for_intro(session.get("hint_level", "standard"))
        question_text = INTRO_QUESTION["text"]

        bullets = []
        if profile_id:
            try:
                bullets = retrieve_bullets_for_profile(profile_id, question_text, top_k=5)
            except Exception as e:
                print("[mock] retrieve_bullets_for_profile error:", e)

        return {
            "question_id": INTRO_QUESTION["question_id"],
            "question": question_text,
            "tag": INTRO_QUESTION["tag"],
            "hints": hints,
            "index": index,
            "total": len(plan),
            "bullets": bullets,
            "reaction": prev_reaction,  # 👈 通常 intro = 第 0 題，這裡多半是空字串
        }
    else:
        q = _generate_non_intro_question(session, spec)
        q["index"] = index
        q["total"] = len(plan)

        bullets = []
        if profile_id:
            try:
                bullets = retrieve_bullets_for_profile(profile_id, q["question"], top_k=5)
            except Exception as e:
                print("[mock] retrieve_bullets_for_profile error:", e)

        q["bullets"] = bullets
        q["reaction"] = prev_reaction  # 👈 把上一題 reaction 附在這題的 payload 裡
        return q


# ============================================================
#  完成 mock：Whisper 轉錄 + 切段 + 評分
# ============================================================

def _load_profile_jd(profile_id: str) -> str:
    """
    嘗試讀取 profile 的 JD 文字，給評分用。
    ⚠️ 如果你的專案路徑或 key 名稱不同，請改這一段。
    """
    profile_dir = USER_DATA_DIR / "profiles"
    p = profile_dir / f"{profile_id}.json"
    if not p.exists():
        return ""
    try:
        with p.open("r", encoding="utf-8") as f:
            data = _json.load(f)
    except Exception:
        return ""
    return (
        data.get("jd_text")
        or data.get("jd")
        or data.get("job_description")
        or ""
    )


def finalize_mock_session(session_id: str) -> Dict[str, Any]:
    """
    使用前面 /api/mock_answer 已經存好的 per-question transcript，
    統一做一次評分 + 報表輸出。
    現在會同時嘗試讀取該題的錄影檔，抽出音訊做 audio 評估，
    並針對 video 計算一些簡單的視覺特徵。
    """
    session = load_mock_session(session_id)
    session["completed"] = True
    session["completed_at"] = datetime.datetime.utcnow().isoformat()
    save_mock_session(session)

    profile_id = session["profile_id"]
    try:
        jd_text = _load_profile_jd(profile_id)
    except Exception as e:
        print("[finalize_mock_session] load JD error:", e)
        jd_text = ""

    answers = session.get("answers") or []

    report_questions: List[Dict[str, Any]] = []
    scores: List[int] = []

    # 照 index 排一下
    answers_sorted = sorted(answers, key=lambda a: a.get("index", 0))

    for a in answers_sorted:
        answer_text = a.get("transcript", "") or ""
        question_text = a.get("question_text", "")
        idx = a.get("index")

        # 預設 audio 區塊（就算失敗也有東西）
        audio_block: Dict[str, Any] = {
            "has_audio": False,
            "features": {},
            "delivery_score": None,
            "delivery_comment": "",
        }

        # 預設 video 區塊
        video_block: Dict[str, Any] = {
            "has_video": False,
            "features": {},
        }

        if not answer_text.strip():
            # 沒有 transcript 的情況
            eval_result = {
                "overall_score": None,
                "subscores": None,
                "strengths": "",
                "improvements_overview": "No transcript was captured for this question.",
                "improvement_items": [],
                "sample_answer": "",
                "audio": audio_block,
            }
        else:
            # 嘗試抽出該題的 .wav 路徑
            audio_wav_path: Optional[Path] = None
            if idx is not None:
                try:
                    audio_wav_path = extract_wav_from_webm(session_id, idx)
                except Exception as e:
                    # 抽音檔失敗不影響整體流程，只是沒 audio 評估
                    print(
                        f"[finalize_mock_session] extract_wav_from_webm error for session={session_id}, idx={idx}:",
                        e,
                    )
                    audio_wav_path = None

            # 嘗試計算 video features
            if idx is not None:
                video_path = MOCK_MEDIA_DIR / f"{session_id}_{idx}.webm"
                if video_path.exists():
                    try:
                        v_feats = extract_video_features(video_path)
                        video_block = {
                            "has_video": True,
                            "features": v_feats,
                        }
                    except Exception as e:
                        print(
                            f"[finalize_mock_session] extract_video_features error for session={session_id}, idx={idx}:",
                            e,
                        )

            try:
                eval_result = evaluate_answer(
                    question=question_text,
                    jd_text=jd_text,
                    bullets=[],  # 之後如果要也可以在這邊加 RAG
                    user_answer=answer_text,
                    audio_wav_path=audio_wav_path,
                )
            except Exception as e:
                print("[finalize_mock_session] evaluate_answer error:", e)
                eval_result = {
                    "overall_score": None,
                    "subscores": None,
                    "strengths": "",
                    "improvements_overview": "Automatic evaluation failed for this question.",
                    "improvement_items": [],
                    "sample_answer": "",
                    "audio": audio_block,
                }

        # ---- 取整體分數（兼容舊欄位） ----
        q_overall_score = eval_result.get("overall_score")
        if q_overall_score is None and isinstance(eval_result.get("score"), int):
            q_overall_score = eval_result.get("score")

        if isinstance(q_overall_score, int):
            scores.append(q_overall_score)

        subscores = eval_result.get("subscores")
        strengths = eval_result.get("strengths", "") or ""
        improvements_overview = (
            eval_result.get("improvements_overview")
            or eval_result.get("improvements")
            or ""
        )
        improvement_items = eval_result.get("improvement_items") or []
        sample_answer = eval_result.get("sample_answer") or ""

        # audio 區塊（evaluate_answer 已經會帶 audio 回來）
        audio_block = eval_result.get("audio") or audio_block

        report_questions.append(
            {
                "index": idx,
                "question_id": a.get("question_id"),
                "question_text": question_text,
                "answer_text": answer_text,

                # 新版欄位
                "overall_score": q_overall_score,
                "subscores": subscores,
                "strengths": strengths,
                "improvements_overview": improvements_overview,
                "improvement_items": improvement_items,
                "sample_answer": sample_answer,
                "audio": audio_block,   # 每題的 audio 評估
                "video": video_block,   # 每題的 video 特徵

                # 舊版欄位（給還沒改掉的 template / 前端用）
                "score": q_overall_score,
                "improvements": improvements_overview,
            }
        )

    overall_score = int(round(sum(scores) / len(scores))) if scores else None

    result_obj: Dict[str, Any] = {
        "session_id": session_id,
        "profile_id": profile_id,
        "resume_id": session.get("resume_id"),
        "created_at": session.get("created_at"),
        "completed_at": session.get("completed_at"),
        "length_type": session.get("length_type"),
        "hint_level": session.get("hint_level"),
        "num_questions_planned": session.get("num_questions"),
        "questions": report_questions,
        "overall_score": overall_score,
    }

    out_path = MOCK_RESULTS_DIR / f"{session_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        _json.dump(result_obj, f, ensure_ascii=False, indent=2)

    return result_obj



# ============================================================
#  History / report 用的小工具
# ============================================================

def list_mock_sessions_for_profile(profile_id: str) -> List[Dict[str, Any]]:
    """
    給 history 頁面用，簡單列出這個 profile 的所有 mock summary。
    （直接讀 results 裡面的 report.json）
    """
    sessions: List[Dict[str, Any]] = []
    for p in MOCK_RESULTS_DIR.glob("*.json"):
        with p.open("r", encoding="utf-8") as f:
            data = _json.load(f)
        if data.get("profile_id") == profile_id:
            # 簡化 summary 給前端用
            sessions.append(
                {
                    "session_id": data.get("session_id"),
                    "created_at": data.get("created_at"),
                    "overall_score": data.get("overall_score"),
                    "num_questions": len(data.get("questions", [])),
                    "length_type": data.get("length_type"),
                    "hint_level": data.get("hint_level"),
                }
            )
    sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return sessions


def load_mock_result(session_id: str) -> Dict[str, Any]:
    path = MOCK_RESULTS_DIR / f"{session_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"mock result not found: {session_id}")
    with path.open("r", encoding="utf-8") as f:
        return _json.load(f)


# ============================================================
#  LLM-based follow-up 插入邏輯
# ============================================================

def _maybe_add_followup_after_answer(session_id: str, answer: Dict[str, Any]) -> None:
    """
    在某一題回答之後，讓 LLM 判斷要不要插入 follow-up。
    - 只針對非 intro / followup 題
    - follow-up 會插在當前題的下一格，並重新 re-index question_plan
    - 同一個主問題最多插入 3 題 follow-up（不能連續出現 4 題）
    - 如果使用者明確表示「不知道 / 沒想法」，就不要追問
    - used in main.py: @app.post("/api/mock_answer")
    """
    try:
        session = load_mock_session(session_id)
    except FileNotFoundError:
        return

    plan = session.get("question_plan") or []
    if not plan:
        return

    idx = answer.get("index")
    if idx is None or not isinstance(idx, int):
        return
    if idx < 0 or idx >= len(plan):
        return

    spec = plan[idx]
    q_type = spec.get("type")

    # 不要對 intro 自我介紹 / 已經是 followup 再追問
    if q_type in ("intro", "followup"):
        return

    # ---- 先看答案內容：如果明確表示不知道，就不要追問 ----
    current_a = (answer.get("transcript") or "").lower()
    # 可以再視情況加更多關鍵字
    clueless_phrases = [
        "i have no idea",
        "no idea",
        "i don't know",
        "i dont know",
        "i dunno",
        "i am not sure",
        "i'm not sure",
        "i am unsure",
        "i'm unsure",
        "haven't done this before",
        "have not done this before",
        "no experience with this",
        "i don't have experience",
    ]
    if any(p in current_a for p in clueless_phrases):
        # 這種回答就不要再往死裡追問，直接結束
        return

    # ---- 同一個主問題最多 3 題 follow-up ----
    existing_fus_for_this = [
        s for s in plan
        if s.get("type") == "followup" and s.get("followup_of") == idx
    ]
    if len(existing_fus_for_this) >= 3:
        # 已經有三題追問了，就不要再插第四題
        return

    profile_id = session.get("profile_id")
    jd_text = _load_profile_jd_for_questions(profile_id)

    # interviewer persona
    persona_text = _build_interviewer_persona(session)
    persona_block = ""
    if persona_text.strip():
        persona_block = (
            "Here is your interviewer persona. Stay consistent with this:\n"
            f"{persona_text}\n\n"
        )

    # 準備 history：拿目前為止（含當前）的 QA，最多 3 題
    answers_all = session.get("answers") or []
    history = [
        a for a in answers_all
        if isinstance(a.get("index"), int) and a["index"] <= idx
    ]
    history = sorted(history, key=lambda a: a["index"])
    history = history[-3:]  # 只保留最後 3 題

    history_lines = []
    for h in history:
        qi = h.get("index")
        qtxt = h.get("question_text") or ""
        atxt = h.get("transcript") or ""
        history_lines.append(f"Q{qi}: {qtxt}\nA{qi}: {atxt}\n")

    history_block = "\n".join(history_lines)

    current_q = answer.get("question_text") or ""
    current_a_full = answer.get("transcript") or ""

    system_msg = (
        "You are a structured, realistic interviewer for data/ML roles.\n"
        "You decide whether to ask a short, focused follow-up question after the candidate's answer.\n"
        "You only ask a follow-up when there is a clear gap, ambiguity, or interesting detail to explore.\n"
        "Roughly half of the time, you should decide that no follow-up is needed.\n"
        "If the candidate explicitly says they don't know, have no idea, or lack experience with this topic, "
        "you MUST NOT ask a follow-up.\n"
        "Your follow-up must be consistent with the job, previous questions, and the candidate's answer.\n\n"
        f"{persona_block}"
    )

    user_msg = (
        "Job description:\n"
        f"{jd_text}\n\n"
        "Recent questions and answers (from oldest to newest):\n"
        f"{history_block}\n\n"
        "Current question and answer:\n"
        f"Question: {current_q}\n"
        f"Answer: {current_a_full}\n\n"
        "Your task:\n"
        "- Decide whether to ask ONE follow-up question.\n"
        "- Ask a follow-up only if it helps clarify the candidate's decisions, trade-offs, metrics, or impact.\n"
        "- The follow-up should be 1 sentence, concise, and directly related to what the candidate just said.\n\n"
        "Return STRICTLY a JSON object in this format:\n"
        "{\n"
        '  \"need_followup\": true or false,\n'
        '  \"question\": \"your follow-up question here, if any\",\n'
        '  \"tag\": \"a short label like \'Follow-up · Metrics\' (optional)\"\n'
        "}\n\n"
        "If you think no follow-up is needed, return:\n"
        "{ \"need_followup\": false }\n"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = _json.loads(raw)
    except Exception as e:
        print("[mock followup] LLM / JSON error:", e)
        return

    need_fu = bool(data.get("need_followup"))
    if not need_fu:
        return

    fu_q = (data.get("question") or "").strip()
    fu_tag = (data.get("tag") or "").strip() or "Follow-up"
    if not fu_q:
        return

    # 真正插入 follow-up slot：放在當前題的下一格，並重排 index
    insert_pos = idx + 1
    fu_spec = {
        "index": insert_pos,
        "type": "followup",
        "followup_of": idx,
        "followup_question_text": fu_q,
        "followup_tag": fu_tag,
    }

    plan.insert(insert_pos, fu_spec)
    # 重新編 index，確保 0..len-1 連續
    for i, s in enumerate(plan):
        s["index"] = i

    session["question_plan"] = plan
    save_mock_session(session)

    print(f"[mock followup] inserted follow-up after Q{idx}: {fu_q}")


def generate_interviewer_reaction(question: str, answer: str, session: Optional[Dict[str, Any]] = None) -> str:
    """
    根據該題的問題與回答，產生一個「面試官的一句話反應」。
    - 自然口語，1 句、20 字以內
    - 不再問新問題，只是回饋 / 承接
    - 如果答案顯示候選人不知道，也試著稍微安撫 & 接話
    - 會依照 interviewer persona 調整語氣
    """
    text = (answer or "").strip()
    if not text:
        return ""

    persona_text = ""
    if session is not None:
        try:
            persona_text = _build_interviewer_persona(session)
        except Exception:
            persona_text = ""

    persona_block = ""
    if persona_text.strip():
        persona_block = (
            "Here is your interviewer persona. Stay consistent with this tone:\n"
            f"{persona_text}\n\n"
        )

    system_msg = (
        "You are a realistic interviewer for data/ML roles.\n"
        "After hearing the candidate's answer, you respond with ONE short sentence.\n"
        "- Be natural, conversational, and concise (max 20 words).\n"
        "- Do NOT ask a new question here; it's just a quick reaction.\n"
        "- If the candidate clearly doesn't know or says 'I have no idea', "
        "briefly reassure them and imply you'll move on.\n\n"
        f"{persona_block}"
    )

    user_msg = (
        "Here is the interview turn:\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Write your one-sentence reaction:"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.6,
        )
        reaction = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("[mock reaction] LLM error:", e)
        return ""

    # 安全處理一下太長或空字串
    if not reaction:
        return ""
    if len(reaction.split()) > 25:
        # 粗暴：只留前 25 個詞
        reaction = " ".join(reaction.split()[:25])

    return reaction

def extract_wav_from_webm(session_id: str, index: int) -> Path:
    """
    自動從 mock_media/<sessionID>_<index>.webm 抽出音檔 .wav
    """
    webm_path = MOCK_MEDIA_DIR / f"{session_id}_{index}.webm"

    if not webm_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {webm_path}")

    wav_path = MOCK_MEDIA_DIR / f"{session_id}_{index}.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(webm_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(wav_path),
    ]

    subprocess.run(cmd, check=True)

    return wav_path