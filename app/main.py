from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from io import BytesIO
import websockets
import aiohttp

from pydantic import BaseModel
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import json, os
import sys, random
import shutil
import asyncio
from tempfile import NamedTemporaryFile


from parsers.resume_parser import (
    extract_pdf_text,
    parse_resume_entries,
    extract_metadata_sections,
    extract_structured_education,
)
from core.embeddings import build_resume_embeddings

from core.llm_client import client
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your_API_key"))

# from the module I built
from core.config import BASE_DIR, USER_DATA_DIR
from core.llm_client import client
from core.profiles import (
    load_job_profiles,
    save_job_profiles,
    load_all_profiles as _load_all_profiles_from_core,
)
from core.retrieval import (
    retrieve_bullets_for_profile,
    get_bullets_for_entry,
    load_resume_entries_and_embs,
)
from core.sessions import (
    load_session,
    get_asked_questions,
    log_practice_turn,
    log_asked_question,
    get_practice_stats,
)
from core.questions import (
    call_llm_for_question,
    get_behavioral_question,
    call_llm_for_project_question,
    generate_followup_question,
)
from core.answers import (
    call_llm_for_sample_answer,
    evaluate_answer,
)

from core.transcription import transcribe_media
from core import mock_interview
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR  # 你的專案根目錄
sys.path.append(str(ROOT_DIR))

USER_DATA_DIR.mkdir(exist_ok=True)
JOB_PROFILES_PATH = USER_DATA_DIR / "job_profiles.json"
SESSION_MEDIA_DIR = USER_DATA_DIR / "session_media"
SESSION_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Intelliview Coach")

# static / templates
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
timestamp = datetime.now(timezone.utc).isoformat()
now = datetime.utcnow().isoformat() + "Z"

def save_job_profiles(profiles: list[dict]) -> None:
    JOB_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOB_PROFILES_PATH.write_text(
        json.dumps({"profiles": profiles}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

# =========================
# 路徑設定（以下保留你原本的寫法，實際上 ROOT_DIR / USER_DATA_DIR 跟前面一致）
# =========================
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
USER_DATA_DIR = ROOT_DIR / "user_data"
USER_DATA_DIR.mkdir(exist_ok=True)
JOB_PROFILES_PATH = USER_DATA_DIR / "job_profiles.json"

# 錄音／錄影會存到這裡：user_data/session_media/<profile_id>/xxx.webm
SESSION_MEDIA_DIR = USER_DATA_DIR / "session_media"
SESSION_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(ROOT_DIR))

app = FastAPI(title="Intelliview Coach")

# static 檔案（CSS, JS）
static_dir = APP_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 🔥 新增：media 檔案（audio/video）
app.mount("/media", StaticFiles(directory=str(SESSION_MEDIA_DIR)), name="media")

templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def ensure_project_dirs(project_id: str):
    raw_dir = USER_DATA_DIR / "raw" / project_id
    parsed_dir = USER_DATA_DIR / "parsed" / project_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, parsed_dir

class JobProfileCreate(BaseModel):
    profile_id: str
    job_title: str
    company: str | None = None
    jd_text: str
    resume_id: str

# =========================
# FrontEnd Page
# =========================
# 首頁：Landing page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# 履歷設定頁：原本的 editor 搬到這裡
@app.get("/resume", response_class=HTMLResponse, name="resume_page")
async def resume_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/profiles", response_class=HTMLResponse, name="profiles_page")
async def profiles_page(request: Request):
    profiles = load_job_profiles()
    # 簡單按照 updated_at 排序（新到舊）
    profiles_sorted = sorted(
        profiles,
        key=lambda p: p.get("updated_at", ""),
        reverse=True,
    )

    # 為每個 profile 算一次 stats 摘要
    enriched = []
    for p in profiles_sorted:
        pid = p.get("profile_id")
        stats = get_practice_stats(pid)
        enriched.append(
            {
                **p,
                "stats": stats,
            }
        )

    return templates.TemplateResponse(
        "profiles.html",
        {
            "request": request,
            "profiles": enriched,   # ⭐ 用 enriched，而不是 profiles_sorted
        },
    )

@app.get("/profiles/new", response_class=HTMLResponse, name="new_profile_page")
async def new_profile_page(
    request: Request,
    resume_id: str | None = None,
):
    # 掃描已有的 resume 版本（parsed 底下的資料夾名）
    parsed_root = USER_DATA_DIR / "parsed"
    resume_ids: list[str] = []
    if parsed_root.exists():
        for folder in parsed_root.iterdir():
            if folder.is_dir():
                resume_ids.append(folder.name)
    resume_ids.sort()

    return templates.TemplateResponse(
        "new_profile.html",
        {
            "request": request,
            "resume_ids": resume_ids,
            "default_resume_id": resume_id,
        },
    )


@app.get("/api/profile/{profile_id}")
async def api_get_profile(profile_id: str):
    profiles = load_job_profiles()
    p = next((x for x in profiles if x.get("profile_id") == profile_id), None)
    if not p:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {
        "profile_id": p.get("profile_id"),
        "job_title": p.get("job_title"),
        "company": p.get("company"),
        "resume_id": p.get("resume_id"),
        "jd_text": p.get("jd_text", ""),
    }

def load_all_profiles():
    with open("user_data/job_profiles.json", "r") as f:
        data = json.load(f)
    # 只拿內層的 list
    return data.get("profiles", [])

@app.get("/profiles/{profile_id}/history", response_class=HTMLResponse)
async def practice_history_page(request: Request, profile_id: str):
    stats = get_practice_stats(profile_id)
    session = load_session(profile_id)
    turns = session.get("turns", [])

    all_profiles = load_all_profiles()  # ⭐ 新增這行

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "profile_id": profile_id,
            "stats": stats,
            "turns": turns,
            "all_profiles": all_profiles,   # ⭐ 傳給 template
        },
    )


@app.get("/practice/{profile_id}", response_class=HTMLResponse, name="practice_page")
async def practice_page(request: Request, profile_id: str):
    return templates.TemplateResponse(
        "practice.html",
        {"request": request, "profile_id": profile_id},
    )

@app.get("/api/profile_entries/{profile_id}")
async def api_profile_entries(profile_id: str):
    profiles = load_job_profiles()
    profile = next((p for p in profiles if p.get("profile_id") == profile_id), None)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    resume_id = profile.get("resume_id")
    if not resume_id:
        raise HTTPException(status_code=400, detail="Profile has no resume_id")

    entries, _ = load_resume_entries_and_embs(resume_id)

    seen = set()
    items = []
    for e in entries:
        section = e.get("section") or "EXPERIENCE"
        entry = e.get("entry") or ""
        if not entry:
            continue
        key = f"{section}||{entry}"
        if key in seen:
            continue
        seen.add(key)
        label = f"[{section}] {entry}"
        items.append({"entry_key": key, "label": label})

    return {"entries": items}

@app.get("/api/practice_stats/{profile_id}")
async def api_practice_stats(profile_id: str):
    stats = get_practice_stats(profile_id)
    return stats

@app.get("/api/practice_history/{profile_id}")
async def api_practice_history(profile_id: str):
    session = load_session(profile_id)
    turns = session.get("turns", [])
    # 你也可以在這裡做簡單排序或截斷
    return {"turns": turns}

@app.get("/mock_settings", response_class=HTMLResponse, name="mock_settings_page")
async def mock_settings_page(
    request: Request,
    resume_id: str | None = None,
):
    # 掃描 parsed/ 底下所有 resume version
    parsed_root = USER_DATA_DIR / "parsed"
    resume_ids: list[str] = []
    if parsed_root.exists():
        for folder in parsed_root.iterdir():
            if folder.is_dir():
                resume_ids.append(folder.name)
    resume_ids.sort()

    all_profiles = load_job_profiles()  # 讓 dropdown 有 profile 列表

    return templates.TemplateResponse(
        "mock_settings.html",
        {
            "request": request,
            "resume_ids": resume_ids,
            "default_resume_id": resume_id,
            "all_profiles": all_profiles,
        },
    )

@app.get("/mock_interview")
async def mock_interview_page(request: Request):
    q = request.query_params

    profile_id = q.get("profile_id")
    if not profile_id:
        raise HTTPException(status_code=400, detail="profile_id is required")

    profiles = load_job_profiles()
    profile = next((p for p in profiles if p.get("profile_id") == profile_id), None)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    resume_id = profile.get("resume_id")
    if not resume_id:
        raise HTTPException(
            status_code=400,
            detail="This profile has no linked resume. Please set it in Profiles first.",
        )

    mode = q.get("mode", "realistic")
    length_type = q.get("length_type", "questions")
    hint_level = q.get("hint_level", "standard")

    num_questions = q.get("num_questions")
    time_limit = q.get("time_limit")

    num_questions_int = int(num_questions) if num_questions else None
    time_limit_int = int(time_limit) if time_limit else None

    # ====== 這是從 settings.html 來的 interviewer 設定 ======
    interviewer_gender = q.get("interviewer_gender", "auto")

    role_preset = q.get("interviewer_role") or "senior_engineer"
    role_custom = q.get("interviewer_role_custom") or ""

    style_preset = q.get("interviewer_style_preset") or "balanced"
    style_custom = q.get("interviewer_style_custom") or ""

    extra_notes = (q.get("interviewer_extra_notes") or "").strip()

    # 可以簡單 resolve（你如果有自己的 resolver 也可以用自己的）
    def resolve_role(preset: str, custom: str) -> str:
        if preset == "custom":
            return custom or "an interviewer for this role"
        # 可以自己 map；這裡先簡單寫
        mapping = {
            "senior_engineer": "a senior data / ML / SWE engineer on the team you’d work with",
            "hiring_manager": "the hiring manager who cares about team fit, ownership, and impact",
            "recruiter": "a recruiter or HR partner focusing on overall fit and communication",
            "peer_teammate": "a future teammate who wants to know what it’s like to work with you day to day",
            "executive": "a director or VP who cares about business impact and prioritization",
        }
        return mapping.get(preset, "an interviewer for this role")

    def resolve_style(preset: str, custom: str) -> str:
        if preset == "custom":
            return custom or "balanced, realistic, and professional"
        mapping = {
            "balanced": "balanced, neutral but probing",
            "supportive": "supportive, encouraging and patient",
            "direct": "direct and concise, to the point",
            "challenging": "challenging and skeptical, pushes on vague claims",
            "high_pressure": "fast-paced, high-pressure, tests how the candidate handles stress",
        }
        return mapping.get(preset, "balanced, realistic, and professional")

    resolved_role = resolve_role(role_preset, role_custom)
    resolved_style = resolve_style(style_preset, style_custom)

    # ⭐ 這個 persona string 會丟到 TTS 的 req.instructions
    tts_persona = (
        f"{resolved_role}. {resolved_style}. "
        f"{extra_notes}" if extra_notes else f"{resolved_role}. {resolved_style}."
    )

    # 組成 interviewer_profile 丟進 session
    interviewer_profile = {
        "gender": interviewer_gender,
        "role_preset": role_preset,
        "role_resolved": resolved_role,
        "style_preset": style_preset,
        "style_resolved": resolved_style,
        "extra_notes": extra_notes,
        # 這個欄位會最後變成 TTS 的 instructions → persona_to_instructions()
        "tts_persona": tts_persona,
    }

    # ====== 建立 session：這裡要記得傳 interviewer_profile=... ======
    session = mock_interview.create_mock_session(
        profile_id=profile_id,
        resume_id=resume_id,
        mode=mode,
        length_type=length_type,
        hint_level=hint_level,
        num_questions=num_questions_int,
        time_limit=time_limit_int,
        interviewer_profile=interviewer_profile,   # 👈 關鍵
    )

    # 前端要用 `SESSION_CONFIG` 來 call /api/tts_question
    session_config = {
        "session_id": session["session_id"],
        "profile_id": profile_id,
        "resume_id": resume_id,
        "mode": mode,
        "length_type": length_type,
        "hint_level": hint_level,
        "num_questions": session.get("num_questions"),
        "time_limit": session.get("time_limit"),

        # 讓 JS 可以拿來當 voice / instructions
        "interviewer_gender": interviewer_gender,
        "interviewer_role": resolved_role,
        "interviewer_style": resolved_style,
        "interviewer_extra_notes": extra_notes,
        "tts_instructions": tts_persona,
    }

    import json as _json
    session_config_json = _json.dumps(session_config)

    return templates.TemplateResponse(
        "mock_interview.html",
        {
            "request": request,
            "session_config_json": session_config_json,
        },
    )




@app.get("/profiles/{profile_id}/mock_history")
async def mock_history_index(request: Request, profile_id: str):
    sessions = mock_interview.list_mock_sessions_for_profile(profile_id)
    all_profiles = load_all_profiles()  # 跟 practice history 一樣，右上用來切 profile

    return templates.TemplateResponse(
        "mock_history.html",
        {
            "request": request,
            "profile_id": profile_id,
            "sessions": sessions,
            "all_profiles": all_profiles,
        },
    )


# ================================
#  Single mock result page
# ================================
@app.get("/mock/{session_id}")
def mock_report_page(request: Request, session_id: str):
    """
    顯示單一 mock interview 的報告頁
    """
    report = mock_interview.load_mock_result(session_id)
    return templates.TemplateResponse(
        "mock_report.html",
        {
            "request": request,
            "report": report,
        }
    )


@app.get("/profiles/{profile_id}/mock_history/{session_id}")
async def mock_report_page_profile(request: Request, profile_id: str, session_id: str):
    report = mock_interview.load_mock_result(session_id)
    return templates.TemplateResponse(
        "mock_report.html",
        {"request": request, "profile_id": profile_id, "report": report},
    )

# =========================
# API：上傳履歷並 parse
# =========================
@app.post("/api/upload_resume")
async def upload_resume(
    project_id: str = Form(...),
    file: UploadFile = File(...)
):
    # 準備目錄
    raw_dir = USER_DATA_DIR / "raw" / project_id
    parsed_dir = USER_DATA_DIR / "parsed" / project_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    # 永遠存成 resume.pdf
    resume_path = raw_dir / "resume.pdf"
    content = await file.read()
    with open(resume_path, "wb") as f:
        f.write(content)

    # 用你自己的 parser
    raw_text = extract_pdf_text(str(resume_path))
    entries = parse_resume_entries(raw_text)
    metadata = extract_metadata_sections(raw_text)
    education_structured = extract_structured_education(raw_text)

    # 存原始 parse 結果（之後 Save 才會存 edited 版）
    with open(parsed_dir / "experience_entries.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    with open(parsed_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(parsed_dir / "education_structured.json", "w", encoding="utf-8") as f:
        json.dump(education_structured, f, ensure_ascii=False, indent=2)

    return JSONResponse(
        content={
            "project_id": project_id,
            "entries": entries,
            "metadata": metadata,
            "education_structured": education_structured
        }
    )


# =========================
# API：存編輯後的結果
# =========================
class SaveResumePayload(BaseModel):
    project_id: str
    entries: list[dict]
    metadata: dict
    education_structured: list[dict]


@app.post("/api/save_resume")
async def save_resume(payload: SaveResumePayload):
    project_id = payload.project_id
    parsed_dir = USER_DATA_DIR / "parsed" / project_id
    parsed_dir.mkdir(parents=True, exist_ok=True)

    # 1) 存 edited 版本
    with open(parsed_dir / "experience_entries_edited.json", "w", encoding="utf-8") as f:
        json.dump(payload.entries, f, ensure_ascii=False, indent=2)

    with open(parsed_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(payload.metadata, f, ensure_ascii=False, indent=2)

    with open(parsed_dir / "education_structured.json", "w", encoding="utf-8") as f:
        json.dump(payload.education_structured, f, ensure_ascii=False, indent=2)

    # 2) 用 fine-tuned model 建 embeddings
    #    → user_data/embeddings/{project_id}/resume_bullets.npy
    try:
        build_resume_embeddings(project_id)
        built = True
    except Exception as e:
        # 不要讓整個 API 爆掉，return 讓前端知道 embedding 失敗
        print("Error building embeddings:", e)
        built = False

    # 3) 回傳給前端
    return JSONResponse(
        content={
            "status": "ok",
            "project_id": project_id,
            "embeddings_built": built
        }
    )

@app.post("/api/create_job_profile")
async def create_job_profile(payload: JobProfileCreate):
    profiles = load_job_profiles()

    now = datetime.utcnow().isoformat() + "Z"

    # 如果同一個 profile_id 已存在，就更新
    existing = None
    for p in profiles:
        if p.get("profile_id") == payload.profile_id:
            existing = p
            break

    if existing:
        existing.update(
            {
                "job_title": payload.job_title,
                "company": payload.company,
                "jd_text": payload.jd_text,
                "resume_id": payload.resume_id,
                "updated_at": now,
            }
        )
    else:
        profiles.append(
            {
                "profile_id": payload.profile_id,
                "job_title": payload.job_title,
                "company": payload.company,
                "jd_text": payload.jd_text,
                "resume_id": payload.resume_id,
                "created_at": now,
                "updated_at": now,
            }
        )

    save_job_profiles(profiles)
    return JSONResponse(
        content={"status": "ok", "profile_id": payload.profile_id},
    )

class NextQuestionRequest(BaseModel):
    profile_id: str
    mode: str                    # "auto" | "behavioral" | "project" | "technical" | "case" | "custom"
    behavioral_type: Optional[str] = None
    entry_key: Optional[str] = None
    prev_answer: Optional[str] = None
    custom_question: Optional[str] = None

@app.post("/api/next_question")
async def api_next_question(req: NextQuestionRequest):
    profiles = load_job_profiles()
    profile = next((p for p in profiles if p.get("profile_id") == req.profile_id), None)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    jd_text = profile.get("jd_text", "")
    mode = (req.mode or "auto").lower()

    # === auto: LLM + JD + 避免重複 ===
    if mode == "auto":
        asked = get_asked_questions(req.profile_id, mode="auto")
        question = call_llm_for_question(jd_text, mode="auto", avoid=asked)

        bullets = retrieve_bullets_for_profile(req.profile_id, question, top_k=5)
        tag = "Auto (from JD)"
        behavioral_type = None
        entry_key = None

    # === behavioral: 題庫 + subtype + 避免重複 ===
    elif mode == "behavioral":
        subtype = req.behavioral_type or "random"
        question = get_behavioral_question(req.profile_id, subtype)
        bullets = retrieve_bullets_for_profile(req.profile_id, question, top_k=5)
        tag = f"Behavioral · {subtype}"
        behavioral_type = subtype
        entry_key = None

    # === project deep dive ===
    elif mode == "project":
        if not req.entry_key:
            raise HTTPException(status_code=400, detail="entry_key required for project mode")

        entry_key = req.entry_key
        resume_id = profile.get("resume_id")
        if not resume_id:
            raise HTTPException(status_code=400, detail="Profile has no resume_id")

        entry_bullets = get_bullets_for_entry(resume_id, entry_key)

        # 建 previous_qas（session 內所有這個 entry 的 Q/A）
        session = load_session(req.profile_id)
        qa_history = []
        for t in session.get("turns", []):
            if t.get("mode") == "project" and t.get("entry_key") == entry_key:
                qa_history.append(
                    {
                        "question": t.get("question"),
                        "answer": t.get("user_answer") or "",
                    }
                )

        # 把這一輪使用者剛打的答案（prev_answer）也串進 context
        last_question = qa_history[-1]["question"] if qa_history else None
        if req.prev_answer and last_question:
            qa_history.append({"question": last_question, "answer": req.prev_answer})

        question = call_llm_for_project_question(
            jd_text=jd_text,
            entry_title=entry_key.split("||", 1)[1],
            bullets=entry_bullets,
            previous_qas=qa_history,
        )
        bullets = entry_bullets
        tag = "Project deep dive"
        behavioral_type = None

    # === technical: 用 JD 生技術題 ===
    elif mode == "technical":
        asked = get_asked_questions(req.profile_id, mode="technical")
        question = call_llm_for_question(
            jd_text=jd_text,
            mode="technical",
            avoid=asked,
        )
        bullets = retrieve_bullets_for_profile(req.profile_id, question, top_k=5)
        tag = "Technical question"
        behavioral_type = None
        entry_key = None

    # === case: 用 JD 生 case reasoning 題 ===
    elif mode == "case":
        asked = get_asked_questions(req.profile_id, mode="case")
        question = call_llm_for_question(
            jd_text=jd_text,
            mode="case",
            avoid=asked,
        )
        bullets = retrieve_bullets_for_profile(req.profile_id, question, top_k=5)
        tag = "Case interview question"
        behavioral_type = None
        entry_key = None

    # === custom: 前端自訂題目 ===
    elif mode == "custom":
        if not req.custom_question:
            raise HTTPException(status_code=400, detail="custom_question is required for custom mode")

        question = req.custom_question
        bullets = retrieve_bullets_for_profile(req.profile_id, question, top_k=5)
        tag = "Custom question"
        behavioral_type = None
        entry_key = None

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported mode: {mode}")

    return {
        "question": question,
        "tag": tag,
        "bullets": bullets,
        "mode": mode,
        "behavioral_type": behavioral_type,
        "entry_key": entry_key,
    }

class BulletsRequest(BaseModel):
    profile_id: str
    question: str

@app.post("/api/retrieve_bullets")
async def api_retrieve_bullets(req: BulletsRequest):
    bullets = retrieve_bullets_for_profile(req.profile_id, req.question, top_k=3)
    return {"bullets": bullets}

class CoachChatRequest(BaseModel):
    profile_id: str
    mode: str
    question: str
    user_message: str
    sample_answer: Optional[str] = None
    bullets: Optional[List[Dict[str, Any]]] = None
    history: Optional[List[Dict[str, str]]] = None   # [{role, content}, ...]

@app.post("/api/coach_chat")
async def api_coach_chat(req: CoachChatRequest):
    """
    Coach chat:
    - 一定會有當前 question
    - sample_answer 可以為空（代表還沒 generate）
    - bullets 如果沒傳，就自己 RAG 撈 top-k
    - history 用來保留此輪 coach 對話記憶
    """
    profiles = load_job_profiles()
    profile = next((p for p in profiles if p.get("profile_id") == req.profile_id), None)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    jd_text = profile.get("jd_text", "")

    # 若前端沒傳 bullets，自己 RAG 一份
    if req.bullets:
        bullets = req.bullets
    else:
        bullets = retrieve_bullets_for_profile(req.profile_id, req.question, top_k=5)

    # 準備 bullet context
    bullet_lines = []
    for b in bullets:
        entry = b.get("entry") or "Unknown entry"
        text = b.get("text") or ""
        bullet_lines.append(f"- [{entry}] {text}")
    bullet_block = "\n".join(bullet_lines) if bullet_lines else "(none)"

    # 對話歷史（只拿最後幾輪）
    history = req.history or []
    trimmed_history = history[-8:]  # 最多 8 則

    # system + user prompt
    system_msg = (
        "You are an interview coach helping a candidate refine their answer, do not give user the sample answer unless they ask for it. "
        "Use the job description, question, resume bullets, and (if available) "
        "the current sample answer. Be concrete and actionable."
    )

    context_block = f"""
Job description:
{jd_text}

Current interview question:
{req.question}

Relevant resume bullets:
{bullet_block}

Current sample answer (may be empty if not generated yet):
{req.sample_answer or "(none yet — help them think about how to answer first.)"}
"""

    messages = [{"role": "system", "content": system_msg}]
    messages.append({"role": "user", "content": context_block})

    # 加入歷史
    for m in trimmed_history:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not content:
            continue
        messages.append({"role": role, "content": content})

    # 最後這一輪使用者的訊息
    messages.append({"role": "user", "content": req.user_message})

    from core.rag_pipeline import client as rag_client  # 避免名稱衝突

    resp = rag_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.6,
    )
    reply = resp.choices[0].message.content.strip()

    return {
        "reply": reply,
        "bullets": bullets,  # 讓前端若要的話可以更新 sidebar
    }


class SampleAnswerRequest(BaseModel):
    profile_id: str
    question: str
    mode: str
    behavioral_type: Optional[str] = None
    entry_key: Optional[str] = None
    user_answer: Optional[str] = None
    bullets: Optional[List[Dict[str, Any]]] = None

@app.post("/api/generate_sample_answer")
async def api_generate_sample_answer(req: SampleAnswerRequest):
    profiles = load_job_profiles()
    profile = next((p for p in profiles if p.get("profile_id") == req.profile_id), None)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    jd_text = profile.get("jd_text", "")

    # 若前端沒傳 bullets，就讓後端自己 RAG 找
    if req.bullets:
        bullets = req.bullets
    else:
        bullets = retrieve_bullets_for_profile(req.profile_id, req.question, top_k=5)

    llm_result = call_llm_for_sample_answer(
        question=req.question,
        jd_text=jd_text,
        bullets=bullets,
        user_answer=req.user_answer,
    )

    return {
        "answer": llm_result.get("answer", ""),
        "hint": llm_result.get("hint", ""),
        "rationale": llm_result.get("rationale", ""),
        "bullets": bullets,
    }


class SaveUserAnswerRequest(BaseModel):
    profile_id: str
    question: str
    mode: str
    behavioral_type: Optional[str] = None
    entry_key: Optional[str] = None
    user_answer: Optional[str] = None
    bullets: Optional[List[Dict[str, Any]]] = None
    sample_answer: Optional[str] = None
    thread_id: Optional[str] = None     
    is_followup: bool = False  

@app.post("/api/save_user_answer")
async def api_save_user_answer(req: SaveUserAnswerRequest):
    profiles = load_job_profiles()
    profile = next((p for p in profiles if p.get("profile_id") == req.profile_id), None)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    jd_text = profile.get("jd_text", "")

    if not req.user_answer or not req.user_answer.strip():
        raise HTTPException(status_code=400, detail="user_answer is empty")

    # bullets: 前端有勾選就用它；沒有就自己 RAG
    if req.bullets:
        bullets = req.bullets
    else:
        bullets = retrieve_bullets_for_profile(req.profile_id, req.question, top_k=5)

    # 評分
    eval_result = evaluate_answer(
        question=req.question,
        jd_text=jd_text,
        bullets=bullets,
        user_answer=req.user_answer,
    )

    # ---- 兼容舊欄位的 mapping ----
    # 新版 evaluate_answer 用 overall_score / improvements_overview
    score = eval_result.get("overall_score")
    if score is None:
        score = eval_result.get("score", 5)

    strengths = (eval_result.get("strengths") or "").strip()

    improvements = (
        eval_result.get("improvements_overview")
        or eval_result.get("improvements")
        or ""
    ).strip()

    # 寫入 session，只在這一步才 log
    log_practice_turn(
        profile_id=req.profile_id,
        question=req.question,
        sample_answer=req.sample_answer,
        bullets=bullets,
        mode=req.mode,
        behavioral_type=req.behavioral_type,
        entry_key=req.entry_key,
        user_answer=req.user_answer,
        score=score,
        strengths=strengths,
        improvements=improvements,
        thread_id=req.thread_id,
        is_followup=req.is_followup,
    )

    # 回給前端的 eval_result 保留完整新版結構
    #（如果前端有寫死用 score / improvements，也可以順手補上）
    eval_result_out = dict(eval_result)
    eval_result_out.setdefault("score", score)
    eval_result_out.setdefault("improvements", improvements)

    return eval_result_out

class FollowupQuestionRequest(BaseModel):
    profile_id: str
    mode: str                     # "auto" | "behavioral" | "project" | "custom"
    base_question: str            # 主題目的問題（第一題）
    user_answer: Optional[str] = None  # 剛剛這題的最新回答（尚未存檔也可以）
    thread_id: Optional[str] = None    # 如果前端有 thread_id（UUID），就傳；沒有就用 base_question 當預設
    entry_key: Optional[str] = None
    bullets: Optional[List[Dict[str, Any]]] = None

MAX_FOLLOWUPS_PER_THREAD = 3  # 你可以之後調整這個數字

@app.post("/api/followup_question")
async def api_followup_question(req: FollowupQuestionRequest):
    """
    產生追問問題：
    - 適用所有 mode（auto/behavioral/project/custom）
    - 根據 base_question + 該 thread 的 QA 歷史 + 最新 user_answer 來問
    - 同一個 thread 內避免問重複的問題
    """
    profiles = load_job_profiles()
    profile = next((p for p in profiles if p.get("profile_id") == req.profile_id), None)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    jd_text = profile.get("jd_text", "")
    mode = (req.mode or "auto").lower()

    # 1) bullets：沒給就自己 RAG
    if req.bullets:
        bullets = req.bullets
    else:
        bullets = retrieve_bullets_for_profile(req.profile_id, req.base_question, top_k=5)

    # 2) 找出這個 thread 底下的既有 QA（已存進 turns 的）
    session = load_session(req.profile_id)
    thread_id = req.thread_id or req.base_question  # 簡單版：沒 thread_id 就用主題目當 ID

    thread_turns = [
        t for t in session.get("turns", [])
        if t.get("thread_id") == thread_id
    ]

    # 計算已經追問幾題
    followup_count = sum(1 for t in thread_turns if t.get("is_followup"))
    if followup_count >= MAX_FOLLOWUPS_PER_THREAD:
        return {
            "question": None,
            "done": True,
            "message": "This topic has already been explored with several follow-up questions. Consider moving on to a new question.",
            "thread_id": thread_id,
        }

    # 3) 組 QA history（只給 LLM 看，不一定全部要顯示在 UI）
    qa_history = []
    for t in thread_turns:
        q = t.get("question") or ""
        a = t.get("user_answer") or ""
        if q or a:
            qa_history.append({"question": q, "answer": a})

    # 把這一輪剛輸入的 user_answer 也加進去（即使還沒存檔）
    if req.user_answer:
        qa_history.append({"question": req.base_question, "answer": req.user_answer})

    # 4) thread 內避免重複的問題（主題 + 既有追問）
    avoid = set()
    for t in thread_turns:
        q = t.get("question")
        if q:
            avoid.add(q.strip())
    avoid.add(req.base_question.strip())

    # 5) 實際叫 LLM 生追問問題（含避免重複）
    followup_q = generate_followup_question(
        jd_text=jd_text,
        mode=mode,
        base_question=req.base_question,
        bullets=bullets,
        qa_history=qa_history,
        avoid=avoid,
    )

    if not followup_q:
        # 代表 LLM 怎麼樣都生不出足夠不同的新問題
        return {
            "question": None,
            "done": True,
            "message": "The model could not generate a sufficiently different follow-up question. Let's move on to a new topic.",
            "thread_id": thread_id,
        }

    return {
        "question": followup_q,
        "mode": mode,
        "bullets": bullets,
        "entry_key": req.entry_key,
        "thread_id": thread_id,
        "is_followup": True,
        "done": False,
        "tag": f"Follow-up \u00b7 {mode}",
    }

@app.post("/api/save_user_answer_with_media")
async def api_save_user_answer_with_media(
    meta: str = Form(...),
    media: UploadFile | None = File(None),
):
    """
    前端會用 FormData 傳：
      - meta: JSON 字串，內容跟 SaveUserAnswerRequest 一樣（沒有 user_answer 也可以）
      - media: 錄音/錄影檔 (optional)

    流程：
      1. 先把 media 存到 user_data/session_media/<profile_id>/xxx.webm
      2. 如果 meta 裡沒有 user_answer 且有 media → 用 Whisper-1 轉錄成文字
      3. 把轉錄文字填進 req.user_answer → 走原本評分 & log pipeline
      4. 回傳評分結果（另外多帶 transcript，前端之後可以用來顯示）
    """
    # ----- 解析 meta -----
    try:
        meta_dict = json.loads(meta)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid meta JSON")

    # 用既有的 Pydantic model 做驗證
    req = SaveUserAnswerRequest(**meta_dict)

    profiles = load_job_profiles()
    profile = next((p for p in profiles if p.get("profile_id") == req.profile_id), None)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    jd_text = profile.get("jd_text", "")

    # ----- 判斷是否有文字 / media -----
    has_text = bool(req.user_answer and req.user_answer.strip())
    has_media_upload = media is not None

    # 如果兩個都沒有，直接擋掉
    if not has_text and not has_media_upload:
        raise HTTPException(status_code=400, detail="No text answer or media provided")

    # ---------- 處理 media 檔案：存到 SESSION_MEDIA_DIR ----------
    media_type: Optional[str] = None
    media_filename: Optional[str] = None
    media_duration_ms: Optional[int] = None
    saved_media_path: Optional[Path] = None

    # meta 裡可以帶一個 media_meta: {type, durationMs}
    media_meta = meta_dict.get("media_meta") or {}
    if media_meta:
        media_type = media_meta.get("type")
        media_duration_ms = media_meta.get("durationMs")

    if has_media_upload:
        content_type = media.content_type or ""
        if "video" in content_type:
            ext = ".webm"  # video/webm
        elif "audio" in content_type:
            ext = ".webm"  # audio/webm
        else:
            ext = ".bin"

        ts = datetime.utcnow().isoformat().replace(":", "-")
        safe_profile = req.profile_id.replace("/", "_")
        filename = f"{safe_profile}_{ts}{ext}"

        profile_dir = SESSION_MEDIA_DIR / safe_profile
        profile_dir.mkdir(parents=True, exist_ok=True)

        media_path = profile_dir / filename
        content = await media.read()
        media_path.write_bytes(content)

        # 存在 session.json 裡面的路徑：相對於 SESSION_MEDIA_DIR
        media_filename = str(media_path.relative_to(SESSION_MEDIA_DIR))
        saved_media_path = media_path

        # 如果前端沒傳 media_type，就從 content_type 猜
        if media_type is None:
            if "video" in content_type:
                media_type = "video"
            elif "audio" in content_type:
                media_type = "audio"

    # ---------- 如果沒有文字但有 media → 做轉錄 ----------
    transcript_text: Optional[str] = None

    if (not has_text) and saved_media_path is not None:
        try:
            # 你之後可以依照 profile / user 習慣調整 language
            transcript_text = transcribe_media(
                saved_media_path,
                language="en",
                prompt="This is an interview answer from a candidate. Please transcribe clearly.",
            )
        except Exception as e:
            print(f"[api_save_user_answer_with_media] Transcription error: {e}")
            transcript_text = None

        if transcript_text and transcript_text.strip():
            req.user_answer = transcript_text.strip()
            has_text = True  # 之後可以進入原本的評分流程

    # ---------- bullets：跟原本一樣 ----------
    if req.bullets:
        bullets = req.bullets
    else:
        bullets = retrieve_bullets_for_profile(req.profile_id, req.question, top_k=5)

    # ---------- 評分邏輯：只要最後有文字就評分 ----------
    if has_text:
        eval_result = evaluate_answer(
            question=req.question,
            jd_text=jd_text,
            bullets=bullets,
            user_answer=req.user_answer,
        )
        score = eval_result["score"]
        strengths = eval_result["strengths"]
        improvements = eval_result["improvements"]
    else:
        # 到這一步還是沒有文字（例如轉錄失敗）→ 不評分，只紀錄
        score = None
        strengths = ""
        improvements = ""
        eval_result = {
            "score": score,
            "strengths": strengths,
            "improvements": improvements,
        }

    # 如果有轉錄文字，就順便回傳給前端（之後你可以用在 practice 頁面顯示）
    if transcript_text:
        eval_result["transcript"] = transcript_text

    # ---------- 寫入 session ----------
    log_practice_turn(
        profile_id=req.profile_id,
        question=req.question,
        sample_answer=req.sample_answer,
        bullets=bullets,
        mode=req.mode,
        behavioral_type=req.behavioral_type,
        entry_key=req.entry_key,
        user_answer=req.user_answer,   # 這裡可能是：使用者打的 或 轉錄文字
        score=score,
        strengths=strengths,
        improvements=improvements,
        thread_id=req.thread_id,
        is_followup=req.is_followup,
        media_type=media_type,
        media_filename=media_filename,
        media_duration_ms=media_duration_ms,
    )

    return eval_result

class MockNextQuestionRequest(BaseModel):
    profile_id: str
    index: int                 # 第幾題（1-based）
    session_config: Dict[str, Any]
    prev_answer: Optional[str] = None
    entry_key: Optional[str] = None   # 如果你想強制某題是 project


@app.post("/api/mock_next_question")
async def api_mock_next_question(payload: Dict[str, Any]):
    """
    body:
    {
      "session_id": "...",
      "index": 0,         # 0-based
      "seconds_left": 900 # ✅ time mode 的時候才會帶，單位：秒
    }
    """
    session_id = payload.get("session_id")
    index_raw = payload.get("index", 0)
    seconds_left = payload.get("seconds_left")   # ✅ time mode 用

    if session_id is None:
        raise HTTPException(status_code=400, detail="session_id is required")

    try:
        index = int(index_raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="index must be an integer")

    try:
        # ✅ 把秒數傳給 mock_interview
        q = mock_interview.get_question_for_index(
            session_id,
            index,
            seconds_left=seconds_left,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    # ---------- 在這裡把 reaction + question 合併 ----------
    # 假設 mock_interview.get_question_for_index 會回傳類似：
    # {
    #   "question": "Can you walk me through ...",
    #   "tag": "...",
    #   "reaction": "It's great to hear you're studying at Columbia; ..."
    #   ...
    # }
    reaction_text = (q.get("reaction") or "").strip()
    question_text = (q.get("question") or "").strip()

    if reaction_text and question_text:
        # 兩行，同一個 bubble、同樣字體
        # 如果你想同一行就改成 f"{reaction_text} {question_text}"
        combined = f"{reaction_text}\n\n{question_text}"
        q["question"] = combined
    elif reaction_text:
        # 萬一沒有 question（理論上不會），至少不要丟掉 reaction
        q["question"] = reaction_text

    return JSONResponse(q)

    
@app.post("/api/mock_finish")
async def api_mock_finish(meta: str = Form(...)):
    """
    End interview 時呼叫：
      - meta: JSON 字串，至少要有 {session_id}
    不再需要整段 media，因為每題已經用 /api/mock_answer 存好 transcript。
    """
    import json as _json

    try:
        meta_obj = _json.loads(meta)
    except _json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid meta JSON")

    session_id = meta_obj.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    try:
        report = mock_interview.finalize_mock_session(session_id=session_id)
    except Exception as e:
        print("[api_mock_finish] finalize_mock_session error:", e)
        raise HTTPException(status_code=500, detail="Failed to finalize mock session")

    return {
        "session_id": session_id,
        "overall_score": report.get("overall_score"),
    }

@app.post("/api/mock_answer")
async def api_mock_answer(
    meta: str = Form(...),
    media: UploadFile = File(...),
):
    """
    一題結束時呼叫：
      - meta: JSON 字串，至少包含 {session_id, index, question_id, question_text}
              （如果有使用 realtime transcript，會多帶 realtime_transcript）
      - media: 這一題的錄音/錄影 (webm)

    後端：
      1) 存檔到 user_data/mock/media/<session_id>_<index>.webm
      2) 優先使用 realtime_transcript；若沒有，再呼叫 transcribe_media
      3) 產生一句短反應（像面試官會說的話）
      4) 存進 mock session 的 answers（含 reaction、transcript_source）
      5) 判斷是否插 follow-up（不算題數）
    """
    import json

    # -----------------------------
    # parse meta
    # -----------------------------
    try:
        meta_obj = json.loads(meta)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid meta JSON")

    session_id = meta_obj.get("session_id")
    index = meta_obj.get("index")
    question_id = meta_obj.get("question_id")
    question_text = meta_obj.get("question_text") or ""

    if session_id is None or index is None:
        raise HTTPException(status_code=400, detail="session_id and index are required")

    # -----------------------------
    # 1) save media
    # -----------------------------
    media_dir = mock_interview.MOCK_MEDIA_DIR
    media_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{session_id}_{index}.webm"
    media_path = media_dir / filename

    with media_path.open("wb") as f:
        shutil.copyfileobj(media.file, f)

    # -----------------------------
    # 2) transcription：優先用 realtime_transcript
    # -----------------------------
    # 前端如果有串 Realtime，就在 meta 裡帶上 realtime_transcript
    realtime_text = meta_obj.get("realtime_transcript") or ""
    if not isinstance(realtime_text, str):
        realtime_text = ""
    realtime_text = realtime_text.strip()

    transcript_source = "none"
    transcript_text = ""

    if realtime_text:
        # ✅ 正常情況：用 Realtime API 已經轉好的文字
        transcript_text = realtime_text
        transcript_source = "realtime"
        print('is working')
    else:
        print('RT not working')
        # 🛟 Fallback：如果沒有 realtime（或失敗），才跑 batch transcribe
        try:
            transcript_text = transcribe_media(
                media_path,
                language="en",
                prompt="This is a mock interview answer. Please transcribe clearly.",
            )
            transcript_source = "batch"
        except Exception as e:
            print("[api_mock_answer] transcription error:", e)
            transcript_text = ""
            transcript_source = "error"

    # -----------------------------
    # 3) interviewer reaction（先算好）
    # -----------------------------
    try:
        reaction = mock_interview.generate_interviewer_reaction(
            question_text,
            transcript_text or "",
        )
    except Exception as e:
        print("[api_mock_answer] reaction error:", e)
        reaction = ""

    # -----------------------------
    # 4) write into session.answers（包含 reaction + transcript_source）
    # -----------------------------
    session = mock_interview.load_mock_session(session_id)
    answers = session.get("answers") or []

    # remove previous record of same index
    answers = [a for a in answers if a.get("index") != index]

    answer_obj = {
        "index": index,
        "question_id": question_id,
        "question_text": question_text,
        "transcript": transcript_text or "",
        "reaction": reaction or "",
        "transcript_source": transcript_source,   # 👈 新增：記錄來源（realtime/batch/error）
    }

    answers.append(answer_obj)
    session["answers"] = answers
    mock_interview.save_mock_session(session)

    # -----------------------------
    # 5) try inserting a follow-up question
    # -----------------------------
    try:
        mock_interview._maybe_add_followup_after_answer(
            session_id=session_id,
            answer=answer_obj,
        )
    except Exception as e:
        print("[api_mock_answer] maybe_add_followup error:", e)

    return {
        "status": "ok",
        "index": index,
        "has_transcript": bool(transcript_text),
        "reaction": reaction,
        "transcript": transcript_text or "",
        "transcript_source": transcript_source,
    }

@app.websocket("/ws/mock_realtime")
async def ws_mock_realtime(client_ws: WebSocket):
    await client_ws.accept()
    print("[ws_mock_realtime] client connected")

    if not OPENAI_API_KEY:
        await client_ws.send_text(json.dumps({
            "type": "error",
            "message": "OPENAI_API_KEY is not set on server",
        }))
        await client_ws.close()
        return

    openai_url = "wss://api.openai.com/v1/realtime?intent=transcription"

    session = aiohttp.ClientSession()

    try:
        async with session.ws_connect(
            openai_url,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "openai-beta": "realtime=v1",
            },
        ) as openai_ws:
            print("[ws_mock_realtime] connected to OpenAI Realtime")

            # ✅ 正確的 transcription_session.update：所有設定包在 "session" 裡
            await openai_ws.send_json({
                "type": "transcription_session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "whisper-1",
                        "prompt": "",
                        "language": "en",
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                    },
                    "input_audio_noise_reduction": {
                        "type": "near_field",
                    },
                }
            })
            print("[ws_mock_realtime] sent transcription_session.update")

            async def pump_client_to_openai():
                try:
                    async for msg in client_ws.iter_text():
                        try:
                            data = json.loads(msg)
                        except Exception:
                            continue

                        if data.get("type") in (
                            "input_audio_buffer.append",
                            "input_audio_buffer.commit",
                        ):
                            await openai_ws.send_json(data)
                except Exception as e:
                    print("[ws_mock_realtime] client->openai error:", e)

            async def pump_openai_to_client():
                """
                把 OpenAI 發回來的 event 裡的文字抓出來，送成：
                  { "type": "transcript", "text": "<全文或目前累積>" }
                """
                current_text = ""

                async for msg in openai_ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                            break
                        continue

                    try:
                        event = msg.json()
                    except Exception as e:
                        print("[ws_mock_realtime] parse error:", e)
                        continue

                    etype = event.get("type", "")
                    print("[ws_mock_realtime] OpenAI event:", etype)

                    # === 1) 部分文字（有些版本叫 partial，有些叫 delta） ===
                    if etype in (
                        "conversation.item.input_audio_transcription.partial",
                        "conversation.item.input_audio_transcription.delta",
                    ):
                        # 目前官方例子是 transcript 或 delta 直接在頂層
                        fragment = (
                            event.get("delta")      # delta 版本
                            or event.get("transcript")  # partial 版本
                            or ""
                        )
                        if fragment:
                            current_text += fragment
                            await client_ws.send_text(json.dumps({
                                "type": "transcript",
                                "text": current_text,
                            }))
                        continue

                    # === 2) 完整一句結束 ===
                    if etype == "conversation.item.input_audio_transcription.completed":
                        final_text = event.get("transcript") or ""
                        if final_text:
                            current_text = final_text
                            await client_ws.send_text(json.dumps({
                                "type": "transcript",
                                "text": current_text,
                            }))
                        continue

                    # 其他事件（speech_started / committed / conversation.item.created 等）先略過
                    # 如果要 debug，可以暫時印整個 event 看結構：
                    # else:
                    #     print("[ws_mock_realtime] DEBUG EVENT:", json.dumps(event, ensure_ascii=False))

            await asyncio.gather(
                pump_client_to_openai(),
                pump_openai_to_client(),
            )

    except Exception as e:
        print("[ws_mock_realtime] error:", e)
        try:
            await client_ws.send_text(json.dumps({
                "type": "error",
                "message": f"realtime websocket error: {e}",
            }))
        except Exception:
            pass
    finally:
        await session.close()
        try:
            await client_ws.close()
        except Exception:
            pass
        print("[ws_mock_realtime] closed")

@app.get("/mock_media/{session_id}/{index}")
async def get_mock_media(session_id: str, index: int):
    path = mock_interview.MOCK_MEDIA_DIR / f"{session_id}_{index}.webm"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(path, media_type="video/webm")


# ---------- TTS Request model ----------

class TTSRequest(BaseModel):
    text: str
    session_id: str    # 用來從 mock session 讀 interviewer_profile
# ---------- Voice pool ----------

VOICE_POOLS = {
    "male": ["onyx", "echo"],
    # female：只留偏女性的聲音
    "female": ["fable", "shimmer", "nova", "coral"],
    # neutral：放 alloy + 中性
    "neutral": ["alloy", "sage", "ballad", "ash"],
}

ALL_VOICES = list({v for lst in VOICE_POOLS.values() for v in lst})


def pick_voice(gender: str | None) -> str:
    """依照使用者選的 gender 選一個 voice."""
    if not gender or gender == "auto":
        return random.choice(ALL_VOICES)

    gender = gender.lower()

    # 如果剛好傳的是某個 voice 名稱，就直接用
    if gender in ALL_VOICES:
        return gender

    # 否則當成 gender key
    if gender in VOICE_POOLS:
        return random.choice(VOICE_POOLS[gender])

    # fallback
    return "alloy"


def combine_style_and_role_for_tts(
    role_desc: str | None,
    style_desc: str | None,
    extra_notes: str | None = None,
) -> str:
    """
    把 interviewer 的 role + style + extra notes 組成給 TTS 的 instructions。
    """
    parts = []

    if role_desc:
        parts.append(f"Speak like this kind of interviewer: {role_desc}.")

    if style_desc:
        parts.append(f"The tone and behavior should match this description: {style_desc}.")

    if extra_notes:
        parts.append(f"Additional interviewer persona notes: {extra_notes}")

    parts.append(
        "Sound like a realistic, professional interviewer in an English job interview. "
        "Be clear and human-like, not robotic."
    )

    return " ".join(parts)



# ---------- API endpoint ----------

@app.post("/api/tts_question")
async def tts_question(req: TTSRequest):
    """
    用 session_id 讀取 mock session 裡的 interviewer_profile，
    根據 gender / role_resolved / style_resolved / extra_notes 來決定 voice + instructions，
    然後把 text 變成 mp3 回傳。

    ✅ 聲音只在這個 session 第一次 TTS 時決定，之後全部沿用同一個 voice。
    """
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    # 1) load mock session
    try:
        session = mock_interview.load_mock_session(req.session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Mock session not found")

    interviewer_profile = session.get("interviewer_profile") or {}

    gender = interviewer_profile.get("gender")               # "male" / "female" / "auto"
    role_desc = interviewer_profile.get("role_resolved")
    style_desc = interviewer_profile.get("style_resolved")
    extra_notes = interviewer_profile.get("extra_notes")
    tts_persona = interviewer_profile.get("tts_persona")

    # 2) 決定 voice：如果這個 session 已經有 tts_voice，就沿用；沒有才挑一次
    selected_voice = interviewer_profile.get("tts_voice")
    if not selected_voice:
        selected_voice = pick_voice(gender)
        interviewer_profile["tts_voice"] = selected_voice
        session["interviewer_profile"] = interviewer_profile
        # ⭐ 寫回檔案，之後這個 session 的所有題目就都用同一個 voice
        mock_interview.save_mock_session(session)

    # 3) 組 instructions：優先用 tts_persona，沒有才自己拼
    if tts_persona:
        tts_instructions = (
            f"{tts_persona} "
            "Sound like a realistic, professional interviewer in an English job interview. "
            "Be clear and human-like, not robotic."
        )
    else:
        tts_instructions = combine_style_and_role_for_tts(
            role_desc,
            style_desc,
            extra_notes,
        )

    print("[TTS] voice:", selected_voice)
    print("[TTS] instructions:", tts_instructions)

    # 4) 呼叫 OpenAI TTS
    try:
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=selected_voice,
            input=text,
            response_format="mp3",
            instructions=tts_instructions,
        ) as response:
            with NamedTemporaryFile(suffix=".mp3") as tmp:
                response.stream_to_file(tmp.name)
                tmp.seek(0)
                audio_bytes = tmp.read()

    except Exception as e:
        print("TTS error:", e)
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=tts.mp3"},
    )