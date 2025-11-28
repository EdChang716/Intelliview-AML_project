from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, WebSocket
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import json
import os
import sys
import random
import shutil
import asyncio
from tempfile import NamedTemporaryFile

import aiohttp

from parsers.resume_parser import (
    extract_pdf_text,
    parse_resume_entries,
    extract_metadata_sections,
    extract_structured_education,
)
from core.embeddings import build_resume_embeddings
from core.llm_client import client

# from the module you built
from core.config import USER_DATA_DIR
from core.profiles import load_job_profiles
from core.retrieval import (
    retrieve_bullets_for_profile,
    get_bullets_for_entry,
    load_resume_entries_and_embs,
)
from core.sessions import (
    load_session,
    get_asked_questions,
    log_practice_turn,
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
ROOT_DIR = APP_DIR  # project root directory (first definition)
sys.path.append(str(ROOT_DIR))

USER_DATA_DIR.mkdir(exist_ok=True)
JOB_PROFILES_PATH = USER_DATA_DIR / "job_profiles.json"
SESSION_MEDIA_DIR = USER_DATA_DIR / "session_media"
SESSION_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Intelliview Coach")

# static / templates (first mount)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

timestamp = datetime.now(timezone.utc).isoformat()
now = datetime.utcnow().isoformat() + "Z"


# Save job profiles JSON to user_data (local override of imported name)
def save_job_profiles(profiles: list[dict]) -> None:
    JOB_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOB_PROFILES_PATH.write_text(
        json.dumps({"profiles": profiles}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =========================
# Path configuration (second definition of dirs, used by the rest of the file)
# =========================
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
USER_DATA_DIR = ROOT_DIR / "user_data"
USER_DATA_DIR.mkdir(exist_ok=True)
JOB_PROFILES_PATH = USER_DATA_DIR / "job_profiles.json"

# Audio / video will be stored here: user_data/session_media/<profile_id>/xxx.webm
SESSION_MEDIA_DIR = USER_DATA_DIR / "session_media"
SESSION_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(ROOT_DIR))

app = FastAPI(title="Intelliview Coach")

# Static files (CSS, JS)
static_dir = APP_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Media files (audio/video)
app.mount("/media", StaticFiles(directory=str(SESSION_MEDIA_DIR)), name="media")

templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


# Ensure raw/parsed directories exist for a given project_id
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
# Frontend pages
# =========================

# Home page: landing page of the app
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


# Resume editor page
@app.get("/resume", response_class=HTMLResponse, name="resume_page")
async def resume_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Profiles list page, including basic practice stats
@app.get("/profiles", response_class=HTMLResponse, name="profiles_page")
async def profiles_page(request: Request):
    profiles = load_job_profiles()
    # Sort profiles by updated_at (newest first)
    profiles_sorted = sorted(
        profiles,
        key=lambda p: p.get("updated_at", ""),
        reverse=True,
    )

    # Attach stats summary for each profile
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
            "profiles": enriched,
        },
    )


# Page to create a new profile, listing available resume versions
@app.get("/profiles/new", response_class=HTMLResponse, name="new_profile_page")
async def new_profile_page(
    request: Request,
    resume_id: str | None = None,
):
    # Scan parsed/ for existing resume versions (folder names)
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


# API: get a single profile's details
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


# Load all profiles from user_data/job_profiles.json
def load_all_profiles():
    with open("user_data/job_profiles.json", "r") as f:
        data = json.load(f)
    # Only take the inner list of profiles
    return data.get("profiles", [])


# Practice history page for a given profile
@app.get("/profiles/{profile_id}/history", response_class=HTMLResponse)
async def practice_history_page(request: Request, profile_id: str):
    stats = get_practice_stats(profile_id)
    session = load_session(profile_id)
    turns = session.get("turns", [])

    all_profiles = load_all_profiles()

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "profile_id": profile_id,
            "stats": stats,
            "turns": turns,
            "all_profiles": all_profiles,
        },
    )


# Practice page for a given profile
@app.get("/practice/{profile_id}", response_class=HTMLResponse, name="practice_page")
async def practice_page(request: Request, profile_id: str):
    return templates.TemplateResponse(
        "practice.html",
        {"request": request, "profile_id": profile_id},
    )


# API: list resume entries for a given profile
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


# API: get summary practice stats for a profile
@app.get("/api/practice_stats/{profile_id}")
async def api_practice_stats(profile_id: str):
    stats = get_practice_stats(profile_id)
    return stats


# API: get practice history turns for a profile
@app.get("/api/practice_history/{profile_id}")
async def api_practice_history(profile_id: str):
    session = load_session(profile_id)
    turns = session.get("turns", [])
    # You can optionally sort or truncate here if needed
    return {"turns": turns}


# Mock settings page for configuring mock interview session
@app.get("/mock_settings", response_class=HTMLResponse, name="mock_settings_page")
async def mock_settings_page(
    request: Request,
    resume_id: str | None = None,
):
    # Scan parsed/ for available resume versions
    parsed_root = USER_DATA_DIR / "parsed"
    resume_ids: list[str] = []
    if parsed_root.exists():
        for folder in parsed_root.iterdir():
            if folder.is_dir():
                resume_ids.append(folder.name)
    resume_ids.sort()

    all_profiles = load_job_profiles()  # For dropdown list of profiles

    return templates.TemplateResponse(
        "mock_settings.html",
        {
            "request": request,
            "resume_ids": resume_ids,
            "default_resume_id": resume_id,
            "all_profiles": all_profiles,
        },
    )


# Mock interview page: creates mock session and passes session_config to frontend
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

    # Interviewer settings from mock_settings.html
    interviewer_gender = q.get("interviewer_gender", "auto")

    role_preset = q.get("interviewer_role") or "senior_engineer"
    role_custom = q.get("interviewer_role_custom") or ""

    style_preset = q.get("interviewer_style_preset") or "balanced"
    style_custom = q.get("interviewer_style_custom") or ""

    extra_notes = (q.get("interviewer_extra_notes") or "").strip()

    # Map preset role to description
    def resolve_role(preset: str, custom: str) -> str:
        if preset == "custom":
            return custom or "an interviewer for this role"
        mapping = {
            "senior_engineer": "a senior data / ML / SWE engineer on the team you’d work with",
            "hiring_manager": "the hiring manager who cares about team fit, ownership, and impact",
            "recruiter": "a recruiter or HR partner focusing on overall fit and communication",
            "peer_teammate": "a future teammate who wants to know what it’s like to work with you day to day",
            "executive": "a director or VP who cares about business impact and prioritization",
        }
        return mapping.get(preset, "an interviewer for this role")

    # Map preset style to description
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

    # Persona string that will be sent to TTS instructions
    tts_persona = (
        f"{resolved_role}. {resolved_style}. "
        f"{extra_notes}" if extra_notes else f"{resolved_role}. {resolved_style}."
    )

    # Build interviewer_profile and store in session
    interviewer_profile = {
        "gender": interviewer_gender,
        "role_preset": role_preset,
        "role_resolved": resolved_role,
        "style_preset": style_preset,
        "style_resolved": resolved_style,
        "extra_notes": extra_notes,
        # This field will be used to construct TTS instructions
        "tts_persona": tts_persona,
    }

    # Create mock session with interviewer_profile
    session = mock_interview.create_mock_session(
        profile_id=profile_id,
        resume_id=resume_id,
        mode=mode,
        length_type=length_type,
        hint_level=hint_level,
        num_questions=num_questions_int,
        time_limit=time_limit_int,
        interviewer_profile=interviewer_profile,
    )

    # Frontend will use SESSION_CONFIG to call /api/tts_question
    session_config = {
        "session_id": session["session_id"],
        "profile_id": profile_id,
        "resume_id": resume_id,
        "mode": mode,
        "length_type": length_type,
        "hint_level": hint_level,
        "num_questions": session.get("num_questions"),
        "time_limit": session.get("time_limit"),
        # For JS to set voice / instructions
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


# Mock history index page for a profile
@app.get("/profiles/{profile_id}/mock_history")
async def mock_history_index(request: Request, profile_id: str):
    sessions = mock_interview.list_mock_sessions_for_profile(profile_id)
    all_profiles = load_all_profiles()  # Same as practice history: used for profile switcher

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
# Single mock result pages
# ================================

# Mock report page for a single session (generic)
@app.get("/mock/{session_id}")
def mock_report_page(request: Request, session_id: str):
    """
    Display a single mock interview report.
    """
    report = mock_interview.load_mock_result(session_id)
    return templates.TemplateResponse(
        "mock_report.html",
        {
            "request": request,
            "report": report,
        }
    )


# Mock report page scoped under a given profile
@app.get("/profiles/{profile_id}/mock_history/{session_id}")
async def mock_report_page_profile(request: Request, profile_id: str, session_id: str):
    report = mock_interview.load_mock_result(session_id)
    return templates.TemplateResponse(
        "mock_report.html",
        {"request": request, "profile_id": profile_id, "report": report},
    )


# =========================
# API: upload resume and parse
# =========================

# Upload PDF resume and parse into entries/metadata/education
@app.post("/api/upload_resume")
async def upload_resume(
    project_id: str = Form(...),
    file: UploadFile = File(...)
):
    # Prepare directories
    raw_dir = USER_DATA_DIR / "raw" / project_id
    parsed_dir = USER_DATA_DIR / "parsed" / project_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    # Always save as resume.pdf
    resume_path = raw_dir / "resume.pdf"
    content = await file.read()
    with open(resume_path, "wb") as f:
        f.write(content)

    # Parse using your own PDF parser
    raw_text = extract_pdf_text(str(resume_path))
    entries = parse_resume_entries(raw_text)
    metadata = extract_metadata_sections(raw_text)
    education_structured = extract_structured_education(raw_text)

    # Save raw parse results (edited version will be saved later)
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
# API: save edited resume result
# =========================

class SaveResumePayload(BaseModel):
    project_id: str
    entries: list[dict]
    metadata: dict
    education_structured: list[dict]


# Save edited resume and rebuild embeddings
@app.post("/api/save_resume")
async def save_resume(payload: SaveResumePayload):
    project_id = payload.project_id
    parsed_dir = USER_DATA_DIR / "parsed" / project_id
    parsed_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save edited version
    with open(parsed_dir / "experience_entries_edited.json", "w", encoding="utf-8") as f:
        json.dump(payload.entries, f, ensure_ascii=False, indent=2)

    with open(parsed_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(payload.metadata, f, ensure_ascii=False, indent=2)

    with open(parsed_dir / "education_structured.json", "w", encoding="utf-8") as f:
        json.dump(payload.education_structured, f, ensure_ascii=False, indent=2)

    # 2) Build embeddings with fine-tuned model
    #    → user_data/embeddings/{project_id}/resume_bullets.npy
    try:
        build_resume_embeddings(project_id)
        built = True
    except Exception as e:
        # Do not crash the API; just return a flag
        print("Error building embeddings:", e)
        built = False

    # 3) Return status to frontend
    return JSONResponse(
        content={
            "status": "ok",
            "project_id": project_id,
            "embeddings_built": built
        }
    )


# Create or update a job profile (linking JD and resume)
@app.post("/api/create_job_profile")
async def create_job_profile(payload: JobProfileCreate):
    profiles = load_job_profiles()

    now = datetime.utcnow().isoformat() + "Z"

    # If profile_id already exists, update it
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
    mode: str  # "auto" | "behavioral" | "project" | "technical" | "case" | "custom"
    behavioral_type: Optional[str] = None
    entry_key: Optional[str] = None
    prev_answer: Optional[str] = None
    custom_question: Optional[str] = None


# API: get the next practice question for a profile (multiple modes)
@app.post("/api/next_question")
async def api_next_question(req: NextQuestionRequest):
    profiles = load_job_profiles()
    profile = next((p for p in profiles if p.get("profile_id") == req.profile_id), None)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    jd_text = profile.get("jd_text", "")
    mode = (req.mode or "auto").lower()

    # === auto: JD-based question, avoid duplicates ===
    if mode == "auto":
        asked = get_asked_questions(req.profile_id, mode="auto")
        question = call_llm_for_question(jd_text, mode="auto", avoid=asked)

        bullets = retrieve_bullets_for_profile(req.profile_id, question, top_k=5)
        tag = "Auto (from JD)"
        behavioral_type = None
        entry_key = None

    # === behavioral: from question bank + subtype + avoid duplicates ===
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

        # Build previous QAs for this entry from session
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

        # Append the latest answer (prev_answer) to context if available
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

    # === technical: JD-based technical question ===
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

    # === case: JD-based case reasoning question ===
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

    # === custom: custom question from frontend ===
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


# API: retrieve top bullets for a given question (RAG only)
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
    history: Optional[List[Dict[str, str]]] = None  # [{role, content}, ...]


# API: interview coach chat for a single question
@app.post("/api/coach_chat")
async def api_coach_chat(req: CoachChatRequest):
    """
    Coach chat:
    - Always has a current question
    - sample_answer may be empty (not generated yet)
    - bullets may be omitted (server will perform RAG)
    - history is used to keep multiple coach chat turns
    """
    profiles = load_job_profiles()
    profile = next((p for p in profiles if p.get("profile_id") == req.profile_id), None)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    jd_text = profile.get("jd_text", "")

    # If frontend did not send bullets, perform RAG lookup
    if req.bullets:
        bullets = req.bullets
    else:
        bullets = retrieve_bullets_for_profile(req.profile_id, req.question, top_k=5)

    # Build bullet context block
    bullet_lines = []
    for b in bullets:
        entry = b.get("entry") or "Unknown entry"
        text = b.get("text") or ""
        bullet_lines.append(f"- [{entry}] {text}")
    bullet_block = "\n".join(bullet_lines) if bullet_lines else "(none)"

    # Keep only the last few turns of history
    history = req.history or []
    trimmed_history = history[-8:]  # max 8 turns

    # System + context prompt
    system_msg = (
        "You are an interview coach helping a candidate refine their answer, "
        "do not give the user the sample answer unless they ask for it. "
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

    # Add preserved history
    for m in trimmed_history:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not content:
            continue
        messages.append({"role": role, "content": content})

    # Append the latest user message
    messages.append({"role": "user", "content": req.user_message})

    from core.rag_pipeline import client as rag_client  # avoid name conflict

    resp = rag_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.6,
    )
    reply = resp.choices[0].message.content.strip()

    return {
        "reply": reply,
        "bullets": bullets,  # frontend can update sidebar if needed
    }


class SampleAnswerRequest(BaseModel):
    profile_id: str
    question: str
    mode: str
    behavioral_type: Optional[str] = None
    entry_key: Optional[str] = None
    user_answer: Optional[str] = None
    bullets: Optional[List[Dict[str, Any]]] = None


# API: generate sample answer for a given question
@app.post("/api/generate_sample_answer")
async def api_generate_sample_answer(req: SampleAnswerRequest):
    profiles = load_job_profiles()
    profile = next((p for p in profiles if p.get("profile_id") == req.profile_id), None)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    jd_text = profile.get("jd_text", "")

    # If bullets not provided, perform RAG lookup
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


# API: save a text-based user answer and evaluate it
@app.post("/api/save_user_answer")
async def api_save_user_answer(req: SaveUserAnswerRequest):
    profiles = load_job_profiles()
    profile = next((p for p in profiles if p.get("profile_id") == req.profile_id), None)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    jd_text = profile.get("jd_text", "")

    if not req.user_answer or not req.user_answer.strip():
        raise HTTPException(status_code=400, detail="user_answer is empty")

    # Use bullets from frontend if provided; otherwise perform RAG
    if req.bullets:
        bullets = req.bullets
    else:
        bullets = retrieve_bullets_for_profile(req.profile_id, req.question, top_k=5)

    # Evaluate the answer
    eval_result = evaluate_answer(
        question=req.question,
        jd_text=jd_text,
        bullets=bullets,
        user_answer=req.user_answer,
    )

    # Backward compatible mapping
    score = eval_result.get("overall_score")
    if score is None:
        score = eval_result.get("score", 5)

    strengths = (eval_result.get("strengths") or "").strip()
    improvements = (
        eval_result.get("improvements_overview")
        or eval_result.get("improvements")
        or ""
    ).strip()

    # Write into session
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

    # Return eval_result with legacy keys for safety
    eval_result_out = dict(eval_result)
    eval_result_out.setdefault("score", score)
    eval_result_out.setdefault("improvements", improvements)

    return eval_result_out


class FollowupQuestionRequest(BaseModel):
    profile_id: str
    mode: str  # "auto" | "behavioral" | "project" | "custom"
    base_question: str  # main question for this thread
    user_answer: Optional[str] = None  # latest answer to base_question
    thread_id: Optional[str] = None
    entry_key: Optional[str] = None
    bullets: Optional[List[Dict[str, Any]]] = None


MAX_FOLLOWUPS_PER_THREAD = 3  # max follow-up questions per thread


# API: generate a follow-up question for a given main question
@app.post("/api/followup_question")
async def api_followup_question(req: FollowupQuestionRequest):
    """
    Generate a follow-up question:
    - Works for all modes (auto/behavioral/project/custom)
    - Uses base_question + QA history + latest user_answer
    - Avoids repeating previous questions in the same thread
    """
    profiles = load_job_profiles()
    profile = next((p for p in profiles if p.get("profile_id") == req.profile_id), None)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    jd_text = profile.get("jd_text", "")
    mode = (req.mode or "auto").lower()

    # 1) bullets: use provided ones or perform RAG
    if req.bullets:
        bullets = req.bullets
    else:
        bullets = retrieve_bullets_for_profile(req.profile_id, req.base_question, top_k=5)

    # 2) get this thread's existing turns from session
    session = load_session(req.profile_id)
    thread_id = req.thread_id or req.base_question  # fallback to base_question as id

    thread_turns = [
        t for t in session.get("turns", [])
        if t.get("thread_id") == thread_id
    ]

    # Count how many followups are already in this thread
    followup_count = sum(1 for t in thread_turns if t.get("is_followup"))
    if followup_count >= MAX_FOLLOWUPS_PER_THREAD:
        return {
            "question": None,
            "done": True,
            "message": "This topic has already been explored with several follow-up questions. Consider moving on to a new question.",
            "thread_id": thread_id,
        }

    # 3) build QA history for the LLM
    qa_history = []
    for t in thread_turns:
        q = t.get("question") or ""
        a = t.get("user_answer") or ""
        if q or a:
            qa_history.append({"question": q, "answer": a})

    # Append latest user answer even if not stored yet
    if req.user_answer:
        qa_history.append({"question": req.base_question, "answer": req.user_answer})

    # 4) avoid repeated questions within this thread
    avoid = set()
    for t in thread_turns:
        q = t.get("question")
        if q:
            avoid.add(q.strip())
    avoid.add(req.base_question.strip())

    # 5) call LLM to generate follow-up question
    followup_q = generate_followup_question(
        jd_text=jd_text,
        mode=mode,
        base_question=req.base_question,
        bullets=bullets,
        qa_history=qa_history,
        avoid=avoid,
    )

    if not followup_q:
        # Model could not generate a sufficiently different follow-up
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
        "tag": f"Follow-up · {mode}",
    }


# API: save user answer along with audio/video media, with transcription fallback
@app.post("/api/save_user_answer_with_media")
async def api_save_user_answer_with_media(
    meta: str = Form(...),
    media: UploadFile | None = File(None),
):
    """
    Frontend sends FormData:
      - meta: JSON string, same shape as SaveUserAnswerRequest (user_answer may be empty)
      - media: audio/video file (optional)

    Flow:
      1. Save media to user_data/session_media/<profile_id>/xxx.webm
      2. If meta has no user_answer but has media → transcribe with Whisper-1
      3. Fill transcript into req.user_answer → run original evaluation + logging
      4. Return evaluation result (plus transcript for frontend display)
    """
    # ----- parse meta -----
    try:
        meta_dict = json.loads(meta)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid meta JSON")

    # Validate with existing Pydantic model
    req = SaveUserAnswerRequest(**meta_dict)

    profiles = load_job_profiles()
    profile = next((p for p in profiles if p.get("profile_id") == req.profile_id), None)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    jd_text = profile.get("jd_text", "")

    # ----- check if we have text and/or media -----
    has_text = bool(req.user_answer and req.user_answer.strip())
    has_media_upload = media is not None

    if not has_text and not has_media_upload:
        raise HTTPException(status_code=400, detail="No text answer or media provided")

    # ---------- handle media file: save to SESSION_MEDIA_DIR ----------
    media_type: Optional[str] = None
    media_filename: Optional[str] = None
    media_duration_ms: Optional[int] = None
    saved_media_path: Optional[Path] = None

    # meta may include media_meta: {type, durationMs}
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

        # Store relative path to SESSION_MEDIA_DIR in session.json
        media_filename = str(media_path.relative_to(SESSION_MEDIA_DIR))
        saved_media_path = media_path

        # If media_type not given, infer from content_type
        if media_type is None:
            if "video" in content_type:
                media_type = "video"
            elif "audio" in content_type:
                media_type = "audio"

    # ---------- transcription if no text but media exists ----------
    transcript_text: Optional[str] = None

    if (not has_text) and saved_media_path is not None:
        try:
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
            has_text = True

    # ---------- bullets: same as text-only endpoint ----------
    if req.bullets:
        bullets = req.bullets
    else:
        bullets = retrieve_bullets_for_profile(req.profile_id, req.question, top_k=5)

    # ---------- evaluation: only if we have final text ----------
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
        # No text even after transcription; record only
        score = None
        strengths = ""
        improvements = ""
        eval_result = {
            "score": score,
            "strengths": strengths,
            "improvements": improvements,
        }

    # Attach transcript to result if available
    if transcript_text:
        eval_result["transcript"] = transcript_text

    # ---------- write into session ----------
    log_practice_turn(
        profile_id=req.profile_id,
        question=req.question,
        sample_answer=req.sample_answer,
        bullets=bullets,
        mode=req.mode,
        behavioral_type=req.behavioral_type,
        entry_key=req.entry_key,
        user_answer=req.user_answer,  # text typed by user or transcribed text
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
    index: int  # 1-based index
    session_config: Dict[str, Any]
    prev_answer: Optional[str] = None
    entry_key: Optional[str] = None  # optional project override


# API: get the next mock question (time/length mode) and combine reaction + question
@app.post("/api/mock_next_question")
async def api_mock_next_question(payload: Dict[str, Any]):
    """
    Body:
    {
      "session_id": "...",
      "index": 0,         # 0-based
      "seconds_left": 900 # time mode only, in seconds
    }
    """
    session_id = payload.get("session_id")
    index_raw = payload.get("index", 0)
    seconds_left = payload.get("seconds_left")   # used in time mode

    if session_id is None:
        raise HTTPException(status_code=400, detail="session_id is required")

    try:
        index = int(index_raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="index must be an integer")

    try:
        # Pass seconds_left into mock_interview
        q = mock_interview.get_question_for_index(
            session_id,
            index,
            seconds_left=seconds_left,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    # Combine interviewer reaction and question text into one bubble
    reaction_text = (q.get("reaction") or "").strip()
    question_text = (q.get("question") or "").strip()

    if reaction_text and question_text:
        combined = f"{reaction_text}\n\n{question_text}"
        q["question"] = combined
    elif reaction_text:
        # Fallback if question is missing (should not normally happen)
        q["question"] = reaction_text

    return JSONResponse(q)


# API: finalize mock session at the end of the interview and generate report
@app.post("/api/mock_finish")
async def api_mock_finish(meta: str = Form(...)):
    """
    Called when the mock interview ends:
      - meta: JSON string, must include {session_id}
    No need to upload all media again because each question was stored via /api/mock_answer.
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


# API: save a single mock interview answer (audio/video) + transcript + short reaction
@app.post("/api/mock_answer")
async def api_mock_answer(
    meta: str = Form(...),
    media: UploadFile = File(...),
):
    """
    Called at the end of each mock question:
      - meta: JSON string, must contain {session_id, index, question_id, question_text}
              (if using realtime transcription, also includes realtime_transcript)
      - media: recorded audio/video for this question (webm)

    Backend:
      1) Save file to user_data/mock/media/<session_id>_<index>.webm
      2) Prefer realtime_transcript; if missing, run batch transcription
      3) Generate a short interviewer reaction to the answer
      4) Save into mock session answers (including reaction and transcript_source)
      5) Optionally insert a follow-up question (not counted in total question count)
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
    # 2) transcription: prefer realtime_transcript
    # -----------------------------
    realtime_text = meta_obj.get("realtime_transcript") or ""
    if not isinstance(realtime_text, str):
        realtime_text = ""
    realtime_text = realtime_text.strip()

    transcript_source = "none"
    transcript_text = ""

    if realtime_text:
        # Normal path: use text from Realtime API
        transcript_text = realtime_text
        transcript_source = "realtime"
        print("Realtime transcript used")
    else:
        print("Realtime transcript not available, using batch transcription")
        # Fallback to batch transcription if realtime not available
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
    # 3) interviewer reaction
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
    # 4) write into session.answers
    # -----------------------------
    session = mock_interview.load_mock_session(session_id)
    answers = session.get("answers") or []

    # Remove previous record for same index
    answers = [a for a in answers if a.get("index") != index]

    answer_obj = {
        "index": index,
        "question_id": question_id,
        "question_text": question_text,
        "transcript": transcript_text or "",
        "reaction": reaction or "",
        "transcript_source": transcript_source,  # realtime/batch/error
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


# WebSocket: bridge between browser and OpenAI Realtime for transcription
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

            # Configure transcription session
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

            # Task: forward messages from client to OpenAI
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

            # Task: forward messages from OpenAI to client, extracting transcript
            async def pump_openai_to_client():
                """
                Convert OpenAI transcription events into:
                  { "type": "transcript", "text": "<current text>" }
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

                    # 1) Partial text (partial or delta)
                    if etype in (
                        "conversation.item.input_audio_transcription.partial",
                        "conversation.item.input_audio_transcription.delta",
                    ):
                        fragment = (
                            event.get("delta")
                            or event.get("transcript")
                            or ""
                        )
                        if fragment:
                            current_text += fragment
                            await client_ws.send_text(json.dumps({
                                "type": "transcript",
                                "text": current_text,
                            }))
                        continue

                    # 2) Completed text
                    if etype == "conversation.item.input_audio_transcription.completed":
                        final_text = event.get("transcript") or ""
                        if final_text:
                            current_text = final_text
                            await client_ws.send_text(json.dumps({
                                "type": "transcript",
                                "text": current_text,
                            }))
                        continue

                    # Other events are ignored for now (can log for debugging)

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


# API: serve recorded mock media file for preview
@app.get("/mock_media/{session_id}/{index}")
async def get_mock_media(session_id: str, index: int):
    path = mock_interview.MOCK_MEDIA_DIR / f"{session_id}_{index}.webm"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(path, media_type="video/webm")


# ---------- TTS request model ----------

class TTSRequest(BaseModel):
    text: str
    session_id: str  # used to look up interviewer_profile from mock session


# ---------- Voice pools ----------

VOICE_POOLS = {
    "male": ["onyx", "echo"],
    "female": ["fable", "shimmer", "nova", "coral"],
    "neutral": ["alloy", "sage", "ballad", "ash"],
}

ALL_VOICES = list({v for lst in VOICE_POOLS.values() for v in lst})


# Pick a voice based on selected gender or explicit voice name
def pick_voice(gender: str | None) -> str:
    """Pick a TTS voice based on user-selected gender or voice name."""
    if not gender or gender == "auto":
        return random.choice(ALL_VOICES)

    gender = gender.lower()

    # If given an exact voice name, use it directly
    if gender in ALL_VOICES:
        return gender

    # Otherwise treat as gender key
    if gender in VOICE_POOLS:
        return random.choice(VOICE_POOLS[gender])

    # Fallback
    return "alloy"


# Build TTS instructions from interviewer role, style, and extra notes
def combine_style_and_role_for_tts(
    role_desc: str | None,
    style_desc: str | None,
    extra_notes: str | None = None,
) -> str:
    """
    Combine interviewer role + style + extra notes into TTS instructions string.
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


# ---------- TTS endpoint ----------

# API: turn interviewer text into speech, using persona stored in mock session
@app.post("/api/tts_question")
async def tts_question(req: TTSRequest):
    """
    Use session_id to read interviewer_profile from the mock session,
    decide voice + instructions based on gender / role / style / extra notes,
    then convert text into an mp3 response.
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

    gender = interviewer_profile.get("gender")  # "male" / "female" / "auto"
    role_desc = interviewer_profile.get("role_resolved")
    style_desc = interviewer_profile.get("style_resolved")
    extra_notes = interviewer_profile.get("extra_notes")
    tts_persona = interviewer_profile.get("tts_persona")

    # 2) select voice: reuse existing session voice if available
    selected_voice = interviewer_profile.get("tts_voice")
    if not selected_voice:
        selected_voice = pick_voice(gender)
        interviewer_profile["tts_voice"] = selected_voice
        session["interviewer_profile"] = interviewer_profile
        # Persist session so all future questions use the same voice
        mock_interview.save_mock_session(session)

    # 3) build instructions: prefer precomputed tts_persona if available
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

    # 4) call OpenAI TTS
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
