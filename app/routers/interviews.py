from fastapi import APIRouter, Request, HTTPException, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import shutil
from pathlib import Path

from app.dependencies import templates
from core.config import USER_DATA_DIR, SESSION_MEDIA_DIR
from core.profiles import load_job_profiles, load_all_profiles
from core.sessions import load_session, log_practice_turn, get_asked_questions
from core.retrieval import retrieve_bullets_for_profile, get_bullets_for_entry
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

router = APIRouter()

@router.get("/practice/{profile_id}", response_class=HTMLResponse, name="practice_page")
async def practice_page(request: Request, profile_id: str, resume_id: Optional[str] = None):
    profiles = load_job_profiles()
    profile = next((p for p in profiles if p.get("profile_id") == profile_id), None)
    job_title = profile.get("job_title", "Interview Practice") if profile else "Interview Practice"
    company = profile.get("company", "") if profile else ""
    
     # If no resume_id provided, get first available resume
    if not resume_id:
        parsed_root = USER_DATA_DIR / "parsed"
        if parsed_root.exists():
            folders = [f.name for f in parsed_root.iterdir() if f.is_dir()]
            resume_id = folders[0] if folders else None

    return templates.TemplateResponse(
        "practice.html",
        {
            "request": request, 
            "profile_id": profile_id,
            "resume_id": resume_id,
            "job_title": job_title,
            "company": company
        },
    )

class NextQuestionRequest(BaseModel):
    profile_id: str
    resume_id: str
    mode: str  # "auto" | "behavioral" | "project" | "technical" | "case" | "custom"
    behavioral_type: Optional[str] = None
    entry_key: Optional[str] = None
    prev_answer: Optional[str] = None
    custom_question: Optional[str] = None

@router.post("/api/next_question")
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

        bullets = retrieve_bullets_for_profile(req.profile_id, question, req.resume_id, top_k=5)
        tag = "Auto (from JD)"
        behavioral_type = None
        entry_key = None

    # === behavioral: from question bank + subtype + avoid duplicates ===
    elif mode == "behavioral":
        subtype = req.behavioral_type or "random"
        question = get_behavioral_question(req.profile_id, subtype)
        bullets = retrieve_bullets_for_profile(req.profile_id, question, req.resume_id, top_k=5)
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
        bullets = retrieve_bullets_for_profile(req.profile_id, question, req.resume_id, top_k=5)
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
        bullets = retrieve_bullets_for_profile(req.profile_id, question, req.resume_id, top_k=5)
        tag = "Case interview question"
        behavioral_type = None
        entry_key = None

    # === custom: custom question from frontend ===
    elif mode == "custom":
        if not req.custom_question:
            raise HTTPException(status_code=400, detail="custom_question is required for custom mode")

        question = req.custom_question
        bullets = retrieve_bullets_for_profile(req.profile_id, question, req.resume_id, top_k=5)
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
    resume_id: str
    question: str


@router.post("/api/retrieve_bullets")
async def api_retrieve_bullets(req: BulletsRequest):
    bullets = retrieve_bullets_for_profile(req.profile_id, req.question, req.resume_id, top_k=3)
    return {"bullets": bullets}

class CoachChatRequest(BaseModel):
    profile_id: str
    resume_id: str
    mode: str
    question: str
    user_message: str
    sample_answer: Optional[str] = None
    bullets: Optional[List[Dict[str, Any]]] = None
    history: Optional[List[Dict[str, str]]] = None  # [{role, content}, ...]

@router.post("/api/coach_chat")
async def api_coach_chat(req: CoachChatRequest):
    profiles = load_job_profiles()
    profile = next((p for p in profiles if p.get("profile_id") == req.profile_id), None)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    jd_text = profile.get("jd_text", "")

    # If frontend did not send bullets, perform RAG lookup
    if req.bullets:
        bullets = req.bullets
    else:
        bullets = retrieve_bullets_for_profile(req.profile_id, req.question, req.resume_id, top_k=5)

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

    from core.llm_client import client as rag_client  # avoid name conflict

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
    resume_id: str
    question: str
    mode: str
    behavioral_type: Optional[str] = None
    entry_key: Optional[str] = None
    user_answer: Optional[str] = None
    bullets: Optional[List[Dict[str, Any]]] = None

@router.post("/api/generate_sample_answer")
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
        bullets = retrieve_bullets_for_profile(req.profile_id, req.question, req.resume_id, top_k=5)

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
    resume_id: str
    question: str
    mode: str
    behavioral_type: Optional[str] = None
    entry_key: Optional[str] = None
    user_answer: Optional[str] = None
    bullets: Optional[List[Dict[str, Any]]] = None
    sample_answer: Optional[str] = None
    thread_id: Optional[str] = None
    is_followup: bool = False

@router.post("/api/save_user_answer")
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
        bullets = retrieve_bullets_for_profile(req.profile_id, req.question, req.resume_id, top_k=5)

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
    resume_id: str
    mode: str  # "auto" | "behavioral" | "project" | "custom"
    base_question: str  # main question for this thread
    user_answer: Optional[str] = None  # latest answer to base_question
    thread_id: Optional[str] = None
    entry_key: Optional[str] = None
    bullets: Optional[List[Dict[str, Any]]] = None

MAX_FOLLOWUPS_PER_THREAD = 3

@router.post("/api/followup_question")
async def api_followup_question(req: FollowupQuestionRequest):
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
        bullets = retrieve_bullets_for_profile(req.profile_id, req.base_question, req.resume_id, top_k=5)

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

@router.post("/api/save_user_answer_with_media")
async def api_save_user_answer_with_media(
    meta: str = Form(...),
    media: UploadFile | None = File(None),
):
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
    saved_media_path: Optional[Path] = None

    # meta may include media_meta: {type, durationMs}
    media_meta = meta_dict.get("media_meta") or {}
    if media_meta:
        media_type = media_meta.get("type")

    if has_media_upload:
        content_type = media.content_type or ""
        if "video" in content_type:
            ext = ".webm"
        elif "audio" in content_type:
            ext = ".webm"
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

        media_filename = str(media_path.relative_to(SESSION_MEDIA_DIR))
        saved_media_path = media_path

        if media_type is None:
            if "video" in content_type:
                media_type = "video"
            elif "audio" in content_type:
                media_type = "audio"

    # ---------- transcription if no text but media exists ----------
    transcript = ""
    if not has_text and saved_media_path:
        # call transcription
        try:
            transcript = transcribe_media(saved_media_path)
            req.user_answer = transcript
        except Exception as e:
            print("Transcription failed:", e)
            req.user_answer = "(Transcription failed)"
    
    # Evaluate
    # Use bullets from frontend if provided; otherwise perform RAG
    if req.bullets:
        bullets = req.bullets
    else:
        bullets = retrieve_bullets_for_profile(req.profile_id, req.question, req.resume_id, top_k=5)

    eval_result = evaluate_answer(
        question=req.question,
        jd_text=jd_text,
        bullets=bullets,
        user_answer=req.user_answer or "",
    )

    # Same saving logic as save_user_answer...
    score = eval_result.get("overall_score")
    if score is None:
        score = eval_result.get("score", 5)

    strengths = (eval_result.get("strengths") or "").strip()
    improvements = (
        eval_result.get("improvements_overview")
        or eval_result.get("improvements")
        or ""
    ).strip()

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

    eval_result_out = dict(eval_result)
    eval_result_out.setdefault("score", score)
    eval_result_out.setdefault("improvements", improvements)
    # Return transcript too so frontend can display it
    eval_result_out["transcript"] = req.user_answer

    return eval_result_out

@router.get("/mock_settings", response_class=HTMLResponse, name="mock_settings_page")
async def mock_settings_page(
    request: Request,
    resume_id: Optional[str] = None,
):
    # Scan parsed/ for available resume versions
    parsed_root = USER_DATA_DIR / "parsed"
    resume_ids: list[str] = []
    if parsed_root.exists():
        for folder in parsed_root.iterdir():
            if folder.is_dir():
                resume_ids.append(folder.name)
    resume_ids.sort()

    all_profiles = load_job_profiles()

    return templates.TemplateResponse(
        "mock_settings.html",
        {
            "request": request,
            "resume_ids": resume_ids,
            "default_resume_id": resume_id,
            "all_profiles": all_profiles,
        },
    )

@router.get("/mock_interview")
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

    tts_persona = (
        f"{resolved_role}. {resolved_style}. "
        f"{extra_notes}" if extra_notes else f"{resolved_role}. {resolved_style}."
    )

    interviewer_profile = {
        "gender": interviewer_gender,
        "role_preset": role_preset,
        "role_resolved": resolved_role,
        "style_preset": style_preset,
        "style_resolved": resolved_style,
        "extra_notes": extra_notes,
        "tts_persona": tts_persona,
    }

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

    session_config = {
        "session_id": session["session_id"],
        "profile_id": profile_id,
        "resume_id": resume_id,
        "mode": mode,
        "length_type": length_type,
        "hint_level": hint_level,
        "num_questions": session.get("num_questions"),
        "time_limit": session.get("time_limit"),
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

@router.get("/profiles/{profile_id}/mock_history")
async def mock_history_index(request: Request, profile_id: str):
    sessions = mock_interview.list_mock_sessions_for_profile(profile_id)
    all_profiles = load_all_profiles()

    return templates.TemplateResponse(
        "mock_history.html",
        {
            "request": request,
            "profile_id": profile_id,
            "sessions": sessions,
            "all_profiles": all_profiles,
        },
    )

@router.get("/mock/{session_id}")
def mock_report_page(request: Request, session_id: str):
    report = mock_interview.load_mock_result(session_id)
    return templates.TemplateResponse(
        "mock_report.html",
        {
            "request": request,
            "report": report,
        }
    )

@router.get("/profiles/{profile_id}/mock_history/{session_id}")
async def mock_report_page_profile(request: Request, profile_id: str, session_id: str):
    report = mock_interview.load_mock_result(session_id)
    return templates.TemplateResponse(
        "mock_report.html",
        {"request": request, "profile_id": profile_id, "report": report},
    )

# ===== Mock Interview API Endpoints =====

class MockNextQuestionRequest(BaseModel):
    session_id: str
    index: int
    seconds_left: Optional[int] = None

@router.post("/api/mock_next_question")
async def api_mock_next_question(req: MockNextQuestionRequest):
    """Get the next question for the mock interview"""
    try:
        question_data = mock_interview.get_question_for_index(
            session_id=req.session_id,
            index=req.index,
            seconds_left=req.seconds_left
        )
        return question_data
    except Exception as e:
        print(f"Error in mock_next_question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class MockAnswerRequest(BaseModel):
    session_id: str
    index: int
    question_id: str
    question_text: str
    realtime_transcript: Optional[str] = None

@router.post("/api/mock_answer")
async def api_mock_answer(
    meta: str = Form(...),
    media: UploadFile = File(...)
):
    """Submit an answer for a mock interview question"""
    try:
        meta_dict = json.loads(meta)
        req = MockAnswerRequest(**meta_dict)

        # Save the video file
        session_id = req.session_id
        index = req.index

        # Create directory for this session's media
        from core.config import USER_DATA_DIR
        MOCK_MEDIA_DIR = USER_DATA_DIR / "mock" / "media"
        MOCK_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

        # Save video file with format expected by finalize: {session_id}_{index}.webm
        video_filename = f"{session_id}_{index}.webm"
        video_path = MOCK_MEDIA_DIR / video_filename

        content = await media.read()
        video_path.write_bytes(content)

        # Transcribe the video to get transcript
        transcript = ""
        if video_path.exists():
            try:
                transcript = transcribe_media(video_path)
            except Exception as e:
                print(f"Transcription failed for {video_path}: {e}")
                transcript = "(Transcription failed)"

        # Load session and add answer
        session = mock_interview.load_mock_session(session_id)

        if "answers" not in session:
            session["answers"] = []

        answer_data = {
            "index": index,
            "question_id": req.question_id,
            "question_text": req.question_text,
            "video_path": str(video_path.relative_to(USER_DATA_DIR)),
            "realtime_transcript": req.realtime_transcript or "",
            "transcript": transcript,  # Use Whisper transcription
            "timestamp": datetime.utcnow().isoformat()
        }

        session["answers"].append(answer_data)
        mock_interview.save_mock_session(session)

        return {"success": True, "message": "Answer saved"}

    except Exception as e:
        print(f"Error in mock_answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class MockFinishRequest(BaseModel):
    session_id: str

@router.post("/api/mock_finish")
async def api_mock_finish(meta: str = Form(...)):
    """Finalize the mock interview session"""
    try:
        meta_dict = json.loads(meta)
        req = MockFinishRequest(**meta_dict)
        result = mock_interview.finalize_mock_session(req.session_id)
        return result
    except Exception as e:
        print(f"Error in mock_finish: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class TTSQuestionRequest(BaseModel):
    text: str
    session_id: str

@router.post("/api/tts_question")
async def api_tts_question(req: TTSQuestionRequest):
    """Convert question text to speech using OpenAI TTS"""
    try:
        from openai import OpenAI
        client = OpenAI()

        # Load session to get interviewer voice preference
        session = mock_interview.load_mock_session(req.session_id)
        interviewer_profile = session.get("interviewer_profile", {})
        gender = interviewer_profile.get("gender", "auto")

        # Map gender to OpenAI voice
        voice_map = {
            "male": "onyx",
            "female": "nova",
            "neutral": "echo",
            "auto": "alloy"
        }
        voice = voice_map.get(gender, "alloy")

        # Generate speech
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=req.text
        )

        # Return audio as streaming response
        from fastapi.responses import StreamingResponse
        import io

        audio_bytes = io.BytesIO(response.content)
        return StreamingResponse(audio_bytes, media_type="audio/mpeg")

    except Exception as e:
        print(f"Error in tts_question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mock_media/{session_id}/{index}")
async def get_mock_video(session_id: str, index: int):
    """Serve mock interview video files"""
    try:
        from fastapi.responses import FileResponse

        MOCK_MEDIA_DIR = USER_DATA_DIR / "mock" / "media"
        video_path = MOCK_MEDIA_DIR / f"{session_id}_{index}.webm"

        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video not found")

        return FileResponse(video_path, media_type="video/webm")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error serving mock video: {e}")
        raise HTTPException(status_code=500, detail=str(e))
