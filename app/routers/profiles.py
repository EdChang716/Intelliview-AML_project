from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import json
from pathlib import Path

from app.dependencies import templates
from core.config import USER_DATA_DIR
from core.profiles import load_job_profiles, save_job_profiles
from core.sessions import load_session, get_practice_stats
from core.retrieval import load_resume_entries_and_embs

router = APIRouter()

@router.get("/profiles", response_class=HTMLResponse, name="profiles_page")
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

@router.get("/profiles/new", response_class=HTMLResponse, name="new_profile_page")
async def new_profile_page(
    request: Request,
    resume_id: Optional[str] = None,
):
    # Scan parsed/ for existing resume versions (folder names)
    # AND check if they have embeddings generated (i.e. are "saved")
    parsed_root = USER_DATA_DIR / "parsed"
    embeddings_root = USER_DATA_DIR / "embeddings"
    
    resume_ids: list[str] = []
    if parsed_root.exists():
        for folder in parsed_root.iterdir():
            if folder.is_dir():
                rid = folder.name
                # Check if embeddings file exists
                emb_path = embeddings_root / rid / "resume_bullets.npy"
                if emb_path.exists():
                    resume_ids.append(rid)
    
    resume_ids.sort()

    return templates.TemplateResponse(
        "new_profile.html",
        {
            "request": request,
            "resume_ids": resume_ids,
            "default_resume_id": resume_id,
        },
    )

@router.get("/api/profile/{profile_id}")
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

class JobProfileCreate(BaseModel):
    profile_id: str
    job_title: str
    company: Optional[str] = None
    jd_text: str
    resume_id: str

@router.post("/api/create_job_profile")
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

@router.delete("/api/profile/{profile_id}")
async def delete_profile(profile_id: str):
    profiles = load_job_profiles()
    before_len = len(profiles)
    profiles = [p for p in profiles if p.get("profile_id") != profile_id]
    
    if len(profiles) == before_len:
        raise HTTPException(status_code=404, detail="Profile not found")

    save_job_profiles(profiles)
    return {"status": "ok", "deleted": profile_id}

@router.get("/api/profile_entries/{profile_id}")
async def api_profile_entries(profile_id: str):
    profiles = load_job_profiles()
    profile = next((p for p in profiles if p.get("profile_id") == profile_id), None)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    resume_id = profile.get("resume_id")
    if not resume_id:
        raise HTTPException(status_code=400, detail="Profile has no resume_id")

    # This might error if embeddings missing, but we filtered creating profiles so it should be safer
    # Still good to catch
    try:
        entries, _ = load_resume_entries_and_embs(resume_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

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

@router.get("/api/practice_stats/{profile_id}")
async def api_practice_stats(profile_id: str):
    stats = get_practice_stats(profile_id)
    return stats

@router.get("/api/practice_history/{profile_id}")
async def api_practice_history(profile_id: str):
    session = load_session(profile_id)
    turns = session.get("turns", [])
    return {"turns": turns}

@router.get("/profiles/{profile_id}/history", response_class=HTMLResponse)
async def practice_history_page(request: Request, profile_id: str):
    stats = get_practice_stats(profile_id)
    session = load_session(profile_id)
    turns = session.get("turns", [])

    profiles = load_job_profiles() # Use load_job_profiles directly

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "profile_id": profile_id,
            "stats": stats,
            "turns": turns,
            "all_profiles": profiles,
        },
    )
