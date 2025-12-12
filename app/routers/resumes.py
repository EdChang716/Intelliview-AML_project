from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import json
import shutil
from pathlib import Path

from app.dependencies import templates
from core.config import USER_DATA_DIR
from core.embeddings import build_resume_embeddings
from parsers.resume_parser import (
    extract_pdf_text,
    parse_resume_entries,
    extract_metadata_sections,
    extract_structured_education,
)

router = APIRouter()

@router.get("/resume", response_class=HTMLResponse, name="resume_page")
async def resume_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/api/upload_resume")
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

    # 20251209 Append 
    # =================================================
    # 【新增】防呆機制：如果解析失敗，塞一個預設值
    # =================================================
    if not education_structured:
        education_structured = [{
            "school_name": "School Not Found (Please Edit)",
            "degree": "",
            "dates": "",
            "gpa": ""
        }]
    # =================================================

    # Save raw parse results (edited version will be saved later)
    with open(parsed_dir / "experience_entries.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


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

class SaveResumePayload(BaseModel):
    project_id: str
    entries: list[dict]
    metadata: dict
    education_structured: list[dict]

@router.post("/api/save_resume")
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

@router.get("/api/resumes")
async def list_resumes():
    """List all available resumes (parsed directories)"""
    parsed_root = USER_DATA_DIR / "parsed"
    resumes = []
    if parsed_root.exists():
        for folder in parsed_root.iterdir():
            if folder.is_dir():
                # We can try to read metadata to get more info, or just return ID
                resumes.append({"project_id": folder.name})
    # Sort alphabetical
    resumes.sort(key=lambda x: x["project_id"])
    return {"resumes": resumes}

@router.get("/api/resume/{project_id}")
async def get_resume(project_id: str):
    """Get full data for edit mode"""
    parsed_dir = USER_DATA_DIR / "parsed" / project_id
    if not parsed_dir.exists():
        return JSONResponse(status_code=404, content={"detail": "Resume not found"})

    # Prefer edited versions, fallback to raw
    entries_path = parsed_dir / "experience_entries_edited.json"
    if not entries_path.exists():
        entries_path = parsed_dir / "experience_entries.json"

    metadata_path = parsed_dir / "metadata.json"
    education_path = parsed_dir / "education_structured.json"

    entries = []
    if entries_path.exists():
        entries = json.loads(entries_path.read_text(encoding="utf-8"))
    
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    education_structured = []
    if education_path.exists():
        education_structured = json.loads(education_path.read_text(encoding="utf-8"))

    return {
        "project_id": project_id,
        "entries": entries,
        "metadata": metadata,
        "education_structured": education_structured
    }

@router.delete("/api/resume/{project_id}")
async def delete_resume(project_id: str):
    # Remove raw, parsed, embeddings
    dirs_to_remove = [
        USER_DATA_DIR / "raw" / project_id,
        USER_DATA_DIR / "parsed" / project_id,
        USER_DATA_DIR / "embeddings" / project_id,
    ]
    for d in dirs_to_remove:
        if d.exists() and d.is_dir():
            shutil.rmtree(d)
    
    return {"status": "ok", "deleted": project_id}
