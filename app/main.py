from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.dependencies import templates
from core.config import USER_DATA_DIR, SESSION_MEDIA_DIR
from app.routers import resumes, profiles, interviews

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

# Ensure directories exist
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
SESSION_MEDIA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Intelliview Coach")

# Mount static files
static_dir = APP_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount media files
app.mount("/media", StaticFiles(directory=str(SESSION_MEDIA_DIR)), name="media")

# Include Routers
app.include_router(resumes.router)
app.include_router(profiles.router)
app.include_router(interviews.router)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# Dynamic resume selection page
def list_available_resumes():
    """List all available parsed resumes"""
    parsed_dir = USER_DATA_DIR / "parsed"
    if not parsed_dir.exists():
        return {"resumes": []}
    
    resumes = []
    for folder in parsed_dir.iterdir():
        if folder.is_dir():
            # Check if it has the required files
            if (folder / "experience_entries.json").exists():
                resumes.append({
                    "resume_id": folder.name,
                    "name": folder.name
                })
    
    return {"resumes": resumes}