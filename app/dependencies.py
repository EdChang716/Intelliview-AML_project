from fastapi.templating import Jinja2Templates
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
