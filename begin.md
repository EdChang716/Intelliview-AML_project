# Intelliview Coach – Local Setup Guide

This guide explains how to install dependencies, set up environment variables, and launch the Intelliview Coach web application locally.

---

## 1. Clone the repository (SSH)

If you have SSH keys set up with GitHub, clone using:

```
git@github.com:EdChang716/Intelliview-AML_project.git
cd Intelliview-AML_project
```

Replace `<your-username>` and `<your-repo>` with the actual GitHub path.

---

## 2. Create and activate a virtual environment

### macOS / Linux
```
python3 -m venv intelliview-env
source intelliview-env/bin/activate
```

### Windows (PowerShell)
```
python -m venv intelliview-env
intelliview-env\Scriptsctivate
```

You must activate this environment every time before running the app.

---

## 3. Install dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Setting up environment variables (.env)

This project requires an OpenAI API key.  
To avoid exposing sensitive information, environment variables are stored in a local `.env` file, which is ignored by `.gitignore`.

### Step 1: Create a `.env` file in the project root directory
```
touch .env
```

### Step 2: Add the following to `.env`
```
OPENAI_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual OpenAI API key.

### Step 3: Ensure your Python code loads `.env`
```
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

---

## 5. Start the backend server

```
uvicorn app.main:app --reload
```

You should see something like:
```
Uvicorn running on http://127.0.0.1:8000
```

If you want to specify a port:
```
uvicorn app.main:app --reload --port 9000
```

---

## 6. Open the web app

Visit:

http://127.0.0.1:8000  
or  
http://localhost:8000

---

## 7. Stop the server

Press `CTRL + C`

---

## 8. Important note: Do NOT push `.env` to GitHub

Your `.env` file contains private keys and must not be committed.  
The repository should already ignore it via `.gitignore`.

If you want to share required environment variables with others, create a `.env.example` file containing:

```
OPENAI_API_KEY=your_api_key_here
```

