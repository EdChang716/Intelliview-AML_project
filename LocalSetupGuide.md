# Intelliview Coach – Local Setup Guide

This guide explains how to install dependencies, set up environment variables, download the fine‑tuned embedding model, and launch the Intelliview Coach web application locally.

---

## 1. Clone the repository

```
git clone git@github.com:<your-username>/<your-repo>.git
cd <your-repo>
```

Replace `<your-username>` and `<your-repo>` with your actual GitHub SSH path.

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
intelliview-env\Scripts\activate
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
Environment variables are stored in a local `.env` file, which is ignored by `.gitignore`.

### Step 1: Create `.env`
```
touch .env
```

### Step 2: Add:
```
OPENAI_API_KEY=your_api_key_here
```

### Step 3: Ensure the backend loads `.env`
```
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

---

## 5. Download the fine‑tuned embedding model (required before running)

The RAG pipeline depends on a fine‑tuned SentenceTransformer located at:

```
models/jdq_bullet_finetuned/
```

This directory is ignored by GitHub because the model exceeds file size limits.

### Step 1: Download the model from Google Drive

https://drive.google.com/drive/folders/1eJNOiU2VUq8HNC9VqckAJ3VPeIQDHU0b

### Step 2: Place the model folder inside:

```
models/jdq_bullet_finetuned/
```

Final structure:

```
models/
    jdq_bullet_finetuned/
        config.json
        model.safetensors
        tokenizer.json
        sentence_transformer_config.json
```

Without this folder, **RAG will not work**.

---

## 6. Start the backend server

```
uvicorn app.main:app --reload
```

You should see:

```
Uvicorn running on http://127.0.0.1:8000
```

Specify a port if needed:

```
uvicorn app.main:app --reload --port 9000
```

---

## 7. Open the web app

Visit:

http://127.0.0.1:8000  
or  
http://localhost:8000

---

## 8. Stop the server

Press `CTRL + C`

---

## 9. Important: Do NOT push `.env` or large models to GitHub

These files are private and/or too large.

Create a `.env.example` for others:

```
OPENAI_API_KEY=your_api_key_here
```

