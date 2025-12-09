# Quick Start Guide: Intelliview Coach

This guide will get you up and running with the Intelliview Coach application locally.

## Prerequisites

- **Python 3.11 (Recommended)**
  - **Important**: `mediapipe` is NOT compatible with Python 3.13 or 3.14 yet.
  - While Python 3.9 works, we strongly recommend **Python 3.11** for best compatibility and syntax support.
  - If you are on macOS: `brew install python@3.11`.
- **Sox** (Required for audio processing)
  - macOS: `brew install sox`
  - Linux: `sudo apt-get install sox`
- **OpenAI API Key**: You need a valid OpenAI API key for LLM functionality.

## 1. Installation

### Clone the Repository
```bash
git clone git@github.com:<your-username>/Intelliview-AML_project.git
cd Intelliview-AML_project
```

### Set Up Virtual Environment

**macOS / Linux:**
```bash
# Install Python 3.11
brew install python@3.11

# Create venv using Python 3.11
/opt/homebrew/bin/python3.11 -m venv venv

# Activate it
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Configuration

Create a `.env` file in the root directory:

```bash
touch .env
```

Add your OpenAI API key to `.env`:
```
OPENAI_API_KEY=sk-your-api-key-here
```

## 3. Download Model Artifacts (CRITICAL)

The application requires fine-tuned models for the Retrieval-Augmented Generation (RAG) system. These are too large for Git and must be downloaded manually.

1.  **Download** the model folder from [Google Drive](https://drive.google.com/drive/folders/1eJNOiU2VUq8HNC9VqckAJ3VPeIQDHU0b).
2.  **Extract/Place** the downloaded contents so the structure looks exactly like this:

```
Intelliview-AML_project/
└── models/
    └── jdq_bullet_finetuned/
        ├── config.json
        ├── model.safetensors
        ├── tokenizer.json
        └── sentence_transformer_config.json
```

> **Note:** Without this folder structure, the application will fail to start or crash during RAG operations.

## 4. Run the Application

Start the FastAPI backend server:

```bash
uvicorn app.main:app --reload
```

The server will start at `http://127.0.0.1:8000`.

## 5. Usage

1.  Open your browser and navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000).
2.  **Upload a Resume**: Start by uploading a PDF resume.
3.  **Create a Job Profile**: Enter a job description you want to practice for.
4.  **Start Practice**: The system will generate interview questions based on your resume and the job description.
5.  **Feedback**: Answer questions via audio/video, and receive detailed feedback.
