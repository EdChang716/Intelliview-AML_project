# Intelliview Coach – Backend Documentation(current version is incorrect, please ignore)

This document describes the backend architecture of Intelliview Coach, including directory structure, module responsibilities, and a list-based explanation of functions inside each Python file.

## Project Structure (Backend)
core/
│── config.py
│── retrieval.py
│── llm_client.py
│── sessions.py
│── profiles.py
│── embeddings.py
│── scoring.py
│
parsers/
│── resume_parser.py
│
main.py
.env
user_data/
models/
scripts(can be ignored)/
notebooks(for test)/

---

## core/config.py

Responsibilities:
- Load environment variables from .env
- Define project directories (BASE_DIR, USER_DATA_DIR, MODEL_DIR)
- Initialize the shared OpenAI client

Key components:
- load_dotenv()
- BASE_DIR, USER_DATA_DIR, MODEL_DIR
- client = OpenAI(...)

---

## core/retrieval.py

Responsibilities:
- Load the retriever embedding model
- Load resume entries and embeddings
- Perform similarity search to retrieve top-k relevant bullets

Key functions:
- get_retriever_model(): Loads SentenceTransformer model once
- load_resume_entries_and_embs(resume_id): Loads resume entries + embedding matrix
- retrieve_bullets_for_profile(profile_or_id, question, top_k):
    - Accepts profile_id or profile dict
    - Encodes question
    - Encodes truncated JD and mixes embeddings (0.8 / 0.2)
    - Computes cosine similarity
    - Returns top-k matching bullets

---

## core/llm.py

Responsibilities:
- Generate interview questions
- Generate follow-up questions
- Generate sample answers
- Behavioral question bank

Key functions:
- call_llm_for_question(jd_text, mode, avoid)
- call_llm_for_followup_question(...)
- generate_followup_question(...)
- call_llm_for_answer(question, jd_text, bullets)
- call_llm_for_sample_answer(...)

---

## core/sessions.py

Responsibilities:
- Manage practice session logs
- Store and retrieve user practice data
- Track question history to avoid repeats

Key functions:
- load_session(profile_id)
- save_session(profile_id, data)
- log_practice_turn(...)
- get_asked_questions(profile_id)
- get_practice_stats(profile_id)

---

## core/profiles.py

Responsibilities:
- Manage job profiles CRUD
- Load and save job profile data

Key functions:
- load_job_profiles()
- save_job_profiles()
- get_profile(profile_id)

---

## core/scoring.py

Responsibilities:
- Evaluate user answers using GPT
- Produce score, strengths, and improvements

Key function:
- evaluate_answer(question, jd_text, bullets, user_answer)

---

## core/embeddings.py

Responsibilities:
- Build embeddings for resume bullets

Key function:
- build_resume_embeddings(resume_id)

---

## parsers/resume_parser.py

Responsibilities:
- Extract text from uploaded PDF resumes
- Parse experience bullets and metadata (skills, education)

Key functions:
- extract_pdf_text(pdf_path)
- parse_resume_entries(raw_text)
- extract_metadata_sections(raw_text)
- extract_structured_education(raw_text)

---

## main.py (FastAPI Application)

Responsibilities:
- Expose API endpoints used by the frontend
- Route requests to the appropriate backend modules

API groups:
- Resume upload/edit
- Job profile CRUD
- Question generation
- Follow-up question generation
- Bullet retrieval
- Sample answer generation
- User answer scoring
- Practice history and statistics
- Coach chat interface