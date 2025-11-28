# user_data Folder

This folder stores all dynamic, user-generated data created during the usage of Intelliview Coach.  
Because these files are specific to each user and may contain sensitive information, the entire directory should be **excluded from Git** and kept local only.

---

## What this folder contains

### 1. raw/
Stores the original uploaded resume files before parsing.

### 2. parsed/
Contains structured JSON outputs created by the resume parser.

### 3. embeddings/
Stores precomputed embedding vectors for each user's resume content.

### 4. sessions/
Stores saved mock interview sessions (transcripts, answers, feedback).

### 5. sessions/
Stores saved mock interview media sessions (audio or video).

### 6. mock/
Stores temporary audio/video or transcription artifacts during mock interviews.
There will be 3 files under this file: media/, results/ and sessions/

---

## Important Notes
- This entire folder is ignored by .gitignore.
- Do not upload these files to GitHub.
- Folder will be auto-created at runtime if missing.