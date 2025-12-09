# Intelliview Coach

**An AI-Powered Interview Coaching System**

Intelliview Coach helps job seekers (DS/SWE/ML/AI Engineers) prepare for interviews by providing real-time, personalized feedback. It integrates **LLM-based question generation**, **RAG-based resume retrieval**, and **multimodal analysis** into a seamless interactive experience.

## 🚀 Key Features

- **📄 Resume Parsing**: Automatically extracts skills, experience, and education from PDF resumes.
- **🔍 Smart RAG Retrieval**: Uses a fine-tuned embedding model to retrieve the most relevant resume experiences for specific interview questions.
- **🤖 AI Interviewer**: Generates tailored technical and behavioral questions based on the Job Description (JD) and your profile.
- **🎙️ Multimodal Feedback**: Captures audio/video answers and provides detailed feedback on content, clarity, and relevance.
- **📊 Scoring & Insights**: Evaluates answers against the JD requirements and offers actionable improvement tips.

---

## ⚡ Getting Started

For installation instructions and local setup, please see the **[Quick Start Guide](QUICK_START.md)**.
> **Note**: This project is optimized for **Python 3.11**. Please ensure you have this version installed.

> **Note**: This project requires downloading a fine-tuned model artifact (approx. 500MB) which is not hosted in this repo. See the Quick Start Guide for the download link.

---

## 📂 Project Structure

```
.
├── app/                  # FastAPI Web Application
│   ├── main.py           # Application Entrypoint
│   ├── templates/        # HTML Frontend
│   └── static/           # CSS & Assets
├── core/                 # Core Logic
│   ├── retrieval.py      # RAG & Embedding Logic
│   ├── llm_client.py     # OpenAI Integration
│   ├── sessions.py       # Session Management
│   └── text_analysis.py  # Feedback Logic
├── data/                 # Training Data & Resources
├── models/               # Fine-tuned Models (Local)
└── parsers/              # Resume Parsing Utilities
```

---

## 🧠 Technical Deep Dive: The RAG Pipeline

To ensure the AI coach gives relevant advice, we don't just dump the whole resume into the prompt. We use a **custom fine-tuned retriever** to find the specific bullet points from your past experience that best answer the current interview question.

### The Problem with Generic Embeddings
Off-the-shelf models (like `all-mpnet-base-v2`) are great at general semantic similarity but often fail to map:
> **Query**: "Tell me about a time you optimized a slow database query." (Interview Question)
> **Target**: "Reduced API latency by 40% by implementing Redis caching and indexing SQL tables." (Resume Bullet)

Generic models might not see the strong connection between "slow database" and "Redis caching" in an interview context.

### Our Solution: Contrastive Fine-Tuning
We fine-tuned a `SentenceTransformer` model using **Contrastive Learning (MultipleNegativesRankingLoss)**.

1.  **Dataset**: ~5,000 pairs of (Job Description + Interview Question) ↔ (Correct Resume Bullet).
2.  **Training**: We forced the model to pull the "Correct Bullet" closer to the "Question" in vector space, while pushing unrelated bullets away.
3.  **Result**:
    - **Hit@10** improved from **43%** (Baseline) to **69%** (Fine-tuned).
    - This means the AI is much more likely to "remember" the right part of your resume when coaching you.

### Embedding Space Visualization
*Before vs. After Fine-tuning (t-SNE)*
- **Before**: Questions and relevant bullets were scattered.
- **After**: Questions cluster tightly with their relevant experience bullets.
