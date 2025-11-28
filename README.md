# Intelliview-AML_project for DS/SWE/ML/AI Engineer job seeker

An AI-powered interview coaching system integrating **LLM-based question generation**, **resume–JD retrieval**, **multimodal real-time feedback**, and a **FastAPI web backend**.  
This project includes modules for video/audio transcription, RAG-based retrieval with fine-tuning sentence Transformers, structured interview session flow, and a front-end built with HTML/CSS templates.

## Project Structure (Overview)

- **app/**
  - `main.py` — FastAPI backend entrypoint  
  - `static/` — All CSS UI styles  
  - `templates/` — HTML templates for the web app  

- **core/**
  - Embeddings, RAG pipeline, interview logic, transcription, video features, retrievers, and session management  

- **data/**
  - CSV resources for finetuning embedding model(JD–bullet dataset, generated questions, job titles, etc.)  

- **models/**
  - (Optional) fine-tuned embedding model stored locally  

- **notebooks/**
  - Jupyter notebooks for preprocessing and experimentation on fine-tuning

- **parsers/**
  - PDF resume parsing utilities  

- **scripts/**
  - Helper scripts (optional)  

- **user_data/**
  - Stored embeddings, user sessions, parsed resumes, and raw uploads  
  - (Usually ignored by git if configured properly)

This repository powers the full version of *Intelliview Coach*, supporting **resume parsing**, **RAG to retrieve most relevant experience**, **audio/video-based mock interviews**, **LLM-driven question flow**, and **automated behavioral + technical evaluation**.

# Retrieval-Augmented Generation (RAG) & Fine-Tuned Embedding Retriever

To retrieve user-specific experience (resume bullets) that can be used to construct strong interview answers, Intelliview uses a custom fine-tuned embedding retriever inside its RAG pipeline.

### Objective

Off-the-shelf embedding models (e.g., `all-mpnet-base-v2`) are not optimized for the task:

> (job description + interview question) → the correct resume bullet

We therefore fine-tuned a SentenceTransformer encoder with a contrastive learning objective so that:

- Queries: JD + interview_question
- Positives: the correct resume bullet
- Negatives: other bullets in the same batch (MultipleNegativesRankingLoss)

The goal is to pull the correct bullet closer to the query and push unrelated bullets further away, improving retrieval quality for RAG and LLM coaching.

### Dataset Construction

We combined:

- Kaggle job title + job description dataset  (https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description. ) 
- Additional Data Scientist job descriptions (https://www.kaggle.com/datasets/andrewmvd/data-scientist-jobs)

For each job description we used LLMs / templates to generate:

- Bullet points: key responsibilities / skills  
- Interview questions: 1–2 questions per bullet  

We then created aligned pairs:

- query_text = JD + interview_question
- bullet = the correct bullet for this question

This yielded ~1000 positive (query, bullet) pairs, which were split into train/test (e.g., 5000:1200 pairs after augmentation).

### Fine-Tuning Setup

- Base model: all-mpnet-base-v2 (SentenceTransformers)
- Loss: MultipleNegativesRankingLoss
- Batch size: 16
- Epochs: 5
- Device: CPU (sufficient for this scale)

Training loop (simplified):

train_examples = [
    InputExample(texts=[row["query_text"], row["bullet"]])
]

train_dataloader = DataLoader(train_examples, batch_size=16, shuffle=True)
loss = MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, loss)],
    epochs=5,
)

We tracked loss per epoch to ensure stable convergence.

### Retrieval Evaluation

We evaluate retrieval as:

> Given (JD + question), retrieve the correct bullet from the full bullet pool.

Metrics:

- Hit@k – whether the true bullet appears in the top-k  
- Precision@k – 1/k if the true bullet is in the top-k, else 0  

Comparison between the base model and the fine-tuned retriever:

k | Baseline Hit@k | Fine-tuned Hit@k
--|----------------|-----------------
1 | 0.21           | 0.37
3 | 0.31           | 0.54
5 | 0.37           | 0.61
10| 0.43           | 0.69

Key improvements:

- Hit@1: 21% → 37%  
- Hit@5: 37% → 61%  
- Hit@10: 43% → 69%  

### Embedding Space Visualization

We additionally visualized the embedding space via 2D t-SNE:

- Blue ×: queries (JD + interview_question)
- Orange ●: their corresponding bullets

Before fine-tuning (base MPNet): query–bullet pairs are scattered with long connecting lines → queries are often far from the correct bullets.

After fine-tuning: pairs are much closer and more clustered → the model better aligns job-specific questions with job-relevant experience.

### Model Saving & Use in the App

The final retriever is saved and loaded by the RAG pipeline:

- Fine-tuned model: models/jdq_bullet_finetuned/
- Precomputed bullet embeddings: models/jdq_bullet_finetuned/bullet_embs.npy

At runtime, the app:

1. Encodes (JD + current interview question) into a query embedding  
2. Retrieves the top-k bullets using cosine similarity  
3. Feeds those bullets as grounding context to the LLM for:
   - Sample answer generation  
   - Behavior/technical feedback based on user experience  

This fine-tuned retriever is a core component of Intelliview’s personalized RAG, enabling the system to reliably select relevant user experience for each question and significantly improving the realism and personalization of interview coaching.
