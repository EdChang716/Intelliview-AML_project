# Download Fine-Tuned Embedding Model

This project requires a fine-tuned SentenceTransformer model for resume–JD retrieval.
Because the model is large, it is not included in this repository.

## Download Link

Download the model folder (jdq_bullet_finetuned) from Google Drive:

https://drive.google.com/drive/folders/1eJNOiU2VUq8HNC9VqckAJ3VPeIQDHU0b?usp=drive_link

## Installation

After downloading, place the model folder here:

models/jdq_bullet_finetuned/

Your directory structure should look like:

models/
  jdq_bullet_finetuned/
    config.json
    model.safetensors
    tokenizer.json
    sentencepiece.bpe.model
    ...

## Notes

- This folder is ignored by .gitignore to prevent large files from being versioned.
- If the directory is missing, the system will not be able to load the embedding model.
- Only the local environment requires this model; it will not be pushed to GitHub.