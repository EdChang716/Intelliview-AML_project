# Screenshots Directory

This directory contains screenshots for the main README.md file.

## Required Screenshots

### Demo Showcase (Website Screenshots)

1. **home-page.png** - Landing page showcasing the AI-powered interview coaching experience
2. **resume-setup.png** - Upload and parse your resume into structured sections
3. **profiles-page.png** - Manage multiple job profiles with different resumes and job descriptions
4. **practice-session.png** - Interactive practice with AI-generated questions tailored to your profile
5. **mock-interview.png** - Full mock interview experience with video recording
6. **report_1.png** - Mock interview report (first part)
7. **report_2.png** - Mock interview report (second part with scores and feedback)

### Technical Deep Dive (t-SNE Visualizations)

8. **tsne-mpnet-base.png** - t-SNE projection for all-mpnet-base-v2 (baseline)
9. **tsne-minilm.png** - t-SNE projection for all-MiniLM-L6-v2 (baseline)
10. **tsne-finetuned-mpnet.png** - t-SNE projection for fine-tuned MPNet model
11. **tsne-finetuned-minilm.png** - t-SNE projection for fine-tuned MiniLM model

## Screenshot Guidelines

- **Format**: PNG or JPG
- **Recommended width**: 1200-1600px for website screenshots, 400-800px for t-SNE plots
- **Aspect ratio**: Maintain original aspect ratio
- **Quality**: High-quality, clear images
- **Content**: Remove any personal or sensitive information before uploading

## How to Capture Screenshots

### Website Screenshots
1. Run the application locally: `uvicorn app.main:app --reload`
2. Navigate to each page mentioned above
3. Take full-page screenshots (you can use browser extensions like "Full Page Screen Capture")
4. Save them with the exact filenames listed above

### t-SNE Visualizations
1. Generate t-SNE plots from your embedding training/evaluation scripts
2. Export as PNG with clear labels and legends
3. Ensure the plots clearly show the clustering differences between baseline and fine-tuned models

Once all screenshots are added, they will automatically appear in the main README.md file.
