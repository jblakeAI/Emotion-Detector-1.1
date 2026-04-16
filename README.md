---
title: Emotion Detector
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.12.0"
python_version: "3.11.9"
app_file: app.py
pinned: false
---

# Emotion Detector

Emotion Detector is a web app that reads a piece of text and tells you what emotion it conveys — and how strongly. Paste in a sentence, a message, a review, or any snippet of prose and the app returns a breakdown of five core emotions with confidence scores, plus a clear dominant-emotion label.

## What it detects

The app classifies text across five emotions:

| Emotion | Example trigger text |
|---------|----------------------|
| **Joy** | "I'm so excited, this is the best day ever!" |
| **Sadness** | "I can't stop thinking about what I lost." |
| **Anger** | "This is completely unacceptable and unfair." |
| **Fear** | "I don't know what's going to happen and it terrifies me." |
| **Disgust** | "That behaviour is revolting and wrong." |

Under the hood it uses [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base), a fine-tuned DistilRoBERTa model, via the Hugging Face Inference API.

## How to use it

1. Type or paste any English text into the input box.
2. Click **Submit**.
3. Read the result — you get:
   - A **dominant emotion** label (the emotion with the highest confidence).
   - A **score chart** showing the confidence for all five emotions, so you can see whether the signal is clear-cut or mixed.

The interface is intentionally minimal so it can be embedded, extended, or used as a starting point for richer emotion-aware applications.

## Recommended architecture

- Code hosted in a GitHub repository.
- UI + runtime hosted on Hugging Face Spaces (Gradio SDK) — free tier, always on.
- `app.py` is the Space entrypoint; `emotion_detection.py` contains the inference logic.

## Local run

Create a Hugging Face access token at [Settings → Access Tokens](https://huggingface.co/settings/tokens).

**Important:** A token with only "read" Hub access is **not** enough. You need permission to **call Inference** (sometimes labeled **Inference Providers** on fine-grained tokens). Without it the API returns 403 and predictions will fail silently.

Supply the token in one of two ways:

- Add it to a `.env` file (auto-loaded): `HF_TOKEN=hf_...`
- Or export it in your shell: `export HF_TOKEN="hf_..."` (PowerShell: `$env:HF_TOKEN = "hf_..."`)

Then install dependencies and launch:

```bash
pip install -r requirements.txt
python app.py
```

## Deploy to Hugging Face Spaces

1. Push this project to GitHub.
2. On Hugging Face, create a new **Space** with SDK set to **Gradio** and visibility set to **Public** (free tier).
3. Connect the Space to your GitHub repo, or upload the files directly.
4. Confirm these files are present in the Space:
   - `app.py`
   - `emotion_detection.py`
   - `requirements.txt`
5. Add `HF_TOKEN` as a **Space secret** (Settings → Variables and secrets). Hugging Face will install dependencies and start the app automatically.

## Notes

- Inference goes through Hugging Face's current `InferenceClient` router. The legacy `api-inference.huggingface.co` endpoints are deprecated and will not work.
- If the Inference API is temporarily unavailable or rate-limited, the app returns a graceful error message rather than crashing.
- The model is English-only; other languages may produce unreliable scores.
