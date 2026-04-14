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

This app can be deployed for free on Hugging Face Spaces so it runs for free and is always on.

## Recommended architecture

- Code hosted in GitHub repository.
- UI + runtime hosted in Hugging Face Spaces (Gradio SDK).
- `app.py` is the Space entrypoint.

## Local run

Create a Hugging Face access token at [Settings → Access Tokens](https://huggingface.co/settings/tokens).

**Important:** A token that only has “read” Hub access is **not** enough. You need permission to **call Inference** (sometimes labeled **Inference Providers** or similar on fine‑grained tokens). If that permission is missing, the API returns **403** and every prediction will look “empty” in the app.

You can either:

- Put it in `.env` (auto-loaded): `HF_TOKEN=hf_...`
- Or set it in shell: PowerShell `$env:HF_TOKEN = "hf_..."` (or `HF_API_TOKEN`)

```bash
pip install -r requirements.txt
python app.py
```

## Deploy to Hugging Face Spaces

1. Push this project to GitHub.
2. In Hugging Face, create a new **Space** with:
   - SDK: **Gradio**
   - Visibility: Public (free)
3. Connect/sync the Space to your GitHub repo, or upload files directly.
4. Ensure these files are present:
   - `app.py`
   - `emotion_detection.py`
   - `requirements.txt`
5. Hugging Face will install dependencies and run automatically.

## Notes

- Inference goes through Hugging Face’s current router (`InferenceClient` in `emotion_detection.py`). The old `api-inference.huggingface.co` URLs are deprecated and will not work.
- Set **`HF_TOKEN`** or **`HF_API_TOKEN`** locally and as a **Space secret** on Hugging Face. Without a token or without Inference permission, predictions fail.
- If the API is temporarily unavailable or rate-limited, the app returns an empty/invalid response instead of crashing.
