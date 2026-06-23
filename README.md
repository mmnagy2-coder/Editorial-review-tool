# Editorial Review Tool

A Streamlit proof-of-concept that analyzes short film YouTube clips for pacing, scene structure, and narrative consistency.

## Local setup

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Install `ffmpeg`:

```bash
brew install ffmpeg
```
```

3. Run the app:

```bash
streamlit run app.py
```

## Streamlit deployment

1. Create `.streamlit/secrets.toml` with your OpenAI key:

```toml
OPENAI_API_KEY = "your-openai-api-key"
```

2. Commit everything except `secrets.toml`.
3. Deploy to Streamlit Cloud or another Python host.

## Notes

- The app downloads a Whisper model on first run.
- It needs `ffmpeg` and `yt-dlp` to fetch and process YouTube video clips.
- If a YouTube URL is blocked, try a different public clip.
