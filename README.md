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

## Running tests

If you want to verify the helper functions and URL parsing, run:

```bash
python3 -m unittest test_app.py
```

## Streamlit deployment

1. Create `.streamlit/secrets.toml` with your Anthropic (Claude) key:

```toml
ANTHROPIC_API_KEY = "your-anthropic-api-key"
```

2. Add the system dependencies needed for OpenCV and ffmpeg.
3. Commit everything except `secrets.toml`.
4. Deploy to Streamlit Cloud or another Python host.

## Streamlit Cloud system packages

Create a `packages.txt` file in the repo root containing:

```text
ffmpeg
libglib2.0-0
libsm6
libxrender1
libxext6
libgl1-mesa-glx
```

This ensures OpenCV and ffmpeg work correctly in the Cloud environment.

## Notes

- The app downloads a Whisper model on first run.
- It needs `ffmpeg` and `yt-dlp` to fetch and process YouTube video clips.
- If a YouTube URL is blocked, try a different public clip.
