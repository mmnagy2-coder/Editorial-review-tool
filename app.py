"""
AI Editorial Review – Proof of Concept
========================================
A single-file Streamlit app that takes a YouTube link to a short film,
analyses pacing, scene structure and narrative continuity using a mix of
local ML (Whisper, PySceneDetect, OpenCV) and a single GPT-4o-mini call,
then renders an interactive dark-themed dashboard.

Run with:  streamlit run app.py
"""

import os
import re
import json
import hashlib
import shutil
import tempfile
from datetime import datetime
from typing import Optional

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import cv2
import whisper
import yt_dlp
from scenedetect import detect, ContentDetector
from fuzzywuzzy import fuzz

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# --------------------------------------------------------------------------
# Constants & configuration
# --------------------------------------------------------------------------
MAX_DURATION_SECONDS = 295          # clip every download to ~< 5 minutes
WHISPER_MODEL_NAME = "base"         # smallest model that still gives usable text
MOTION_SAMPLE_FPS = 2               # frames per second sampled for motion energy
RUSH_SHOT_COUNT = 10                # > this many shots in RUSH_WINDOW = "rush zone"
RUSH_WINDOW_SECONDS = 5
PACING_DIP_MIN_DURATION = 15        # seconds of near-identical shot length = "dip"
PACING_DIP_VARIANCE_THRESHOLD = 0.6 # std-dev (s) below this counts as "low variance"
SILENCE_GAP_FOR_SCENE = 2.0         # seconds of transcript silence => possible scene cut
DEMO_URL = "https://www.youtube.com/watch?v=7d4ZgJFbF0A"  # Chaplin's "The Kid"

# Pre-written micro-lessons shown when "Teaching Mode" is enabled.
# These are static editorial-craft explanations, not AI generated.
TEACHING_NOTES = {
    "pacing_dip": (
        "**Why this matters:** Long unbroken shots ask a lot of an audience's "
        "attention. Without a cut, a reframe, or a change in performance, the eye "
        "has nothing new to track and engagement drifts. This doesn't mean every "
        "shot needs to be short — slow scenes can be powerful — but a static shot "
        "that runs long *without intentional staging* often reads as a pacing dip "
        "rather than a deliberate choice.\n\n"
        "**Try this:** Look for a natural insert point — a reaction shot, a cutaway "
        "to an object the character is using, or a push-in — roughly every "
        "8–12 seconds during dialogue-light passages. If the shot is meant to "
        "breathe, consider trimming 5–8 seconds of dead air at the head or tail."
    ),
    "rush_cut": (
        "**Why this matters:** A burst of very short shots (rapid cutting) is a "
        "classic device for tension, action or comedic punchlines — but if it "
        "happens where the story doesn't call for urgency, it can feel chaotic or "
        "disorient the viewer instead of exciting them.\n\n"
        "**Try this:** Watch the sequence with sound off. If you can't follow the "
        "geography of the scene, the cutting rate is probably outpacing the "
        "story information. Consider lengthening 2–3 of the shots to give the eye "
        "an anchor before the next rapid burst."
    ),
    "scene_structure": (
        "**Why this matters:** Scenes are the basic unit of dramatic structure — "
        "each one should change something (a goal, a piece of information, a "
        "relationship). Mapping scene boundaries by combining silence (dialogue "
        "gaps) and visual change (shot boundaries) approximates how an editor "
        "reads structure on a first pass: where does the story visibly *move on*?\n\n"
        "**Try this:** For each detected scene, write a one-line answer to "
        "'what changed by the end of this scene?' If you can't answer it, the "
        "scene may be redundant or need a clearer turn."
    ),
    "narrative_continuity": (
        "**Why this matters:** Audiences track a protagonist's knowledge and "
        "motivation closely, even subconsciously. If a character acts on "
        "information they shouldn't have yet, or abruptly drops a goal they were "
        "pursuing, it breaks the implicit contract that the story is being told "
        "fairly — and viewers feel it as confusion rather than surprise.\n\n"
        "**Try this:** Build a simple 'knowledge timeline' for your protagonist — "
        "list what they know and want at each scene boundary. Continuity errors "
        "usually show up as gaps in that timeline, not in the dialogue itself."
    ),
    "script_mismatch": (
        "**Why this matters:** When the cut diverges from the script — scenes "
        "missing, or extra material not in the script — it's not automatically "
        "wrong, but it's worth a deliberate check. Cut scenes might be improving "
        "the pace; added scenes might be scope creep that dilutes the story.\n\n"
        "**Try this:** For each mismatch, ask whether the change serves the "
        "throughline established in the script's opening scene. If it doesn't, "
        "it's a candidate for revision."
    ),
}

# --------------------------------------------------------------------------
# Page config + dark theme / custom CSS
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Editorial Review – Proof of Concept",
    page_icon="🎬",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    .stApp { background-color: #0E1117; color: #E6E6E6; }
    h1, h2, h3 { color: #F5F5F5; }
    .accent { color: #00C7B7; }

    .card {
        background-color: #161B22;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.45);
        border-left: 5px solid #30363D;
    }
    .card-red   { border-left-color: #FF5C5C; }
    .card-amber { border-left-color: #FFB84D; }
    .card-blue  { border-left-color: #4DA3FF; }
    .card-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 4px; }
    .card-time  { color: #00C7B7; font-family: monospace; font-size: 0.85rem; }
    .card-body  { margin-top: 8px; font-size: 0.92rem; line-height: 1.45; }
    .teaching   { margin-top: 10px; padding-top: 10px; border-top: 1px dashed #30363D;
                  color: #C9D1D9; font-size: 0.88rem; }

    .stButton>button {
        background-color: #00C7B7; color: #0E1117; border: none;
        border-radius: 8px; font-weight: 600;
    }
    .stButton>button:hover { background-color: #00A99B; color: white; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def fmt_time(seconds: float) -> str:
    """Format seconds as mm:ss for display."""
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else ""


def url_hash(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:12]


def get_openai_client():
    api_key = None
    if hasattr(st, "secrets"):
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            api_key = None
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


# --------------------------------------------------------------------------
# Cached pipeline steps
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def download_video(url: str) -> str:
    """Download a 360p clip (first MAX_DURATION_SECONDS) of the given YouTube URL.
    Returns the local file path. Cached by URL so re-runs are instant."""
    out_dir = os.path.join(tempfile.gettempdir(), "editorial_review_poc")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{url_hash(url)}.mp4")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    ffmpeg_path = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("Could not download video: ffmpeg binary not found. Install ffmpeg or set FFMPEG_BINARY.")

    ydl_opts = {
        "format": "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360]",
        "outtmpl": out_path,
        "merge_output_format": "mp4",
        "download_ranges": yt_dlp.utils.download_range_func(None, [(0, MAX_DURATION_SECONDS)]),
        "force_keyframes_at_cuts": True,
        "ffmpeg_location": ffmpeg_path,
        "quiet": True,
        "no_warnings": True,
    }

    def _download(opts):
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

    try:
        _download(ydl_opts)
    except Exception as exc:
        original_error = str(exc)
        fallback_attempts = [
            {
                "format": "best[height<=360]",
                "outtmpl": out_path,
                "merge_output_format": "mp4",
                "ffmpeg_location": ffmpeg_path,
                "quiet": True,
                "no_warnings": True,
            },
            {
                "format": "best[height<=360][ext=mp4]/best[height<=360]",
                "outtmpl": out_path,
                "merge_output_format": "mp4",
                "ffmpeg_location": ffmpeg_path,
                "allow_unplayable_formats": True,
                "geo_bypass": True,
                "quiet": True,
                "no_warnings": True,
            },
        ]

        last_error = exc
        for fallback_opts in fallback_attempts:
            try:
                _download(fallback_opts)
                last_error = None
                break
            except Exception as exc2:
                last_error = exc2

        if last_error is not None:
            raise RuntimeError(
                "Could not download video: original error: %s; fallback error: %s" % (
                    original_error, last_error
                )
            ) from last_error

    if not os.path.exists(out_path):
        raise RuntimeError("Download finished but no output file was produced.")
    return out_path


@st.cache_resource(show_spinner=False)
def load_whisper_model():
    """Load (and on first run, download) the Whisper 'base' model. Cached as a
    resource since the model object itself is not a plain serialisable value."""
    return whisper.load_model(WHISPER_MODEL_NAME)


@st.cache_data(show_spinner=False)
def transcribe_audio(video_path: str) -> list:
    """Run Whisper transcription, returning a list of {start, end, text} segments."""
    model = load_whisper_model()
    result = model.transcribe(video_path, fp16=False)
    return [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in result.get("segments", [])
    ]


@st.cache_data(show_spinner=False)
def detect_shots(video_path: str) -> list:
    """Content-aware shot detection. Returns list of (start_seconds, end_seconds)."""
    scene_list = detect(video_path, ContentDetector())
    return [(s.get_seconds(), e.get_seconds()) for s, e in scene_list]


@st.cache_data(show_spinner=False)
def compute_motion_energy(video_path: str) -> list:
    """Sample frames at MOTION_SAMPLE_FPS and compute normalised frame-differencing
    motion energy at each sample point. Returns list of (time_seconds, energy_0_1)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(1, int(round(fps / MOTION_SAMPLE_FPS)))

    raw = []
    prev_gray = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray).mean()
                raw.append((frame_idx / fps, diff))
            prev_gray = gray
        frame_idx += 1
    cap.release()

    if not raw:
        return []
    max_diff = max(d for _, d in raw) or 1.0
    return [(t, d / max_diff) for t, d in raw]


# --------------------------------------------------------------------------
# Analysis logic (pure functions, not cached — cheap to recompute)
# --------------------------------------------------------------------------
def compute_pacing_issues(shots: list) -> list:
    """Detect 'pacing dips' (long, low-variance shot runs) and 'rush zones'
    (many shots packed into a short window)."""
    issues = []
    durations = [end - start for start, end in shots]

    # Rush zones: > RUSH_SHOT_COUNT shot starts within RUSH_WINDOW_SECONDS.
    starts = [s for s, _ in shots]
    i = 0
    while i < len(starts):
        window_end = starts[i] + RUSH_WINDOW_SECONDS
        j = i
        while j < len(starts) and starts[j] <= window_end:
            j += 1
        count = j - i
        if count > RUSH_SHOT_COUNT:
            issues.append({
                "type": "rush",
                "start": starts[i],
                "end": starts[j - 1] if j - 1 < len(starts) else starts[i] + RUSH_WINDOW_SECONDS,
                "message": f"{count} cuts in {RUSH_WINDOW_SECONDS}s – risk of disorienting rapid cutting.",
            })
            i = j
        else:
            i += 1

    # Pacing dips: a run of shots whose combined duration > PACING_DIP_MIN_DURATION
    # with low variance in shot length (i.e. nothing visually breaks the rhythm).
    run_start_idx = 0
    while run_start_idx < len(shots):
        run_end_idx = run_start_idx
        total = durations[run_start_idx]
        while run_end_idx + 1 < len(shots) and total < PACING_DIP_MIN_DURATION:
            run_end_idx += 1
            total += durations[run_end_idx]
        run = durations[run_start_idx:run_end_idx + 1]
        if total >= PACING_DIP_MIN_DURATION and (np.std(run) if len(run) > 1 else 0) < PACING_DIP_VARIANCE_THRESHOLD:
            issues.append({
                "type": "dip",
                "start": shots[run_start_idx][0],
                "end": shots[run_end_idx][1],
                "message": f"{int(total)}s without a meaningful change in cutting rhythm – risk of visual monotony.",
            })
            run_start_idx = run_end_idx + 1
        else:
            run_start_idx += 1

    return issues


def detect_scenes(transcript: list, shots: list, total_duration: float) -> list:
    """Heuristic scene detection: a scene boundary is a transcript silence gap
    > SILENCE_GAP_FOR_SCENE seconds that coincides with (or is near) a shot
    boundary. Falls back to shot boundaries alone if there's no transcript."""
    boundaries = {0.0}
    shot_starts = [s for s, _ in shots]

    for i in range(len(transcript) - 1):
        gap = transcript[i + 1]["start"] - transcript[i]["end"]
        if gap > SILENCE_GAP_FOR_SCENE:
            gap_mid = (transcript[i]["end"] + transcript[i + 1]["start"]) / 2
            nearest_shot = min(shot_starts, key=lambda s: abs(s - gap_mid), default=gap_mid)
            if abs(nearest_shot - gap_mid) < 3.0:
                boundaries.add(nearest_shot)
            else:
                boundaries.add(gap_mid)

    if len(boundaries) <= 1 and shot_starts:
        # No usable transcript signal — fall back to every 4th shot as a scene cut.
        boundaries.update(shot_starts[::4])

    boundaries.add(total_duration)
    ordered = sorted(boundaries)

    scenes = []
    for idx in range(len(ordered) - 1):
        start, end = ordered[idx], ordered[idx + 1]
        if end - start < 1.0:
            continue
        lines = [t["text"] for t in transcript if start <= t["start"] < end]
        scenes.append({
            "index": len(scenes) + 1,
            "start": start,
            "end": end,
            "duration": end - start,
            "dialogue": " ".join(lines).strip(),
        })
    return scenes


def compare_to_script(scenes: list, script_text: str) -> dict:
    """Fuzzy-match each detected scene's dialogue against lines in the uploaded
    script to flag scenes that appear to be missing from (or extra vs) the script."""
    script_lines = [l.strip() for l in script_text.splitlines() if l.strip()]
    if not script_lines:
        return {"missing": [], "extra": []}

    matched_script_lines = set()
    missing = []  # detected scenes with no good match in the script
    for scene in scenes:
        if not scene["dialogue"]:
            continue
        best_score, best_line = 0, None
        for line in script_lines:
            score = fuzz.partial_ratio(scene["dialogue"][:200].lower(), line.lower())
            if score > best_score:
                best_score, best_line = score, line
        if best_score >= 60:
            matched_script_lines.add(best_line)
        else:
            missing.append(scene)

    extra = [line for line in script_lines if line not in matched_script_lines]
    return {"missing": missing, "extra": extra[:10]}  # cap noise on 'extra'


@st.cache_data(show_spinner=False)
def check_narrative_consistency(transcript_key: str, early_text: str, late_text: str) -> str:
    """Single GPT-4o-mini call comparing early vs. late dialogue for continuity.
    `transcript_key` exists purely so the cache is keyed per-video."""
    client = get_openai_client()
    if client is None:
        return (
            "⚠️ No OpenAI API key found (set OPENAI_API_KEY in st.secrets or the "
            "environment), so the narrative consistency check was skipped."
        )

    prompt = (
        "You are a script editor checking continuity. Based on the following dialogue "
        "from early scenes:\n\n"
        f"{early_text}\n\n"
        "...and dialogue from later scenes:\n\n"
        f"{late_text}\n\n"
        "Does the main character's knowledge or behaviour stay consistent? Identify any "
        "contradictions or changes in their goals or information. Explain simply, in "
        "3-5 sentences."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"⚠️ Narrative consistency check failed: {exc}"


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------
def build_timeline_figure(shots, motion, pacing_issues, scene_boundaries, total_duration):
    fig = go.Figure()

    shot_x = [s for s, _ in shots]
    shot_y = [e - s for s, e in shots]
    fig.add_trace(go.Scatter(
        x=shot_x, y=shot_y, mode="lines+markers", name="Shot duration (s)",
        line=dict(color="#4DA3FF", width=2), marker=dict(size=5),
    ))

    if motion:
        m_x = [t for t, _ in motion]
        m_y_raw = [e for _, e in motion]
        # rescale motion (0-1) onto the same visual range as shot durations for overlay clarity
        scale = max(shot_y) if shot_y else 1.0
        m_y = [v * scale for v in m_y_raw]
        fig.add_trace(go.Scatter(
            x=m_x, y=m_y, mode="lines", name="Motion energy (scaled)",
            line=dict(color="#00C7B7", width=1.5, dash="dot"), opacity=0.7,
        ))

    for issue in pacing_issues:
        color = "rgba(255,184,77,0.18)" if issue["type"] == "dip" else "rgba(255,92,92,0.18)"
        fig.add_vrect(x0=issue["start"], x1=issue["end"], fillcolor=color, line_width=0,
                       annotation_text="Pacing Dip" if issue["type"] == "dip" else "Rush",
                       annotation_position="top left", annotation_font_size=10)

    for b in scene_boundaries:
        fig.add_vline(x=b, line_width=1, line_dash="dash", line_color="#8B949E")

    tick_vals = list(range(0, int(total_duration) + 1, max(10, int(total_duration // 10) or 10)))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        height=380, margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", y=1.08),
        xaxis=dict(title="Time", tickvals=tick_vals, ticktext=[fmt_time(v) for v in tick_vals]),
        yaxis=dict(title="Seconds / scaled energy"),
    )
    return fig


def youtube_embed(video_id: str, start_seconds: int = 0, height: int = 380):
    """Embed the YouTube player via the IFrame API. `start_seconds` lets issue
    cards 'jump' the player to a timecode — clicking a card sets session_state
    and Streamlit's rerun recreates this embed with the new start time."""
    html = f"""
    <div id="player"></div>
    <script src="https://www.youtube.com/iframe_api"></script>
    <script>
      var player;
      function onYouTubeIframeAPIReady() {{
        player = new YT.Player('player', {{
          height: '{height}', width: '100%', videoId: '{video_id}',
          playerVars: {{ start: {int(start_seconds)}, autoplay: 0, rel: 0 }}
        }});
      }}
    </script>
    """
    st.components.v1.html(html, height=height + 10)


# --------------------------------------------------------------------------
# Card rendering
# --------------------------------------------------------------------------
def render_card(color_class, title, time_label, body, teaching_key, teaching_mode, button_key):
    st.markdown(f"""
        <div class="card {color_class}">
            <div class="card-title">{title}</div>
            <div class="card-time">{time_label}</div>
            <div class="card-body">{body}</div>
        </div>
    """, unsafe_allow_html=True)
    if teaching_mode and teaching_key in TEACHING_NOTES:
        with st.expander("📚 Teaching note", expanded=True):
            st.markdown(TEACHING_NOTES[teaching_key])


# --------------------------------------------------------------------------
# Report export
# --------------------------------------------------------------------------
def generate_text_report(url, scenes, pacing_issues, narrative_result, script_diff) -> str:
    lines = [
        "AI EDITORIAL REVIEW – PROOF OF CONCEPT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Source video: {url}",
        "", "=== SCENE STRUCTURE ===",
    ]
    for sc in scenes:
        lines.append(f"Scene {sc['index']}: {fmt_time(sc['start'])}–{fmt_time(sc['end'])} "
                      f"({sc['duration']:.0f}s)")

    lines += ["", "=== PACING ISSUES ==="]
    if not pacing_issues:
        lines.append("No pacing dips or rush zones detected.")
    for issue in pacing_issues:
        kind = "Pacing Dip" if issue["type"] == "dip" else "Rush Zone"
        lines.append(f"[{kind}] {fmt_time(issue['start'])}-{fmt_time(issue['end'])}: {issue['message']}")

    lines += ["", "=== NARRATIVE CONSISTENCY ===", narrative_result]

    if script_diff is not None:
        lines += ["", "=== SCRIPT COMPARISON ==="]
        lines.append(f"Scenes with no clear match in script: {len(script_diff['missing'])}")
        for sc in script_diff["missing"]:
            lines.append(f"  - Scene {sc['index']} ({fmt_time(sc['start'])})")
        lines.append(f"Script lines with no matching detected scene: {len(script_diff['extra'])}")
        for l in script_diff["extra"]:
            lines.append(f"  - {l[:100]}")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main analysis pipeline (drives the spinner / progress steps)
# --------------------------------------------------------------------------
def run_analysis(url: str, script_text: Optional[str], progress_bar, status_text):
    steps = [
        "Downloading video...",
        "Transcribing audio...",
        "Detecting shots...",
        "Calculating pacing metrics...",
        "Checking narrative consistency...",
        "Generating report...",
    ]
    n = len(steps)

    status_text.text(steps[0]); progress_bar.progress(1 / n)
    video_path = download_video(url)

    status_text.text(steps[1]); progress_bar.progress(2 / n)
    transcript = transcribe_audio(video_path)

    status_text.text(steps[2]); progress_bar.progress(3 / n)
    shots = detect_shots(video_path)
    motion = compute_motion_energy(video_path)

    status_text.text(steps[3]); progress_bar.progress(4 / n)
    cap = cv2.VideoCapture(video_path)
    total_duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25.0))
    cap.release()
    pacing_issues = compute_pacing_issues(shots) if shots else []
    scenes = detect_scenes(transcript, shots, total_duration)

    status_text.text(steps[4]); progress_bar.progress(5 / n)
    full_dialogue = [t["text"] for t in transcript if t["text"]]
    early_text = " ".join(full_dialogue[:30]) or "(no dialogue detected)"
    late_text = " ".join(full_dialogue[-30:]) or "(no dialogue detected)"
    narrative_result = check_narrative_consistency(url_hash(url), early_text, late_text)

    script_diff = compare_to_script(scenes, script_text) if script_text else None

    status_text.text(steps[5]); progress_bar.progress(1.0)

    return {
        "video_path": video_path,
        "transcript": transcript,
        "shots": shots,
        "motion": motion,
        "total_duration": total_duration,
        "pacing_issues": pacing_issues,
        "scenes": scenes,
        "narrative_result": narrative_result,
        "script_diff": script_diff,
    }


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
def main():
    st.title("🎬 AI Editorial Review")
    st.markdown('<span class="accent">Proof of Concept</span> — pacing, scene structure '
                'and narrative-consistency analysis for short films.', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Settings")
        teaching_mode = st.checkbox("📚 Teaching Mode", value=False,
                                     help="Expand every issue card with an educational explanation.")
        st.divider()
        st.subheader("Input")
        url = st.text_input("YouTube URL (short film, under ~5 min)", value="")
        script_file = st.file_uploader("Optional script (.txt)", type=["txt"])
        analyse_clicked = st.button("Analyse", use_container_width=True)
        demo_clicked = st.button("Run Demo (Chaplin's \"The Kid\")", use_container_width=True)
        st.caption("First run downloads the Whisper 'base' model (~140MB) — this only happens once.")
        if get_openai_client() is None:
            st.warning("No OPENAI_API_KEY found — narrative consistency check will be skipped.")

    if "results" not in st.session_state:
        st.session_state.results = None
    if "seek_to" not in st.session_state:
        st.session_state.seek_to = 0

    target_url = None
    if demo_clicked:
        target_url = DEMO_URL
    elif analyse_clicked:
        target_url = url.strip() or DEMO_URL

    if target_url:
        video_id = extract_video_id(target_url)
        if not video_id:
            st.error("That doesn't look like a valid YouTube URL.")
            return

        script_text = None
        if script_file is not None:
            script_text = script_file.read().decode("utf-8", errors="ignore")

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        try:
            with st.spinner("Running editorial analysis..."):
                results = run_analysis(target_url, script_text, progress_bar, status_text)
            results["video_id"] = video_id
            results["url"] = target_url
            st.session_state.results = results
            st.session_state.seek_to = 0
            status_text.text("Done.")
        except RuntimeError as exc:
            st.error(f"Could not analyse this video: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the user
            st.error(f"Unexpected error during analysis: {exc}")
            return

    results = st.session_state.results
    if not results:
        st.info("Paste a YouTube link and click **Analyse**, or click **Run Demo** to try "
                "the tool with a public-domain example.")
        return

    # ---- Dashboard ----
    st.subheader("Source video")
    youtube_embed(results["video_id"], start_seconds=st.session_state.seek_to)

    st.subheader("Pacing & shot timeline")
    scene_boundaries = [sc["start"] for sc in results["scenes"]]
    fig = build_timeline_figure(
        results["shots"], results["motion"], results["pacing_issues"],
        scene_boundaries, results["total_duration"],
    )
    st.plotly_chart(fig, use_container_width=True)

    col_scenes, col_pacing = st.columns(2)

    with col_scenes:
        st.markdown("### 🟦 Scene structure")
        if not results["scenes"]:
            st.write("No scenes could be detected.")
        for sc in results["scenes"]:
            body = f"Duration: {sc['duration']:.0f}s"
            if sc["dialogue"]:
                snippet = sc["dialogue"][:140] + ("…" if len(sc["dialogue"]) > 140 else "")
                body += f"<br><i>{snippet}</i>"
            btn_key = f"jump_scene_{sc['index']}"
            render_card("card-blue", f"Scene {sc['index']}",
                        f"{fmt_time(sc['start'])} – {fmt_time(sc['end'])}",
                        body, "scene_structure", teaching_mode, btn_key)
            if st.button(f"⏩ Jump to Scene {sc['index']}", key=btn_key):
                st.session_state.seek_to = int(sc["start"])
                st.rerun()

        if results["script_diff"] is not None:
            diff = results["script_diff"]
            st.markdown("#### Script comparison")
            if not diff["missing"] and not diff["extra"]:
                st.success("Detected scenes line up with the uploaded script.")
            else:
                body = (f"{len(diff['missing'])} detected scene(s) had no clear match in the "
                        f"script, and {len(diff['extra'])} script line(s) had no matching scene "
                        f"in the cut.")
                render_card("card-blue", "Script vs. cut mismatch", "", body,
                            "script_mismatch", teaching_mode, "script_diff")

    with col_pacing:
        st.markdown("### 🟧 Pacing issues")
        if not results["pacing_issues"]:
            st.write("No pacing dips or rush zones detected — cutting rhythm looks healthy.")
        for idx, issue in enumerate(results["pacing_issues"]):
            kind = "Pacing Dip" if issue["type"] == "dip" else "Rapid Cutting / Rush"
            color = "card-amber" if issue["type"] == "dip" else "card-red"
            teaching_key = "pacing_dip" if issue["type"] == "dip" else "rush_cut"
            btn_key = f"jump_pacing_{idx}"
            render_card(color, kind, f"{fmt_time(issue['start'])} – {fmt_time(issue['end'])}",
                        issue["message"], teaching_key, teaching_mode, btn_key)
            if st.button(f"⏩ Jump to {fmt_time(issue['start'])}", key=btn_key):
                st.session_state.seek_to = int(issue["start"])
                st.rerun()

    st.markdown("### 🟥 Narrative consistency")
    render_card("card-red", "Main character knowledge / goal check", "",
                results["narrative_result"], "narrative_continuity", teaching_mode,
                "jump_narrative")

    st.divider()
    report_text = generate_text_report(
        results["url"], results["scenes"], results["pacing_issues"],
        results["narrative_result"], results["script_diff"],
    )
    st.download_button(
        "⬇️ Download report (.txt)", data=report_text,
        file_name=f"editorial_review_{results['video_id']}.txt", mime="text/plain",
    )


if __name__ == "__main__":
    main()
