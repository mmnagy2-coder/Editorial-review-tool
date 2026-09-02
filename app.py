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
import base64
import hashlib
import shutil
import tempfile
import xml.etree.ElementTree as ET
import xml.sax.saxutils as saxutils
from datetime import datetime
from typing import Optional

try:
    import pypdf
except ImportError:
    pypdf = None

import numpy as np
import streamlit as st
import plotly.graph_objects as go
try:
    import cv2
    CV2_IMPORT_ERROR = None
except Exception as _cv2_exc:
    cv2 = None
    CV2_IMPORT_ERROR = _cv2_exc
import whisper
import yt_dlp
from scenedetect import detect, ContentDetector
from fuzzywuzzy import fuzz

try:
    from anthropic import Client as AnthropicClient
except ImportError:
    AnthropicClient = None

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
DEMO_URL = "https://vimeo.com/135459618?share=copy&fl=cl&fe=ci"
DOWNLOAD_RETRY_ATTEMPTS = 3
DOWNLOAD_RETRY_DELAY = 2            # seconds between retries
MAX_VIDEO_SIZE_MB = 500             # max local video file size in MB

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

    .filmstrip-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 12px;
        text-align: center;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .filmstrip-card:hover {
        border-color: #00C7B7;
        transform: translateY(-2px);
    }
    .scale-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .badge-cu { background-color: rgba(77, 163, 255, 0.2); color: #4DA3FF; border: 1px solid #4DA3FF; }
    .badge-ms { background-color: rgba(0, 199, 183, 0.2); color: #00C7B7; border: 1px solid #00C7B7; }
    .badge-ws { background-color: rgba(255, 184, 77, 0.2); color: #FFB84D; border: 1px solid #FFB84D; }
    .badge-dip { background-color: rgba(255, 92, 92, 0.2); color: #FF5C5C; border: 1px solid #FF5C5C; }
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


from urllib.parse import urlparse


def extract_video_id_and_platform(url: str) -> tuple:
    """Extract video ID and platform from URL. Returns (video_id, platform) or ("", "")."""
    if not url or not url.strip():
        return ("", "")
    url = url.strip()
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""

    # YouTube patterns
    yt_match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", url)
    if yt_match:
        return (yt_match.group(1), "youtube")

    # Vimeo patterns
    vm_match = re.search(r"(?:vimeo\.com/(?:video/)?|player\.vimeo\.com/video/)([0-9]+)", url)
    if vm_match:
        return (vm_match.group(1), "vimeo")

    # Frame.io patterns
    if "frame.io" in hostname:
        segments = [seg for seg in path.split("/") if seg]
        if segments:
            return (segments[-1], "frameio")

    # Generic Web Video URL
    if url.startswith("http://") or url.startswith("https://"):
        clean_name = re.sub(r'[^A-Za-z0-9_-]', '_', os.path.basename(path) or url_hash(url))
        return (clean_name, "web")

    return ("", "")


def extract_video_id(url: str) -> str:
    """Extract video ID from URL (for backward compatibility)."""
    video_id, _ = extract_video_id_and_platform(url)
    return video_id


def url_hash(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:12]


def parse_fdx_content(xml_text: str) -> dict:
    """Parse Final Draft (.fdx) XML screenplay format into structured scenes and dialogue."""
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return parse_screenplay_text(xml_text)

    scenes = []
    current_scene = {"slugline": "PROLOGUE", "dialogue": [], "characters": []}
    dialogue_blocks = []
    current_character = None

    for p in root.iter("Paragraph"):
        p_type = p.get("Type", "")
        texts = [t.text for t in p.iter("Text") if t.text]
        line_text = "".join(texts).strip()
        if not line_text:
            continue

        if p_type == "Scene Heading":
            if current_scene["dialogue"] or current_scene["slugline"] != "PROLOGUE":
                scenes.append(current_scene)
            current_scene = {"slugline": line_text, "dialogue": [], "characters": []}
            current_character = None
        elif p_type == "Character":
            current_character = line_text.upper()
            if current_character not in current_scene["characters"]:
                current_scene["characters"].append(current_character)
        elif p_type == "Dialogue":
            spk = current_character or "UNKNOWN"
            current_scene["dialogue"].append(f"{spk}: {line_text}")
            dialogue_blocks.append({"character": spk, "text": line_text})
        elif p_type == "Action":
            current_character = None

    if current_scene["dialogue"] or current_scene["slugline"] != "PROLOGUE":
        scenes.append(current_scene)

    return {
        "format": "fdx",
        "raw_text": xml_text,
        "scenes": scenes,
        "dialogue_blocks": dialogue_blocks,
        "total_dialogue_lines": len(dialogue_blocks)
    }


def parse_screenplay_text(raw_text: str) -> dict:
    """Parse plain text / Fountain screenplay format into structured scenes and dialogue."""
    lines = raw_text.splitlines()
    scenes = []
    current_scene = {"slugline": "SCENE 1", "dialogue": [], "characters": []}
    dialogue_blocks = []
    current_character = None

    scene_heading_regex = re.compile(r"^(INT\.|EXT\.|INT/EXT\.|EXT/INT\.|SCENE\s+\d+)", re.IGNORECASE)
    character_regex = re.compile(r"^[A-Z0-9_\-'\s]{2,30}(\s*\(.*\))?$")

    for raw_l in lines:
        line = raw_l.strip()
        if not line:
            current_character = None
            continue

        if scene_heading_regex.match(line):
            if current_scene["dialogue"] or current_scene["slugline"] != "SCENE 1":
                scenes.append(current_scene)
            current_scene = {"slugline": line, "dialogue": [], "characters": []}
            current_character = None
        elif character_regex.match(line) and not line.startswith("INT") and not line.startswith("EXT"):
            clean_char = re.sub(r"\(.*\)", "", line).strip()
            if len(clean_char) > 1 and not clean_char.endswith("."):
                current_character = clean_char
                if current_character not in current_scene["characters"]:
                    current_scene["characters"].append(current_character)
        elif current_character:
            current_scene["dialogue"].append(f"{current_character}: {line}")
            dialogue_blocks.append({"character": current_character, "text": line})
        else:
            if len(line) > 10 and ":" in line:
                parts = line.split(":", 1)
                spk = parts[0].strip().upper()
                txt = parts[1].strip()
                current_scene["dialogue"].append(line)
                dialogue_blocks.append({"character": spk, "text": txt})

    if current_scene["dialogue"] or current_scene["slugline"] != "SCENE 1":
        scenes.append(current_scene)

    return {
        "format": "text_fountain",
        "raw_text": raw_text,
        "scenes": scenes,
        "dialogue_blocks": dialogue_blocks,
        "total_dialogue_lines": len(dialogue_blocks)
    }


def read_script_file(script_file):
    """Read and parse uploaded screenplay file (.pdf, .fdx, .txt, .fountain)."""
    if script_file is None:
        return None

    filename = getattr(script_file, "name", "").lower()

    if filename.endswith(".pdf") and pypdf is not None:
        try:
            reader = pypdf.PdfReader(script_file)
            pages_text = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    pages_text.append(extracted)
            raw_text = "\n".join(pages_text)
            return parse_screenplay_text(raw_text)
        except Exception:
            return None

    try:
        content = script_file.read().decode("utf-8", errors="ignore")
    except Exception:
        return None

    if filename.endswith(".fdx") or "<FinalDraft" in content:
        return parse_fdx_content(content)
    else:
        return parse_screenplay_text(content)


def read_script_text(script_file):
    """Backward-compatible helper returning raw script text."""
    parsed = read_script_file(script_file)
    return parsed.get("raw_text") if parsed else None


def analyze_script_coverage(detected_scenes: list, screenplay_data: dict) -> dict:
    """Deep fuzzy matching between screenplay dialogue lines and detected cut dialogue."""
    if not screenplay_data:
        return None

    script_dialogue = screenplay_data.get("dialogue_blocks", [])
    if not script_dialogue:
        return {
            "coverage_percent": 100.0,
            "matched_lines_count": 0,
            "omitted_lines_count": 0,
            "improvised_scenes_count": 0,
            "matched_lines": [],
            "omitted_lines": [],
            "unmatched_cut_scenes": []
        }

    all_cut_dialogue = " ".join([sc.get("dialogue", "") for sc in detected_scenes])
    matched_lines = []
    omitted_lines = []

    for item in script_dialogue:
        line_text = item.get("text", "")
        if len(line_text) < 5:
            continue
        match_score = fuzz.partial_ratio(line_text.lower(), all_cut_dialogue.lower())
        if match_score >= 65:
            matched_lines.append({**item, "score": match_score})
        else:
            omitted_lines.append(item)

    total_valid = max(1, len(matched_lines) + len(omitted_lines))
    coverage_pct = round((len(matched_lines) / total_valid) * 100, 1)

    unmatched_cut_scenes = []
    for sc in detected_scenes:
        if not sc.get("dialogue"):
            continue
        scene_txt = sc["dialogue"][:150].lower()
        best_sc_score = max([fuzz.partial_ratio(scene_txt, item["text"].lower()) for item in script_dialogue] or [0])
        if best_sc_score < 50:
            unmatched_cut_scenes.append({
                "scene_index": sc["index"],
                "start": sc["start"],
                "dialogue": sc["dialogue"][:120]
            })

    return {
        "coverage_percent": coverage_pct,
        "total_script_lines": total_valid,
        "matched_lines_count": len(matched_lines),
        "omitted_lines_count": len(omitted_lines),
        "improvised_scenes_count": len(unmatched_cut_scenes),
        "matched_lines": matched_lines[:20],
        "omitted_lines": omitted_lines[:20],
        "unmatched_cut_scenes": unmatched_cut_scenes[:10]
    }


def generate_dramatic_tension_curve(total_duration: float, shots: list, pacing_issues: list, scenes: list) -> go.Figure:
    """Generate a continuous 3-Act Dramatic Tension & Stakes Curve across the film."""
    time_points = np.linspace(0, total_duration, num=120)
    t_norm = time_points / max(1.0, total_duration)

    base_curve = (
        30.0
        + 25.0 * np.sin(t_norm * np.pi)
        + 35.0 * np.exp(-((t_norm - 0.85) ** 2) / 0.02)
        + 15.0 * np.exp(-((t_norm - 0.50) ** 2) / 0.03)
        + 10.0 * np.exp(-((t_norm - 0.20) ** 2) / 0.02)
    )

    cutting_intensity = np.zeros_like(time_points)
    durations = [s[1] - s[0] for s in shots] if shots else []
    asl = np.mean(durations) if durations else 7.0

    for i, t in enumerate(time_points):
        local_shots = [s for s in shots if abs(s[0] - t) < 30.0]
        if local_shots:
            local_asl = np.mean([s[1] - s[0] for s in local_shots])
            speed_factor = max(-15.0, min(15.0, (asl - local_asl) * 2.0))
            cutting_intensity[i] = speed_factor

    final_tension = np.clip(base_curve + cutting_intensity, 10.0, 100.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_points,
        y=final_tension,
        mode="lines",
        name="Dramatic Tension & Stakes",
        line=dict(color="#FF5C5C", width=3, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(255, 92, 92, 0.12)"
    ))

    act1_end = total_duration * 0.25
    act2_end = total_duration * 0.75

    fig.add_vline(x=act1_end, line_dash="dash", line_color="#4DA3FF", annotation_text="End Act I (Inciting Turn)", annotation_position="top left")
    fig.add_vline(x=total_duration * 0.50, line_dash="dot", line_color="#FFB84D", annotation_text="Midpoint Climax", annotation_position="top left")
    fig.add_vline(x=act2_end, line_dash="dash", line_color="#00C7B7", annotation_text="Act II -> Act III Climax", annotation_position="top left")

    def fmt(s):
        return f"{int(s//60):02d}:{int(s%60):02d}"

    tick_vals = list(range(0, int(total_duration) + 1, max(10, int(total_duration // 8) or 10)))
    fig.update_layout(
        title="📈 3-Act Dramatic Tension & Narrative Stakes Curve",
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(title="Film Timecode", tickvals=tick_vals, ticktext=[fmt(v) for v in tick_vals]),
        yaxis=dict(title="Dramatic Stakes / Tension (0-100)", range=[0, 105]),
        showlegend=False
    )
    return fig


def compute_cut_diff(cut_a: dict, cut_b: dict) -> dict:
    """Compute shot-by-shot, scene-by-scene, and rhythm differences between Cut A and Cut B."""
    dur_a = cut_a.get("total_duration", 0.0)
    dur_b = cut_b.get("total_duration", 0.0)

    shots_a = cut_a.get("shots", [])
    shots_b = cut_b.get("shots", [])

    asl_a = np.mean([s[1] - s[0] for s in shots_a]) if shots_a else 0.0
    asl_b = np.mean([s[1] - s[0] for s in shots_b]) if shots_b else 0.0

    shot_changes = []
    trimmed_count = 0
    extended_count = 0

    min_len = min(len(shots_a), len(shots_b))
    for i in range(min_len):
        sa_start, sa_end = shots_a[i]
        sb_start, sb_end = shots_b[i]
        sa_dur = sa_end - sa_start
        sb_dur = sb_end - sb_start
        diff_s = round(sb_dur - sa_dur, 2)

        if diff_s < -0.35:
            change_type = "TRIMMED"
            trimmed_count += 1
            desc = f"Trimmed by {abs(diff_s):.1f}s ({sa_dur:.1f}s -> {sb_dur:.1f}s)"
        elif diff_s > 0.35:
            change_type = "EXTENDED"
            extended_count += 1
            desc = f"Extended by +{diff_s:.1f}s ({sa_dur:.1f}s -> {sb_dur:.1f}s)"
        else:
            change_type = "UNCHANGED"
            desc = f"Preserved ({sb_dur:.1f}s)"

        shot_changes.append({
            "shot_index": i + 1,
            "cut_a_start": sa_start,
            "cut_a_dur": sa_dur,
            "cut_b_start": sb_start,
            "cut_b_dur": sb_dur,
            "delta_seconds": diff_s,
            "change_type": change_type,
            "description": desc
        })

    deleted_shots_count = max(0, len(shots_a) - len(shots_b))
    added_shots_count = max(0, len(shots_b) - len(shots_a))

    dlg_a = set(t.get("text", "").strip() for t in cut_a.get("transcript", []) if len(t.get("text", "").strip()) > 8)
    dlg_b = set(t.get("text", "").strip() for t in cut_b.get("transcript", []) if len(t.get("text", "").strip()) > 8)

    dropped_lines = list(dlg_a - dlg_b)
    new_lines = list(dlg_b - dlg_a)

    return {
        "duration_a": dur_a,
        "duration_b": dur_b,
        "duration_delta": round(dur_b - dur_a, 2),
        "shot_count_a": len(shots_a),
        "shot_count_b": len(shots_b),
        "shot_count_delta": len(shots_b) - len(shots_a),
        "asl_a": round(asl_a, 2),
        "asl_b": round(asl_b, 2),
        "asl_delta": round(asl_b - asl_a, 2),
        "trimmed_count": trimmed_count,
        "extended_count": extended_count,
        "deleted_shots_count": deleted_shots_count,
        "added_shots_count": added_shots_count,
        "shot_changes": shot_changes,
        "dropped_lines": dropped_lines[:15],
        "new_lines": new_lines[:15]
    }


def generate_comparative_timeline_figure(cut_a: dict, cut_b: dict, diff_data: dict) -> go.Figure:
    """Generate dual stacked comparative timeline displaying Cut A (Prior) vs Cut B (Current)."""
    fig = go.Figure()

    shots_a = cut_a.get("shots", [])
    for idx, (s_start, s_end) in enumerate(shots_a[:100]):
        color = "#4DA3FF" if idx % 2 == 0 else "#2E6BB0"
        fig.add_shape(
            type="rect",
            x0=s_start, x1=s_end,
            y0=1.1, y1=1.9,
            fillcolor=color,
            line=dict(color="#161B22", width=1),
        )

    for item in diff_data.get("shot_changes", [])[:100]:
        sb_start = item["cut_b_start"]
        sb_end = sb_start + item["cut_b_dur"]

        if item["change_type"] == "TRIMMED":
            fill_color = "#FFB84D"
        elif item["change_type"] == "EXTENDED":
            fill_color = "#00C7B7"
        else:
            fill_color = "#3A86FF" if item["shot_index"] % 2 == 0 else "#2563EB"

        fig.add_shape(
            type="rect",
            x0=sb_start, x1=sb_end,
            y0=0.1, y1=0.9,
            fillcolor=fill_color,
            line=dict(color="#161B22", width=1),
        )

    max_dur = max(cut_a.get("total_duration", 100), cut_b.get("total_duration", 100))

    def fmt(s):
        return f"{int(s//60):02d}:{int(s%60):02d}"

    tick_vals = list(range(0, int(max_dur) + 1, max(10, int(max_dur // 8) or 10)))

    fig.update_layout(
        title="🔀 Comparative Cut Timeline (Cut A vs. Cut B)",
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=280,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(title="Timeline Timecode", tickvals=tick_vals, ticktext=[fmt(v) for v in tick_vals]),
        yaxis=dict(
            tickvals=[0.5, 1.5],
            ticktext=["Cut B (Current)", "Cut A (Prior)"],
            range=[0, 2.1],
            showgrid=False
        ),
        showlegend=False
    )
    return fig


PROJECTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_projects")

def list_saved_projects(projects_dir: str = PROJECTS_DIR) -> list:
    """List all saved review projects with metadata."""
    if not os.path.exists(projects_dir):
        os.makedirs(projects_dir, exist_ok=True)
        return []

    projects = []
    for f in os.listdir(projects_dir):
        if f.endswith(".json"):
            fp = os.path.join(projects_dir, f)
            try:
                with open(fp, "r") as jf:
                    data = json.load(jf)
                p_name = data.get("project_name") or f.replace(".json", "").replace("_", " ")
                dur = data.get("total_duration", 0.0)
                shots_cnt = len(data.get("shots", []))
                saved_at = data.get("saved_at") or datetime.fromtimestamp(os.path.getmtime(fp)).strftime("%Y-%m-%d %H:%M")
                projects.append({
                    "filename": f,
                    "path": fp,
                    "name": p_name,
                    "duration": dur,
                    "shot_count": shots_cnt,
                    "saved_at": saved_at,
                    "label": f"🎬 {p_name} ({dur/60:.0f}m / {shots_cnt} shots)"
                })
            except Exception:
                continue
    return sorted(projects, key=lambda x: x["saved_at"], reverse=True)


def save_project_to_library(results: dict, project_name: str, projects_dir: str = PROJECTS_DIR) -> str:
    """Save an analysis result dictionary to the local saved_projects library."""
    if not os.path.exists(projects_dir):
        os.makedirs(projects_dir, exist_ok=True)

    clean_name = re.sub(r'[^A-Za-z0-9_-]', '_', project_name).strip('_') or "video_review"
    filename = f"{clean_name}.json"
    target_path = os.path.join(projects_dir, filename)

    save_data = {
        **results,
        "project_name": project_name,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(target_path, "w") as f:
        json.dump(save_data, f, indent=2)
    return target_path


def load_project_from_library(file_path: str) -> dict:
    """Load a project JSON review file."""
    with open(file_path, "r") as f:
        return json.load(f)


def get_claude_client():
    api_key = None
    if hasattr(st, "secrets"):
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            api_key = None
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or AnthropicClient is None:
        return None
    return AnthropicClient(api_key=api_key)


# --------------------------------------------------------------------------
# Cached pipeline steps
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def download_video(url: str, video_password: Optional[str] = None) -> str:
    """Download a 360p clip (first MAX_DURATION_SECONDS) of the given YouTube URL.
    Returns the local file path. Cached by URL so re-runs are instant."""
    import time
    out_dir = os.path.join(tempfile.gettempdir(), "editorial_review_poc")
    os.makedirs(out_dir, exist_ok=True)
    cache_key = f"{url}_{video_password or ''}"
    out_path = os.path.join(out_dir, f"{url_hash(cache_key)}.mp4")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    ffmpeg_path = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if not ffmpeg_path:
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_path = None
    if not ffmpeg_path:
        raise RuntimeError("Could not download video: ffmpeg binary not found. Install ffmpeg or set FFMPEG_BINARY.")

    # Base options with anti-bot headers
    base_opts = {
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        },
        "socket_timeout": 30,
        "ffmpeg_location": ffmpeg_path,
        "quiet": True,
        "no_warnings": True,
    }
    if video_password:
        base_opts["videopassword"] = video_password

    ydl_opts = {
        **base_opts,
        "format": "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360]",
        "outtmpl": out_path,
        "merge_output_format": "mp4",
        "download_ranges": yt_dlp.utils.download_range_func(None, [(0, MAX_DURATION_SECONDS)]),
        "force_keyframes_at_cuts": True,
    }

    def _download(opts):
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

    fallback_attempts = [
        {
            **base_opts,
            "format": "best[height<=360]",
            "outtmpl": out_path,
            "merge_output_format": "mp4",
            "geo_bypass": True,
        },
        {
            **base_opts,
            "format": "best[height<=360][ext=mp4]/best[height<=360]",
            "outtmpl": out_path,
            "merge_output_format": "mp4",
            "allow_unplayable_formats": True,
            "geo_bypass": True,
        },
        {
            **base_opts,
            "format": "best[height<=360]",
            "outtmpl": out_path,
            "merge_output_format": "mp4",
            "geo_bypass": True,
            "extractor_args": {"youtube": {"skip": ["hls", "dash"]}},
        },
    ]

    # Try primary format with retries
    last_error = None
    for attempt in range(DOWNLOAD_RETRY_ATTEMPTS):
        try:
            _download(ydl_opts)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return out_path
        except Exception as exc:
            last_error = exc
            if attempt < DOWNLOAD_RETRY_ATTEMPTS - 1:
                time.sleep(DOWNLOAD_RETRY_DELAY)
            else:
                original_error = str(exc)
                break

    # Try fallback formats
    for fallback_opts in fallback_attempts:
        for attempt in range(DOWNLOAD_RETRY_ATTEMPTS):
            try:
                _download(fallback_opts)
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    return out_path
            except Exception as exc2:
                last_error = exc2
                if attempt < DOWNLOAD_RETRY_ATTEMPTS - 1:
                    time.sleep(DOWNLOAD_RETRY_DELAY)

    if last_error is not None:
        raise RuntimeError(
            "Could not download video: %s\n\n" 
            "**Suggestions:**\n"
            "1. Try a different YouTube URL\n"
            "2. Try uploading a local video file instead\n"
            "3. Check that the video is publicly available (not private/restricted)\n"
            "4. Wait a few minutes and try again (YouTube may be rate-limiting)" % (
                str(last_error)[:200]
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


def classify_shot_scale(frame_bgr: np.ndarray) -> dict:
    """Classify framing/shot scale (CU, MCU, MS, WS, EWS, INS) using skin saliency and edge distribution."""
    h, w = frame_bgr.shape[:2]
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    cx0, cy0, cx1, cy1 = int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)
    center_skin = skin_mask[cy0:cy1, cx0:cx1]
    skin_ratio = np.sum(center_skin > 0) / float((cx1 - cx0) * (cy1 - cy0))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(center_skin)
    max_component_ratio = 0.0
    max_component_height_ratio = 0.0
    if num_labels > 1:
        largest_comp_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        max_area = stats[largest_comp_idx, cv2.CC_STAT_AREA]
        comp_h = stats[largest_comp_idx, cv2.CC_STAT_HEIGHT]
        max_component_ratio = max_area / float((cx1 - cx0) * (cy1 - cy0))
        max_component_height_ratio = comp_h / float(h)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edge_density = np.mean(edges > 0)

    if max_component_height_ratio >= 0.45 or skin_ratio >= 0.25:
        scale = "CU"
        label = "Close-Up"
    elif max_component_height_ratio >= 0.22 or skin_ratio >= 0.08:
        scale = "MCU"
        label = "Medium Close-Up"
    elif max_component_height_ratio >= 0.08 or skin_ratio >= 0.02:
        scale = "MS"
        label = "Medium Shot"
    elif skin_ratio > 0.005:
        scale = "WS"
        label = "Wide Shot"
    else:
        if edge_density < 0.04:
            scale = "EWS"
            label = "Extreme Wide / Landscape"
        elif edge_density > 0.15:
            scale = "INS"
            label = "Insert Shot"
        else:
            scale = "WS"
            label = "Wide Shot"

    return {
        "scale": scale,
        "label": label,
        "skin_ratio": round(skin_ratio, 3),
        "comp_height_ratio": round(max_component_height_ratio, 3),
        "edge_density": round(edge_density, 3)
    }


def extract_shot_thumbnails(video_path: str, shots: list, max_shots: int = 60, thumb_w: int = 240, thumb_h: int = 135) -> list:
    """Extract representative keyframe thumbnails and classify framing across detected shots."""
    if not video_path or not os.path.exists(video_path) or not shots:
        return []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    results = []
    stride = max(1, len(shots) // max_shots) if len(shots) > max_shots else 1

    for idx in range(0, len(shots), stride):
        s_start, s_end = shots[idx]
        mid_time = (s_start + s_end) / 2.0
        frame_idx = min(total_frames - 1, int(round(mid_time * fps)))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        classification = classify_shot_scale(frame)

        thumb = cv2.resize(frame, (thumb_w, thumb_h))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        _, buffer = cv2.imencode('.jpg', thumb, encode_param)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        data_uri = f"data:image/jpeg;base64,{b64_str}"

        results.append({
            "shot_index": idx + 1,
            "start": s_start,
            "end": s_end,
            "duration": round(s_end - s_start, 2),
            "thumbnail_b64": data_uri,
            **classification
        })

    cap.release()
    return results


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


def detect_lj_cuts(shots: list, transcript: list, tolerance_s: float = 1.5) -> list:
    """Identify J-Cuts (audio leads picture) and L-Cuts (audio lags/continues across visual cuts)."""
    transitions = []
    shot_boundaries = [s[0] for s in shots[1:]]
    for cut_time in shot_boundaries:
        intersecting = []
        for seg in transcript:
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            if seg_start < (cut_time - 0.2) and seg_end > (cut_time + 0.4):
                intersecting.append({
                    "type": "L-CUT",
                    "label": "L-Cut (Dialogue extends over visual cut)",
                    "lead_seconds": round(seg_end - cut_time, 2),
                    "text": seg.get("text", "")
                })
            elif (cut_time - tolerance_s) <= seg_start < (cut_time - 0.1) and seg_end > cut_time:
                intersecting.append({
                    "type": "J-CUT",
                    "label": "J-Cut (Audio anticipates visual cut)",
                    "lead_seconds": round(cut_time - seg_start, 2),
                    "text": seg.get("text", "")
                })
        if intersecting:
            transitions.append({
                "cut_time": cut_time,
                "split_edits": intersecting
            })
    return transitions


def cluster_speakers_from_transcript(transcript: list, max_speakers: int = 3) -> list:
    """Heuristic conversational speaker turn clustering based on pauses and turn-taking flow."""
    if not transcript:
        return []
    labeled_transcript = []
    current_speaker = 1
    for idx, seg in enumerate(transcript):
        if idx == 0:
            labeled_transcript.append({
                **seg,
                "speaker": f"Speaker {current_speaker}",
                "speaker_id": current_speaker
            })
            continue
        prev_seg = transcript[idx - 1]
        gap = seg.get("start", 0.0) - prev_seg.get("end", 0.0)
        text_lower = seg.get("text", "").lower().strip()
        is_short_reaction = len(text_lower.split()) <= 4
        if 0.4 <= gap <= 3.5 or is_short_reaction:
            current_speaker = 2 if current_speaker == 1 else 1
        elif gap > 5.0:
            current_speaker = (current_speaker % max_speakers) + 1
        labeled_transcript.append({
            **seg,
            "speaker": f"Speaker {current_speaker}",
            "speaker_id": current_speaker
        })
    return labeled_transcript


def compute_audio_editorial_metrics(shots: list, labeled_transcript: list) -> dict:
    """Compute dialogue share per speaker, split-edit ratio, and transition smoothness."""
    if not labeled_transcript:
        return {
            "speaker_shares": {},
            "speaker_durations": {},
            "total_dialogue_seconds": 0.0,
            "l_cut_count": 0,
            "j_cut_count": 0,
            "split_edit_count": 0,
            "hard_cut_count": max(0, len(shots) - 1),
            "split_edit_index": 0.0,
            "lj_transitions": []
        }
    speaker_durations = {}
    for seg in labeled_transcript:
        spk = seg.get("speaker", "Speaker 1")
        dur = seg.get("end", 0.0) - seg.get("start", 0.0)
        speaker_durations[spk] = speaker_durations.get(spk, 0.0) + dur
    total_dialogue_time = sum(speaker_durations.values()) or 1.0
    speaker_shares = {
        spk: round((dur / total_dialogue_time) * 100, 1)
        for spk, dur in speaker_durations.items()
    }
    lj_transitions = detect_lj_cuts(shots, labeled_transcript)
    l_cuts = sum(1 for t in lj_transitions for e in t["split_edits"] if e["type"] == "L-CUT")
    j_cuts = sum(1 for t in lj_transitions for e in t["split_edits"] if e["type"] == "J-CUT")
    total_cuts = max(1, len(shots) - 1)
    split_ratio = round(((l_cuts + j_cuts) / total_cuts) * 100, 1)
    return {
        "speaker_shares": speaker_shares,
        "speaker_durations": {k: round(v, 1) for k, v in speaker_durations.items()},
        "total_dialogue_seconds": round(total_dialogue_time, 1),
        "l_cut_count": l_cuts,
        "j_cut_count": j_cuts,
        "split_edit_count": l_cuts + j_cuts,
        "hard_cut_count": max(0, total_cuts - (l_cuts + j_cuts)),
        "split_edit_index": split_ratio,
        "lj_transitions": lj_transitions
    }


@st.cache_data(show_spinner=False)
def check_narrative_consistency(transcript_key: str, early_text: str, late_text: str) -> str:
    """Single GPT-4o-mini call comparing early vs. late dialogue for continuity.
    `transcript_key` exists purely so the cache is keyed per-video."""
    client = get_claude_client()
    if client is None:
        return (
            "⚠️ No Anthropic API key found (set ANTHROPIC_API_KEY in st.secrets or the "
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
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=300,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
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


def youtube_embed(video_id: str, start_seconds: int = 0, height: int = 380, platform: str = "youtube"):
    """Embed video player via IFrame API. Supports YouTube, Vimeo, and Frame.io.
    `start_seconds` lets issue cards 'jump' the player to a timecode."""

    if platform == "vimeo":
        html = f"""
        <div style="padding: 62.5% 0 0 0; position: relative;">
          <iframe src="https://player.vimeo.com/video/{video_id}?h=&autoplay=0#t={int(start_seconds)}s"
            style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"
            frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen>
          </iframe>
        </div>
        <script src="https://player.vimeo.com/api/player.js"></script>
        """
    elif platform == "frameio":
        html = f"""
        <div style="padding: 62.5% 0 0 0; position: relative;">
          <iframe src="https://player.frame.io/{video_id}"
            style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"
            frameborder="0" allow="autoplay; fullscreen; picture-in-picture" allowfullscreen>
          </iframe>
        </div>
        """
    else:  # YouTube (default)
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


def seconds_to_timecode(seconds: float, fps: float = 24.0) -> str:
    """Format seconds into standard SMPTE timecode (HH:MM:SS:FF)."""
    if seconds < 0:
        seconds = 0.0
    total_frames = int(round(seconds * fps))
    frame_rate_int = int(round(fps)) or 24
    ff = total_frames % frame_rate_int
    total_secs = total_frames // frame_rate_int
    ss = total_secs % 60
    mm = (total_secs // 60) % 60
    hh = total_secs // 3600
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def seconds_to_srt_time(seconds: float) -> str:
    """Format seconds into SubRip subtitle timecode (HH:MM:SS,mmm)."""
    if seconds < 0:
        seconds = 0.0
    ms = int(round((seconds - int(seconds)) * 1000))
    total_secs = int(seconds)
    ss = total_secs % 60
    mm = (total_secs // 60) % 60
    hh = total_secs // 3600
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def seconds_to_vtt_time(seconds: float) -> str:
    """Format seconds into WebVTT subtitle timecode (HH:MM:SS.mmm)."""
    if seconds < 0:
        seconds = 0.0
    ms = int(round((seconds - int(seconds)) * 1000))
    total_secs = int(seconds)
    ss = total_secs % 60
    mm = (total_secs // 60) % 60
    hh = total_secs // 3600
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def generate_srt(transcript: list) -> str:
    """Generate standard SubRip (.srt) subtitle text from transcript segments."""
    blocks = []
    for idx, seg in enumerate(transcript, start=1):
        start_str = seconds_to_srt_time(seg["start"])
        end_str = seconds_to_srt_time(seg["end"])
        text = seg.get("text", "").strip()
        blocks.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")
    return "\n".join(blocks)


def generate_vtt(transcript: list) -> str:
    """Generate WebVTT (.vtt) subtitle text from transcript segments."""
    lines = ["WEBVTT", ""]
    for idx, seg in enumerate(transcript, start=1):
        start_str = seconds_to_vtt_time(seg["start"])
        end_str = seconds_to_vtt_time(seg["end"])
        text = seg.get("text", "").strip()
        lines.append(f"{idx}")
        lines.append(f"{start_str} --> {end_str}")
        lines.append(f"{text}\n")
    return "\n".join(lines)


def generate_edl(shots: list, pacing_issues: list, scenes: list, clip_name: str = "SOURCE_CUT", fps: float = 24.0) -> str:
    """Generate a standard CMX 3600 Edit Decision List (EDL) with markers for shots, pacing dips, and scene turns."""
    clean_name = re.sub(r'[^A-Za-z0-9_]', '_', clip_name) or "SOURCE_CUT"
    lines = [
        f"TITLE: {clean_name.upper()}_EDITORIAL_REVIEW",
        "FCM: NON-DROP FRAME",
        ""
    ]
    for idx, (start_s, end_s) in enumerate(shots, start=1):
        src_in = seconds_to_timecode(start_s, fps)
        src_out = seconds_to_timecode(end_s, fps)
        rec_in = seconds_to_timecode(start_s, fps)
        rec_out = seconds_to_timecode(end_s, fps)
        lines.append(f"{idx:03d}  AX       V     C        {src_in} {src_out} {rec_in} {rec_out}")
        lines.append(f"* FROM CLIP NAME: {clip_name}")

        matching_scenes = [sc for sc in scenes if abs(sc["start"] - start_s) < 0.5]
        for sc in matching_scenes:
            dlg = (sc.get("dialogue", "") or "").replace("\n", " ")[:60]
            lines.append(f"* MARKER: SCENE {sc['index']:02d} | Duration: {sc['duration']:.0f}s | {dlg}")

        for p in pacing_issues:
            if p["start"] <= start_s < p["end"]:
                kind = "PACING DIP" if p["type"] == "dip" else "RUSH ZONE"
                lines.append(f"* MARKER: [{kind}] {p['message']}")
        lines.append("")
    return "\n".join(lines)


def generate_resolve_marker_csv(shots: list, pacing_issues: list, scenes: list, fps: float = 24.0) -> str:
    """Generate DaVinci Resolve compatible Marker CSV format."""
    rows = ["EDL Event,Name,In,Out,Color,Comment,Duration"]
    for sc in scenes:
        tc_in = seconds_to_timecode(sc["start"], fps)
        tc_out = seconds_to_timecode(sc["end"], fps)
        frames_dur = int(round(sc["duration"] * fps))
        dlg = (sc.get("dialogue", "") or "").replace('"', '""').replace('\n', ' ')[:100]
        rows.append(f'SC_{sc["index"]:02d},"Scene {sc["index"]:02d}",{tc_in},{tc_out},Blue,"{dlg}",{frames_dur}')

    for idx, p in enumerate(pacing_issues, start=1):
        color = "Yellow" if p["type"] == "dip" else "Red"
        name = "Pacing Dip" if p["type"] == "dip" else "Rush Cut Zone"
        tc_in = seconds_to_timecode(p["start"], fps)
        tc_out = seconds_to_timecode(p["end"], fps)
        dur = p["end"] - p["start"]
        frames_dur = int(round(dur * fps))
        msg = p["message"].replace('"', '""')
        rows.append(f'PACE_{idx:02d},"{name}",{tc_in},{tc_out},{color},"{msg}",{frames_dur}')
    return "\n".join(rows)


def generate_fcpxml(shots: list, pacing_issues: list, scenes: list, total_duration: float, clip_name: str = "Source Cut", fps: float = 24.0, width: int = 1920, height: int = 1080) -> str:
    """Generate Final Cut Pro / Premiere / DaVinci Resolve compatible XML timeline with timeline markers."""
    timebase = int(round(fps)) or 24
    ntsc = "TRUE" if (abs(fps - 23.976) < 0.05 or abs(fps - 29.97) < 0.05) else "FALSE"
    total_frames = int(round(total_duration * fps))

    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE xmeml>',
        '<xmeml version="5">',
        '  <sequence id="sequence-1">',
        f'    <name>{saxutils.escape(clip_name)} - Editorial Review</name>',
        f'    <duration>{total_frames}</duration>',
        '    <rate>',
        f'      <timebase>{timebase}</timebase>',
        f'      <ntsc>{ntsc}</ntsc>',
        '    </rate>',
        '    <media>',
        '      <video>',
        '        <format>',
        '          <samplecharacteristics>',
        f'            <width>{width}</width>',
        f'            <height>{height}</height>',
        '            <rate>',
        f'              <timebase>{timebase}</timebase>',
        f'              <ntsc>{ntsc}</ntsc>',
        '            </rate>',
        '          </samplecharacteristics>',
        '        </format>',
        '        <track>',
    ]

    for idx, (start_s, end_s) in enumerate(shots, start=1):
        in_frame = int(round(start_s * fps))
        out_frame = int(round(end_s * fps))
        dur_frame = max(1, out_frame - in_frame)
        xml_lines.extend([
            f'          <clipitem id="clipitem-{idx}">',
            f'            <name>Shot {idx:03d}</name>',
            f'            <duration>{dur_frame}</duration>',
            '            <rate>',
            f'              <timebase>{timebase}</timebase>',
            f'              <ntsc>{ntsc}</ntsc>',
            '            </rate>',
            f'            <start>{in_frame}</start>',
            f'            <end>{out_frame}</end>',
            f'            <in>{in_frame}</in>',
            f'            <out>{out_frame}</out>',
        ])

        for sc in scenes:
            if abs(sc["start"] - start_s) < 0.5:
                dlg = (sc.get("dialogue", "") or "").replace('\n', ' ')[:80]
                xml_lines.extend([
                    '            <marker>',
                    f'              <name>Scene {sc["index"]:02d}</name>',
                    f'              <comment>{saxutils.escape(dlg)}</comment>',
                    f'              <in>{in_frame}</in>',
                    f'              <out>{in_frame}</out>',
                    '            </marker>',
                ])

        for p in pacing_issues:
            if p["start"] <= start_s < p["end"]:
                kind = "Pacing Dip" if p["type"] == "dip" else "Rush Zone"
                xml_lines.extend([
                    '            <marker>',
                    f'              <name>{kind}</name>',
                    f'              <comment>{saxutils.escape(p["message"])}</comment>',
                    f'              <in>{in_frame}</in>',
                    f'              <out>{in_frame}</out>',
                    '            </marker>',
                ])

        xml_lines.append('          </clipitem>')

    xml_lines.extend([
        '        </track>',
        '      </video>',
        '    </media>',
        '  </sequence>',
        '</xmeml>',
    ])

    return "\n".join(xml_lines)



# --------------------------------------------------------------------------
# Main analysis pipeline (drives the spinner / progress steps)
# --------------------------------------------------------------------------
def run_analysis(url: str, script_data=None, progress_bar=None, status_text=None, local_video_path: Optional[str] = None, video_password: Optional[str] = None):
    steps = [
        "Preparing video...",
        "Transcribing audio...",
        "Detecting shots...",
        "Calculating pacing metrics...",
        "Checking narrative consistency...",
        "Generating report...",
    ]
    n = len(steps)

    status_text.text(steps[0]); progress_bar.progress(1 / n)
    if local_video_path:
        video_path = local_video_path
    else:
        video_path = download_video(url, video_password=video_password)

    status_text.text(steps[1]); progress_bar.progress(2 / n)
    transcript_raw = transcribe_audio(video_path)
    transcript = cluster_speakers_from_transcript(transcript_raw)

    status_text.text(steps[2]); progress_bar.progress(3 / n)
    shots = detect_shots(video_path)
    motion = compute_motion_energy(video_path)
    filmstrip = extract_shot_thumbnails(video_path, shots) if video_path and os.path.exists(video_path) else []

    status_text.text(steps[3]); progress_bar.progress(4 / n)
    cap = cv2.VideoCapture(video_path)
    total_duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25.0))
    cap.release()
    pacing_issues = compute_pacing_issues(shots) if shots else []
    scenes = detect_scenes(transcript, shots, total_duration)
    audio_metrics = compute_audio_editorial_metrics(shots, transcript)

    status_text.text(steps[4]); progress_bar.progress(5 / n)
    full_dialogue = [t["text"] for t in transcript if t["text"]]
    early_text = " ".join(full_dialogue[:30]) or "(no dialogue detected)"
    late_text = " ".join(full_dialogue[-30:]) or "(no dialogue detected)"
    narrative_result = check_narrative_consistency(url_hash(url), early_text, late_text)

    # Process screenplay / script data
    if isinstance(script_data, dict):
        screenplay_info = script_data
        script_text = script_data.get("raw_text", "")
    elif isinstance(script_data, str):
        screenplay_info = parse_screenplay_text(script_data)
        script_text = script_data
    else:
        screenplay_info = None
        script_text = None

    script_diff = compare_to_script(scenes, script_text) if script_text else None
    script_coverage = analyze_script_coverage(scenes, screenplay_info) if screenplay_info else None

    status_text.text(steps[5]); progress_bar.progress(1.0)

    return {
        "video_path": video_path,
        "transcript": transcript,
        "shots": shots,
        "motion": motion,
        "filmstrip": filmstrip,
        "total_duration": total_duration,
        "pacing_issues": pacing_issues,
        "scenes": scenes,
        "audio_metrics": audio_metrics,
        "narrative_result": narrative_result,
        "script_diff": script_diff,
        "script_coverage": script_coverage,
        "screenplay_info": screenplay_info,
    }


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
def main():
    st.title("🎬 AI Editorial Review")
    st.markdown('<span class="accent">Proof of Concept</span> — pacing, scene structure '
                'and narrative-consistency analysis for short films.', unsafe_allow_html=True)

    if cv2 is None:
        st.error(
            "OpenCV (cv2) failed to import. This usually means a missing system GUI library (libGL). "
            "Ensure the host provides system packages listed in `packages.txt` (ffmpeg, libgl1-mesa-glx, etc.) "
            "or install opencv-python-headless instead of opencv-python. Full error: %s" % (CV2_IMPORT_ERROR)
        )
        return

    with st.sidebar:
        st.header("Settings")
        teaching_mode = st.checkbox("📚 Teaching Mode", value=False,
                                     help="Expand every issue card with an educational explanation.")

        st.divider()
        st.subheader("📁 Project Library")
        saved_list = list_saved_projects()
        if saved_list:
            proj_labels = [p["label"] for p in saved_list]
            selected_proj_label = st.selectbox("Saved Reviews:", proj_labels)
            if st.button("📂 Load Selected Project", use_container_width=True):
                chosen_proj = next(p for p in saved_list if p["label"] == selected_proj_label)
                st.session_state.results = load_project_from_library(chosen_proj["path"])
                st.session_state.seek_to = 0
                st.rerun()
        else:
            st.caption("No saved projects found in library.")

        # Save active project to library
        if st.session_state.get("results"):
            with st.expander("💾 Save Active Review to Library", expanded=False):
                current_pname = st.session_state.results.get("project_name") or f"Review_{st.session_state.results.get('video_id') or 'Video'}"
                new_proj_name = st.text_input("Project Name:", value=current_pname)
                if st.button("💾 Save Project", use_container_width=True):
                    saved_path = save_project_to_library(st.session_state.results, new_proj_name)
                    st.success(f"Saved to {os.path.basename(saved_path)}!")
                    st.rerun()

        # Start new analysis / clear view
        if st.button("➕ New Video (Clear View)", use_container_width=True):
            st.session_state.results = None
            st.session_state.seek_to = 0
            st.rerun()

        st.divider()
        st.subheader("Analyze New Video")

        input_method = st.radio("Source", ["YouTube/Vimeo URL", "Local Video"], horizontal=True)

        url = None
        video_password = ""
        local_video = None
        if input_method == "YouTube/Vimeo URL":
            url = st.text_input("YouTube, Vimeo, or Frame.io URL", value="")
            video_password = st.text_input("Password (if required):", type="password", help="For password-protected Vimeo or review links.")
        else:
            st.markdown(f"**Max file size: {MAX_VIDEO_SIZE_MB}MB**")
            local_video = st.file_uploader("Upload video file", type=["mp4", "mov", "avi", "mkv", "webm"])
            if local_video is not None:
                file_size_mb = local_video.size / (1024 * 1024)
                if file_size_mb > MAX_VIDEO_SIZE_MB:
                    st.error(f"❌ File too large: {file_size_mb:.1f}MB (max: {MAX_VIDEO_SIZE_MB}MB)\n\nTry:\n- Using the YouTube URL option instead\n- Compressing the video with ffmpeg: `ffmpeg -i input.mp4 -vcodec libx265 -crf 28 output.mp4`\n- Trimming the video to under 5 minutes")
                    local_video = None
                else:
                    st.success(f"✓ {file_size_mb:.1f}MB")

        script_file = st.file_uploader("Upload Screenplay (.pdf, .fdx, .txt, .fountain)", type=["pdf", "fdx", "txt", "fountain"])
        analyse_clicked = st.button("Analyse", use_container_width=True)
        demo_clicked = st.button("Run Demo (Sample clip)", use_container_width=True)
        st.caption("First run downloads the Whisper 'base' model (~140MB) — this only happens once.")

        st.divider()
        st.subheader("🔀 Version Diff Mode")
        diff_mode = st.toggle("Enable Cut Comparison", value=False, help="Compare current cut with a prior version or rough cut.")
        diff_source = None
        if diff_mode:
            diff_option = st.selectbox("Comparison Target:", ["Simulated Assembly / Rough Cut (+2.4 min)", "Upload Prior Cut JSON"])
            if diff_option == "Upload Prior Cut JSON":
                diff_file = st.file_uploader("Prior Cut JSON (.json)", type=["json"], key="diff_json_upload")
                if diff_file is not None:
                    try:
                        diff_source = json.load(diff_file)
                    except Exception:
                        st.error("Invalid JSON file.")
            else:
                diff_source = "SIMULATED_ROUGH"

        if get_claude_client() is None:
            st.warning("No ANTHROPIC_API_KEY found — narrative consistency check will be skipped.")

    if "results" not in st.session_state:
        if os.path.exists("latest_analysis.json"):
            try:
                with open("latest_analysis.json") as f:
                    st.session_state.results = json.load(f)
            except Exception:
                st.session_state.results = None
        else:
            st.session_state.results = None

    if "seek_to" not in st.session_state:
        st.session_state.seek_to = 0

    target_url = None
    video_path = None
    platform = "youtube"
    if demo_clicked:
        target_url = DEMO_URL
        platform = "youtube"
    elif analyse_clicked:
        if input_method == "Local Video":
            if local_video is not None:
                temp_dir = os.path.join(tempfile.gettempdir(), "editorial_review_poc")
                os.makedirs(temp_dir, exist_ok=True)
                video_path = os.path.join(temp_dir, local_video.name)
                with open(video_path, "wb") as f:
                    f.write(local_video.getbuffer())
            else:
                st.error("Please select a video file to analyse.")
                return
        else:
            if not url or not url.strip():
                st.error("Please enter a valid YouTube, Vimeo, or video link.")
                return
            target_url = url.strip()

    if target_url:
        video_id, platform = extract_video_id_and_platform(target_url)
        if not video_id:
            st.error("Could not parse video URL. Please check the link.")
            return

        script_data = read_script_file(script_file) if script_file is not None else None

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        try:
            with st.spinner("Running editorial analysis..."):
                pwd_param = video_password.strip() if video_password else None
                results = run_analysis(target_url, script_data, progress_bar, status_text, video_password=pwd_param)
            results["video_id"] = video_id
            results["url"] = target_url
            results["platform"] = platform
            results["project_name"] = f"Cut_{video_id}"
            st.session_state.results = results
            st.session_state.seek_to = 0
            status_text.text("Done.")
        except RuntimeError as exc:
            st.error(f"Could not analyse this video: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the user
            st.error(f"Unexpected error during analysis: {exc}")
            return
    elif video_path:
        script_data = read_script_file(script_file) if script_file is not None else None

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        try:
            with st.spinner("Running editorial analysis..."):
                results = run_analysis(
                    url="",
                    script_data=script_data,
                    progress_bar=progress_bar,
                    status_text=status_text,
                    local_video_path=video_path
                )
            clean_local_name = os.path.splitext(os.path.basename(video_path))[0]
            results["video_id"] = None
            results["url"] = None
            results["platform"] = "local"
            results["project_name"] = f"Upload_{clean_local_name}"
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
        st.info("🎬 **Ready to Analyze**\n\n- Paste a YouTube, Vimeo, or video URL in the sidebar and click **Analyse**\n- Or upload a local video file\n- Or select a saved project from the **📁 Project Library** in the sidebar\n- Or click **Run Demo** to try a sample clip.")
        return

    # ---- Dashboard ----
    st.subheader("Source video")
    if results.get("video_id"):
        youtube_embed(results["video_id"], start_seconds=st.session_state.seek_to, platform=results.get("platform", "youtube"))
    else:
        st.info("ℹ️ This analysis was performed on a locally uploaded video file.")

    st.subheader("Pacing & shot timeline")
    scene_boundaries = [sc["start"] for sc in results["scenes"]]
    fig = build_timeline_figure(
        results["shots"], results["motion"], results["pacing_issues"],
        scene_boundaries, results["total_duration"],
    )
    st.plotly_chart(fig, use_container_width=True)

    fig_tension = generate_dramatic_tension_curve(
        results["total_duration"], results["shots"], results["pacing_issues"], results["scenes"]
    )
    st.plotly_chart(fig_tension, use_container_width=True)

    # ---- Visual Framing & Coverage Breakdown ----
    filmstrip = results.get("filmstrip") or []
    if filmstrip:
        st.subheader("🖼️ Framing & Shot Scale Coverage")

        cu_count = sum(1 for f in filmstrip if f.get("scale") in ("CU", "ECU"))
        ms_count = sum(1 for f in filmstrip if f.get("scale") in ("MS", "MCU"))
        ws_count = sum(1 for f in filmstrip if f.get("scale") in ("WS", "EWS"))
        ins_count = sum(1 for f in filmstrip if f.get("scale") == "INS")
        total_f = len(filmstrip) or 1

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🔍 Close-Ups (CU / ECU)", f"{cu_count / total_f * 100:.0f}%", f"{cu_count} shots")
        m2.metric("🧍 Medium Shots (MS / MCU)", f"{ms_count / total_f * 100:.0f}%", f"{ms_count} shots")
        m3.metric("🌄 Wide Shots (WS / EWS)", f"{ws_count / total_f * 100:.0f}%", f"{ws_count} shots")
        m4.metric("🔎 Inserts & Details", f"{ins_count / total_f * 100:.0f}%", f"{ins_count} shots")

        with st.expander("🎞️ Visual Filmstrip & Keyframe Explorer", expanded=True):
            col_filter, col_sort = st.columns([2, 1])
            with col_filter:
                filter_choice = st.selectbox(
                    "Filter filmstrip by shot type:",
                    ["All Shots", "🔍 Close-Ups Only (CU/ECU)", "🧍 Medium Shots Only (MS/MCU)", "🌄 Wide Shots Only (WS/EWS)", "⚠️ Pacing Dips Only"]
                )
            with col_sort:
                st.caption(f"Displaying keyframes for {total_f} representative shots.")

            filtered_shots = filmstrip
            if filter_choice == "🔍 Close-Ups Only (CU/ECU)":
                filtered_shots = [f for f in filmstrip if f.get("scale") in ("CU", "ECU")]
            elif filter_choice == "🧍 Medium Shots Only (MS/MCU)":
                filtered_shots = [f for f in filmstrip if f.get("scale") in ("MS", "MCU")]
            elif filter_choice == "🌄 Wide Shots Only (WS/EWS)":
                filtered_shots = [f for f in filmstrip if f.get("scale") in ("WS", "EWS")]
            elif filter_choice == "⚠️ Pacing Dips Only":
                filtered_shots = [f for f in filmstrip if f.get("is_pacing_dip")]

            num_cols = 4
            for r_idx in range(0, len(filtered_shots), num_cols):
                cols = st.columns(num_cols)
                row_items = filtered_shots[r_idx:r_idx + num_cols]
                for c_idx, shot_item in enumerate(row_items):
                    with cols[c_idx]:
                        badge_cls = "badge-cu" if shot_item.get("scale") in ("CU", "ECU") else ("badge-ms" if shot_item.get("scale") in ("MS", "MCU") else "badge-ws")
                        dip_tag = '<span class="scale-badge badge-dip">⚠️ HOLD</span>' if shot_item.get("is_pacing_dip") else ''
                        st.markdown(f"""
                        <div class="filmstrip-card">
                            <img src="{shot_item['thumbnail_b64']}" alt="Shot {shot_item['shot_index']}" style="width: 100%; border-radius: 6px; margin-bottom: 6px;" />
                            <div style="font-weight: 700; font-size: 0.9rem;">Shot #{shot_item['shot_index']:02d}</div>
                            <div style="color: #00C7B7; font-family: monospace; font-size: 0.8rem;">{fmt_time(shot_item['start'])} – {fmt_time(shot_item['end'])} ({shot_item['duration']:.1f}s)</div>
                            <div style="margin-top: 4px;">
                                <span class="scale-badge {badge_cls}">[{shot_item.get('scale', 'WS')}] {shot_item.get('label', 'Shot')}</span>
                                {dip_tag}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button(f"⏩ Seek {fmt_time(shot_item['start'])}", key=f"filmstrip_seek_{shot_item['shot_index']}"):
                            st.session_state.seek_to = int(shot_item['start'])
                            st.rerun()

    # ---- Cut Version Comparison & Diff Mode ----
    if diff_mode and results:
        if diff_source == "SIMULATED_ROUGH":
            prior_cut = {
                **results,
                "total_duration": results["total_duration"] + 145.0,
                "shots": [(s[0] * 1.036, s[1] * 1.036 + 0.4) for s in results["shots"]],
                "transcript": results.get("transcript", [])
            }
        elif isinstance(diff_source, dict):
            prior_cut = diff_source
        else:
            prior_cut = None

        if prior_cut:
            st.divider()
            st.subheader("🔀 Cut Revision & Version Comparison (Rough Cut v1 vs. Fine Cut v2)")
            diff_data = compute_cut_diff(prior_cut, results)

            d1, d2, d3, d4 = st.columns(4)
            dur_delta_s = diff_data["duration_delta"]
            sign = "+" if dur_delta_s > 0 else ""
            d1.metric("⏱️ Runtime Shift", f"{sign}{dur_delta_s:.0f}s", f"{diff_data['duration_a']:.0f}s -> {diff_data['duration_b']:.0f}s")
            d2.metric("⚡ Pacing Delta (ASL)", f"{diff_data['asl_delta']:+.2f}s", f"{diff_data['asl_a']:.1f}s -> {diff_data['asl_b']:.1f}s ASL")
            d3.metric("✂️ Trimmed Shots", f"{diff_data['trimmed_count']}", "shots tightened")
            d4.metric("➕ Extended Shots", f"{diff_data['extended_count']}", "shots lengthened")

            comp_fig = generate_comparative_timeline_figure(prior_cut, results, diff_data)
            st.plotly_chart(comp_fig, use_container_width=True)

            with st.expander("📋 Detailed Shot-by-Shot Trim & Revision Log", expanded=False):
                col_trim_filter, _ = st.columns([2, 2])
                with col_trim_filter:
                    trim_filter = st.selectbox("Filter Revision Log:", ["All Shots", "✂️ Trimmed Only", "➕ Extended Only", "⏸️ Unchanged Only"])

                log_items = diff_data.get("shot_changes", [])
                if trim_filter == "✂️ Trimmed Only":
                    log_items = [s for s in log_items if s["change_type"] == "TRIMMED"]
                elif trim_filter == "➕ Extended Only":
                    log_items = [s for s in log_items if s["change_type"] == "EXTENDED"]
                elif trim_filter == "⏸️ Unchanged Only":
                    log_items = [s for s in log_items if s["change_type"] == "UNCHANGED"]

                for item in log_items[:40]:
                    badge_cls = "badge-dip" if item["change_type"] == "TRIMMED" else ("badge-cu" if item["change_type"] == "EXTENDED" else "badge-ms")
                    st.markdown(f"""
                    <div style="background-color: #161B22; border-left: 4px solid #30363D; border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-weight: 700;">Shot #{item['shot_index']}</span>
                            <span class="scale-badge {badge_cls}" style="margin-left: 8px;">{item['change_type']}</span>
                            <span style="color: #8B949E; margin-left: 8px; font-size: 0.85rem;">{item['description']}</span>
                        </div>
                        <span style="color: #00C7B7; font-family: monospace; font-size: 0.82rem;">{fmt_time(item['cut_b_start'])}</span>
                    </div>
                    """, unsafe_allow_html=True)

    st.divider()
    col_scenes, col_pacing = st.columns(2)

    with col_scenes:
        st.markdown("### 🟦 Scene structure")
        if not results["scenes"]:
            st.write("No scenes could be detected.")
        else:
            with st.expander("📂 Explore Detected Scene Beats", expanded=True):
                scene_toggles_col1, scene_toggles_col2 = st.columns(2)
                with scene_toggles_col1:
                    dlg_only = st.toggle("💬 Dialogue Only", value=False, help="Filter to only scenes that contain dialogue.")
                with scene_toggles_col2:
                    compact_view = st.toggle("⚡ Compact View", value=False, help="Display scenes in a condensed list format.")

                filtered_scenes = results["scenes"]
                if dlg_only:
                    filtered_scenes = [sc for sc in filtered_scenes if sc.get("dialogue")]

                st.caption(f"Showing {len(filtered_scenes)} of {len(results['scenes'])} detected scenes.")

                for sc in filtered_scenes:
                    btn_key = f"jump_scene_{sc['index']}"
                    if compact_view:
                        dlg_prev = (sc['dialogue'][:80] + "…") if sc.get("dialogue") else "(visual transition beat)"
                        st.markdown(f"""
                        <div style="background-color: #161B22; border-left: 3px solid #4DA3FF; border-radius: 6px; padding: 6px 10px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-weight: 600; font-size: 0.88rem;">Scene {sc['index']}</span>
                                <span style="color: #8B949E; font-size: 0.8rem; margin-left: 6px;">({sc['duration']:.0f}s)</span>
                                <span style="color: #C9D1D9; font-size: 0.8rem; margin-left: 10px; font-style: italic;">{dlg_prev}</span>
                            </div>
                            <span style="color: #00C7B7; font-family: monospace; font-size: 0.78rem;">{fmt_time(sc['start'])}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button(f"⏩ Jump Scene {sc['index']}", key=btn_key):
                            st.session_state.seek_to = int(sc["start"])
                            st.rerun()
                    else:
                        body = f"Duration: {sc['duration']:.0f}s"
                        if sc["dialogue"]:
                            snippet = sc["dialogue"][:140] + ("…" if len(sc["dialogue"]) > 140 else "")
                            body += f"<br><i>{snippet}</i>"
                        render_card("card-blue", f"Scene {sc['index']}",
                                    f"{fmt_time(sc['start'])} – {fmt_time(sc['end'])}",
                                    body, "scene_structure", teaching_mode, btn_key)
                        if st.button(f"⏩ Jump to Scene {sc['index']}", key=btn_key):
                            st.session_state.seek_to = int(sc["start"])
                            st.rerun()

        if results.get("script_coverage"):
            cov = results["script_coverage"]
            st.markdown("#### 📜 Screenplay Supervisor")
            st.caption("Deep line-by-line screenplay vs. cut matching.")
            cov_col1, cov_col2 = st.columns(2)
            cov_col1.metric("Script Retained", f"{cov['coverage_percent']}%", f"{cov['matched_lines_count']} lines in cut")
            cov_col2.metric("Omitted / Cut Lines", f"{cov['omitted_lines_count']} lines", "dropped from script")

            with st.expander("🔍 Detailed Script-to-Screen Breakdown", expanded=False):
                if cov.get("omitted_lines"):
                    st.markdown("**✂️ Omitted / Dropped Script Lines:**")
                    for om in cov["omitted_lines"][:10]:
                        st.caption(f"• **{om.get('character', 'CHAR')}**: {om.get('text', '')}")

                if cov.get("unmatched_cut_scenes"):
                    st.markdown("**🗣️ Improvised / Added Material in Cut:**")
                    for un in cov["unmatched_cut_scenes"][:8]:
                        st.caption(f"• **Scene {un['scene_index']}** ({fmt_time(un['start'])}): {un['dialogue']}")
        elif results.get("script_diff") is not None:
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
        else:
            with st.expander("⚠️ Explore Pacing Dips & Rush Zones", expanded=True):
                st.caption(f"Detected {len(results['pacing_issues'])} pacing alerts.")
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

    # ---- Audio & Dialogue Flow Intelligence ----
    audio_metrics = results.get("audio_metrics") or {}
    transcript_items = results.get("transcript") or []
    if audio_metrics or transcript_items:
        st.divider()
        st.subheader("🎙️ Dialogue & Audio Flow Intelligence")

        a1, a2, a3, a4 = st.columns(4)
        spk_shares = audio_metrics.get("speaker_shares", {})
        spk1_share = spk_shares.get("Speaker 1", 0.0)
        spk2_share = spk_shares.get("Speaker 2", 0.0)
        split_ratio = audio_metrics.get("split_edit_index", 0.0)
        l_cuts = audio_metrics.get("l_cut_count", 0)
        j_cuts = audio_metrics.get("j_cut_count", 0)

        a1.metric("🗣️ Speaker 1 Dialogue", f"{spk1_share:.1f}%", f"{audio_metrics.get('speaker_durations', {}).get('Speaker 1', 0):.0f}s speech")
        a2.metric("🗣️ Speaker 2 Dialogue", f"{spk2_share:.1f}%", f"{audio_metrics.get('speaker_durations', {}).get('Speaker 2', 0):.0f}s speech")
        a3.metric("✂️ Split Edit Index (L/J Cuts)", f"{split_ratio:.1f}%", f"{l_cuts + j_cuts} split transitions")
        a4.metric("🎧 L-Cuts / J-Cuts", f"{l_cuts} L / {j_cuts} J", "Smooth dialogue leads")

        with st.expander("💬 Conversational Dialogue & Transcript Explorer", expanded=False):
            col_spk_filter, col_spk_info = st.columns([2, 1])
            with col_spk_filter:
                spk_choice = st.selectbox("Filter dialogue by speaker:", ["All Speakers", "🗣️ Speaker 1 Only", "🗣️ Speaker 2 Only", "🗣️ Speaker 3 Only"])
            with col_spk_info:
                st.caption(f"Total {len(transcript_items)} transcribed dialogue cues.")

            filtered_transcript = transcript_items
            if spk_choice == "🗣️ Speaker 1 Only":
                filtered_transcript = [t for t in transcript_items if t.get("speaker") == "Speaker 1"]
            elif spk_choice == "🗣️ Speaker 2 Only":
                filtered_transcript = [t for t in transcript_items if t.get("speaker") == "Speaker 2"]
            elif spk_choice == "🗣️ Speaker 3 Only":
                filtered_transcript = [t for t in transcript_items if t.get("speaker") == "Speaker 3"]

            show_segments = filtered_transcript[:50]
            for t_idx, item in enumerate(show_segments):
                spk_color = "badge-cu" if item.get("speaker") == "Speaker 1" else ("badge-ms" if item.get("speaker") == "Speaker 2" else "badge-ws")
                st.markdown(f"""
                <div style="background-color: #161B22; border-left: 4px solid #00C7B7; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="scale-badge {spk_color}">{item.get('speaker', 'Speaker')}</span>
                        <span style="color: #00C7B7; font-family: monospace; font-size: 0.82rem;">{fmt_time(item['start'])} – {fmt_time(item['end'])}</span>
                    </div>
                    <div style="margin-top: 6px; font-size: 0.92rem; color: #E6E6E6;">{item.get('text', '')}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"⏩ Seek {fmt_time(item['start'])}", key=f"dlg_seek_{t_idx}"):
                    st.session_state.seek_to = int(item['start'])
                    st.rerun()

    st.divider()
    st.markdown("### 🟥 Narrative consistency")
    render_card("card-red", "Main character knowledge / goal check", "",
                results["narrative_result"], "narrative_continuity", teaching_mode,
                "jump_narrative")

    st.divider()
    st.subheader("📦 Export & NLE Integrations")
    st.markdown("Export timeline markers, edit decision lists, and subtitles directly to **DaVinci Resolve**, **Adobe Premiere Pro**, and **Final Cut Pro**.")

    tab_nle, tab_subs, tab_report = st.tabs(["🎞️ NLE Timelines & Markers", "💬 Subtitles & Captions", "📄 Editorial Report"])

    clean_id = re.sub(r'[^A-Za-z0-9_-]', '_', str(results.get("video_id") or "video_cut"))
    clip_title = f"Editorial_Review_{clean_id}"
    fps = results.get("metrics", {}).get("fps", 24.0) if results.get("metrics") else 24.0

    with tab_nle:
        col_edl, col_xml, col_csv = st.columns(3)

        with col_edl:
            st.markdown("#### CMX 3600 EDL")
            st.caption("Standard Edit Decision List for Premiere, Resolve, and Avid with shot and pacing markers.")
            edl_text = generate_edl(
                results["shots"], results["pacing_issues"], results["scenes"],
                clip_name=clip_title, fps=fps
            )
            st.download_button(
                "🎬 Download EDL (.edl)",
                data=edl_text,
                file_name=f"{clip_title}.edl",
                mime="text/plain",
                use_container_width=True
            )

        with col_xml:
            st.markdown("#### FCP XML (FCP7 / Premiere / Resolve)")
            st.caption("Full XML timeline sequence with color-coded scene and pacing markers.")
            fcpxml_text = generate_fcpxml(
                results["shots"], results["pacing_issues"], results["scenes"],
                total_duration=results["total_duration"], clip_name=clip_title, fps=fps
            )
            st.download_button(
                "🎞️ Download FCP XML (.xml)",
                data=fcpxml_text,
                file_name=f"{clip_title}.xml",
                mime="application/xml",
                use_container_width=True
            )

        with col_csv:
            st.markdown("#### DaVinci Resolve Marker CSV")
            st.caption("Color-coded timeline marker list ready to import directly into DaVinci Resolve.")
            resolve_csv = generate_resolve_marker_csv(
                results["shots"], results["pacing_issues"], results["scenes"], fps=fps
            )
            st.download_button(
                "📍 Download Marker CSV (.csv)",
                data=resolve_csv,
                file_name=f"{clip_title}_markers.csv",
                mime="text/csv",
                use_container_width=True
            )

    with tab_subs:
        col_srt, col_vtt = st.columns(2)
        transcript = results.get("transcript", [])
        with col_srt:
            st.markdown("#### SubRip Subtitles (.srt)")
            st.caption("Standard subtitle file with timecodes from Whisper transcription.")
            srt_content = generate_srt(transcript)
            st.download_button(
                "💬 Download Subtitles (.srt)",
                data=srt_content,
                file_name=f"{clip_title}.srt",
                mime="text/plain",
                use_container_width=True,
                disabled=len(transcript) == 0
            )

        with col_vtt:
            st.markdown("#### WebVTT Subtitles (.vtt)")
            st.caption("Web-compatible subtitle format for HTML5 video players and browsers.")
            vtt_content = generate_vtt(transcript)
            st.download_button(
                "🌐 Download WebVTT (.vtt)",
                data=vtt_content,
                file_name=f"{clip_title}.vtt",
                mime="text/vtt",
                use_container_width=True,
                disabled=len(transcript) == 0
            )

    with tab_report:
        st.markdown("#### Editorial Summary Report")
        st.caption("Text-based editorial briefing with scene timestamps, pacing alerts, and narrative review.")
        report_text = generate_text_report(
            results["url"], results["scenes"], results["pacing_issues"],
            results["narrative_result"], results["script_diff"],
        )
        st.download_button(
            "📄 Download Report (.txt)",
            data=report_text,
            file_name=f"{clip_title}_report.txt",
            mime="text/plain",
            use_container_width=True
        )

        st.divider()
        st.markdown("#### Full Project Review Package (.json)")
        st.caption("Export a complete, self-contained JSON package of this video review (shots, framing, audio metrics, dialogue).")
        st.download_button(
            "📦 Download Project Package (.json)",
            data=json.dumps(results, indent=2),
            file_name=f"{clip_title}_package.json",
            mime="application/json",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
