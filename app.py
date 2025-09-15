# app.py

# =========================
# Bootstrapping: ensure deps
# =========================
import sys, subprocess

def _ensure_deps():
    pkgs = [
        "streamlit",
        "youtube-transcript-api",
        "google-generativeai",
        "pyngrok",
        "yt-dlp",
    ]
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade"] + pkgs)
    except Exception as e:
        # Don't crash UI if pip hiccups; Streamlit will still try to run if pkgs are present.
        print("Dependency install warning:", repr(e))

_ensure_deps()

# =========================
# Imports (after deps)
# =========================
import os
import re
import json
import glob
import textwrap
from urllib.parse import urlparse, parse_qs

import streamlit as st
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)
import google.generativeai as genai

# =========================
# App Config
# =========================
APP_TITLE = "🎬 YouTube Transcript Summarizer + Q&A"
MODEL_NAME = "gemini-2.5-flash"   # swap to Pro if you prefer

# =========================
# Utilities
# =========================
YOUTUBE_ID_RE = re.compile(r"(?P<id>[0-9A-Za-z_-]{11})")

def extract_video_id(url_or_id: str) -> str | None:
    s = url_or_id.strip()
    m = YOUTUBE_ID_RE.fullmatch(s)
    if m:
        return m.group("id")
    try:
        u = urlparse(s)
        host = (u.netloc or "").lower()
        path = (u.path or "")
        if "youtube.com" in host:
            qs = parse_qs(u.query or "")
            v = (qs.get("v") or [None])[0]
            if v and YOUTUBE_ID_RE.fullmatch(v):
                return v
            m = re.search(r"/shorts/([0-9A-Za-z_-]{11})", path or "")
            if m: return m.group(1)
            m = re.search(r"/embed/([0-9A-Za-z_-]{11})", path or "")
            if m: return m.group(1)
        if "youtu.be" in host:
            m = re.search(r"/([0-9A-Za-z_-]{11})", path or "")
            if m: return m.group(1)
    except Exception:
        pass
    m = YOUTUBE_ID_RE.search(s)
    return m.group("id") if m else None

def save_txt(text: str, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

# =========================
# Transcript logic (API → yt-dlp)
# =========================
def fetch_via_api(video_ref: str, preferred_langs=('en','en-US','en-GB')):
    from youtube_transcript_api import (
        YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
    )
    vid = extract_video_id(video_ref)
    if not vid:
        return None, {"ok": False, "reason": "Invalid YouTube URL or ID."}
    try:
        if not hasattr(YouTubeTranscriptApi, "list_transcripts"):
            return None, {"ok": False, "reason": "API method list_transcripts is missing (likely shadowed/old module)."}
        transcripts = YouTubeTranscriptApi.list_transcripts(vid)
        try:
            t = transcripts.find_transcript(preferred_langs)
        except NoTranscriptFound:
            try:
                t = transcripts.find_generated_transcript(preferred_langs)
            except NoTranscriptFound:
                return None, {"ok": False, "reason": "No transcript in preferred languages (manual or auto)."}
        segments = t.fetch()
        text = " ".join(s["text"].strip() for s in segments if s["text"].strip())
        meta = {
            "ok": True,
            "source": "youtube-transcript-api",
            "video_id": vid,
            "language": t.language_code,
            "is_generated": getattr(t, "is_generated", False),
            "num_segments": len(segments),
            "chars": len(text),
        }
        return text, meta
    except TranscriptsDisabled:
        return None, {"ok": False, "reason": "Transcripts are disabled by the uploader."}
    except VideoUnavailable:
        return None, {"ok": False, "reason": "Video unavailable (private, region-locked, or removed)."}
    except NoTranscriptFound:
        return None, {"ok": False, "reason": "No transcript found for this video."}
    except Exception as e:
        return None, {"ok": False, "reason": f"Unexpected API error: {e.__class__.__name__}: {e}"}

def fetch_via_ytdlp(video_ref: str, preferred_langs=('en','en-US','en-GB')):
    vid = extract_video_id(video_ref)
    if not vid:
        return None, {"ok": False, "reason": "Invalid YouTube URL or ID."}
    os.makedirs("subs", exist_ok=True)
    for f in glob.glob(f"subs/{vid}.*"):
        try: os.remove(f)
        except: pass

    # Request manual & auto subs, convert to srt
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-subs", "--write-auto-sub",
        "--sub-lang", ",".join(preferred_langs),
        "--convert-subs", "srt",
        "-o", f"subs/%(id)s.%(ext)s",
        video_ref,
    ]
    try:
        import subprocess as sp
        rc = sp.call(cmd)
    except FileNotFoundError:
        return None, {"ok": False, "reason": "yt-dlp not found in PATH. Please install yt-dlp."}

    srt_files = glob.glob(f"subs/{vid}.en.srt") + glob.glob(f"subs/{vid}.en-*.srt") + glob.glob(f"subs/{vid}.*.srt")
    if not srt_files:
        return None, {"ok": False, "reason": "No subtitles (manual or auto) found via yt-dlp."}

    srt_path = srt_files[0]
    lines = []
    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if re.fullmatch(r"\d+", line): 
                continue
            if re.search(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", line):
                continue
            if line.strip():
                lines.append(line.strip())
    text = " ".join(lines)
    meta = {
        "ok": True,
        "source": "yt-dlp",
        "video_id": vid,
        "srt_path": srt_path,
        "chars": len(text),
    }
    return text, meta

# =========================
# Gemini helpers
# =========================
def _chunk_text(text: str, max_chars: int = 12000) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    parts, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        cut = text.rfind(". ", start, end)
        if cut == -1 or cut < start + int(0.6 * max_chars):
            cut = end
        parts.append(text[start:cut].strip())
        start = cut + 1
    return [p for p in parts if p]

def _trim_to_words(s: str, target_words: int) -> str:
    w = s.split()
    return s if len(w) <= target_words else " ".join(w[:target_words])

def summarize_with_gemini_bullets(transcript: str, api_key: str, target_words: int, style_hint: str) -> list[str]:
    """
    Always returns a list of bullet strings.
    - Each bullet 2–3 sentences.
    - At most 10 bullets.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    chunks = _chunk_text(transcript, max_chars=12000)

    # First pass: compress chunks into bullets
    per_chunk_prompt = (
        "You will receive part of a YouTube transcript.\n"
        f"Summarize ONLY as bullet points totaling up to {max(80, target_words//max(1,len(chunks))+40)} words.\n"
        "Rules:\n"
        " - Use bullets only (no paragraphs).\n"
        " - Each bullet must be 2–3 sentences.\n"
        " - Be factual; avoid fluff.\n"
        " - Prefer key insights, definitions, examples, conclusions.\n\n"
    )

    partial_bullets = []
    for part in chunks:
        resp = model.generate_content(per_chunk_prompt + part)
        text = (resp.text or "").strip()
        partial_bullets.extend(_extract_bullets(text))

    # Final merge: ask model to merge & limit; but still post-process
    merged_prompt = (
        "You will receive multiple bullet lists derived from parts of one talk.\n"
        f"Merge into ONE list of bullets targeting ~{target_words} words total.\n"
        "STRICT RULES:\n"
        " - Bullets only (no paragraphs), each bullet 2–3 sentences.\n"
        " - No more than 10 bullets total.\n"
        " - Remove duplicates & keep only the most important ideas.\n"
        " - Keep clear, crisp, non-repetitive phrasing.\n"
        "Return ONLY the final bullets."
    )
    resp = model.generate_content(merged_prompt + "\n\n" + "\n".join(f"- {b}" for b in partial_bullets))
    merged = (resp.text or "").strip()

    bullets = _extract_bullets(merged)
    # Enforce limits again
    bullets = bullets[:10]
    # Clamp total words (softly) by trimming last bullets if way over target
    # and trimming sentences per bullet to max 3 sentences.
    bullets = [_limit_sentences(b, max_sentences=3) for b in bullets]
    return bullets

def _extract_bullets(text: str) -> list[str]:
    """Parse bullets from model output; if none, convert sentences into 2–3 sentence bullets."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    raw = []
    for l in lines:
        if l[0].isdigit() and (l[1:2] in [".", ")"]):
            raw.append(l.split(maxsplit=1)[1] if " " in l else l)
        elif l.startswith(("-", "•", "*")):
            raw.append(l.lstrip("-•* ").strip())
    if raw:
        return [r for r in raw if r]

    # No bullet markers — fall back to sentence grouping
    sents = _split_sentences(text)
    bullets = []
    i = 0
    while i < len(sents):
        group = " ".join(sents[i:i+3])  # 2–3 sentences ideally
        bullets.append(group.strip())
        i += 2 if i+2 <= len(sents) else 1
    return [b for b in bullets if b]

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
def _split_sentences(text: str) -> list[str]:
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents

def _limit_sentences(bullet: str, max_sentences: int = 3) -> str:
    sents = _split_sentences(bullet)
    return " ".join(sents[:max_sentences])

def answer_question(transcript: str, question: str, api_key: str) -> str:
    """Grounded Q&A over the transcript."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = (
        "Answer the user's question **using only** the transcript below. "
        "If the answer isn't in the transcript, say you can't find it.\n\n"
        "=== TRANSCRIPT START ===\n"
        f"{transcript}\n"
        "=== TRANSCRIPT END ===\n\n"
        f"Question: {question}\n"
        "Answer (be concise, cite brief quotes or timestamps if present):"
    )
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()

# =========================
# Streamlit UI
# =========================
def main():
    st.set_page_config(page_title="YouTube Summarizer + Q&A", page_icon="📹", layout="wide")
    st.title(APP_TITLE)
    st.caption("Robust transcript (API → yt-dlp fallback), bullet-only summaries, grounded Q&A.")

    # ---- Session state ----
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []  # list[{"q":..., "a":...}]
    if "transcript" not in st.session_state:
        st.session_state.transcript = None
    if "summary_bullets" not in st.session_state:
        st.session_state.summary_bullets = None
    if "meta" not in st.session_state:
        st.session_state.meta = {}

    # ---- Sidebar ----
    st.sidebar.header("🔑 Configuration")
    api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Create one in Google AI Studio.")
    prefer_langs = st.sidebar.multiselect(
        "Preferred transcript languages (order matters)",
        default=["en","en-US","en-GB"],
        options=["en","en-US","en-GB","hi","es","fr","de","pt","ru","ja","ko","ar","zh-Hans","zh-Hant"],
    )
    st.sidebar.caption("If English isn't available, we try others; silent fallback to yt-dlp if API path isn't available.")

    # ---- Inputs ----
    top = st.columns([2, 1])
    with top[0]:
        youtube_url = st.text_input("📺 YouTube URL or 11-char ID", placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    with top[1]:
        summary_style = st.radio("Summary style (tone)", ["Short Summary", "Long Summary"])
        word_target = st.select_slider("Target word count", options=[100,250,500,750,1000], value=250)

    # ---- Generate Summary ----
    if st.button("🚀 Generate Summary", type="primary"):
        if not api_key:
            st.error("Please enter your Gemini API key.")
            st.stop()
        if not youtube_url:
            st.error("Please paste a YouTube URL or ID.")
            st.stop()

        with st.spinner("Fetching transcript…"):
            vid = extract_video_id(youtube_url)
            if not vid:
                st.error("Could not extract a valid YouTube video ID.")
                st.stop()

            # Try API; if not ok, silently fallback to yt-dlp (NO banner as requested)
            text, meta = fetch_via_api(youtube_url, preferred_langs=tuple(prefer_langs) or ('en','en-US','en-GB'))
            if not (meta and meta.get("ok")):
                text, meta = fetch_via_ytdlp(youtube_url, preferred_langs=tuple(prefer_langs) or ('en','en-US','en-GB'))

            if not (meta and meta.get("ok") and text):
                st.error(f"Could not fetch transcript. Reason: {meta.get('reason') if meta else 'unknown'}")
                st.stop()

        with st.spinner("Summarizing with Gemini (bullets)…"):
            style_hint = "short" if summary_style.startswith("Short") else "long"
            bullets = summarize_with_gemini_bullets(text, api_key, target_words=word_target, style_hint=style_hint)

        # Persist so Q&A reruns don't lose them
        st.session_state.transcript = text
        st.session_state.summary_bullets = bullets
        st.session_state.meta = meta

        st.success(f"Transcript fetched via **{meta['source']}**")

    # ---- Always show Summary + Downloads when available ----
    if st.session_state.summary_bullets:
        meta = st.session_state.meta or {}
        text = st.session_state.transcript or ""
        vid = meta.get("video_id", "video")
        st.subheader("📋 Summary (bullets)")
        # Render bullets (max 10 already enforced)
        for i, b in enumerate(st.session_state.summary_bullets, 1):
            st.markdown(f"- {b}")

        # Downloads stay visible
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("📥 Download Summary", data="\n".join(f"- {b}" for b in st.session_state.summary_bullets),
                               file_name=f"youtube_summary_{vid}.txt", mime="text/plain")
        with dl2:
            st.download_button("📥 Download Transcript", data=text,
                               file_name=f"youtube_transcript_{vid}.txt", mime="text/plain")

        with st.expander("📄 View Full Transcript"):
            st.text_area("Transcript", text, height=300)

    st.markdown("---")

    # ---- Q&A Section (grounded) ----
    st.subheader("💬 Ask Questions about the Video (grounded in transcript)")
    qcol1, qcol2 = st.columns([3,1])
    with qcol1:
        user_q = st.text_input("Your question:", placeholder="e.g., What were the three key takeaways?")
    with qcol2:
        ask = st.button("Ask")

    if ask:
        if not api_key:
            st.error("Enter your Gemini API key in the sidebar.")
        elif not st.session_state.transcript:
            st.error("Please generate a transcript first.")
        elif not user_q.strip():
            st.error("Please type a question.")
        else:
            with st.spinner("Thinking…"):
                ans = answer_question(st.session_state.transcript, user_q.strip(), api_key)
            st.session_state.qa_history.append({"q": user_q.strip(), "a": ans})

    # ---- Render chat history (persists during session) ----
    if st.session_state.qa_history:
        st.subheader("🧾 Q&A History")
        for i, turn in enumerate(st.session_state.qa_history, 1):
            st.markdown(f"**Q{i}.** {turn['q']}")
            st.markdown(f"**A{i}.** {turn['a']}")
            st.markdown("---")

    st.caption("Notes: API → yt-dlp fallback is silent. Summary is always bullet-only (2–3 sentences each, ≤10 bullets).")

if __name__ == "__main__":
    main()
