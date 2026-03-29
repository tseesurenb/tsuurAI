"""
TsuurAI Real-time Streaming App
Continuous speech-to-text transcription
"""

import streamlit as st
import numpy as np
import tempfile
import time
import sys
import os
from pathlib import Path
import io

# Add parent directory to path for imports
SRC_DIR = Path(__file__).parent.parent.resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import LANGUAGE_CODES, MODEL_INFO, API_LLM_INFO
from common.models import load_whisper_model
from common.llm import openai_client, correct_with_llm
from common.auth import show_login_page, show_user_sidebar, log_usage

# Try to import audio recorder
try:
    from audio_recorder_streamlit import audio_recorder
    RECORDER_AVAILABLE = True
except ImportError:
    RECORDER_AVAILABLE = False

st.set_page_config(page_title="TsuurAI - Real-time", page_icon="🎙️", layout="wide")

# Check authentication
if not st.session_state.get("authenticated"):
    show_login_page()
    st.stop()

st.title("🎙️ TsuurAI - Real-time Transcription")

if not RECORDER_AVAILABLE:
    st.warning("For best experience, install audio-recorder-streamlit:")
    st.code("pip install audio-recorder-streamlit")
    st.info("Falling back to standard audio input mode")

# Show user info in sidebar
show_user_sidebar()

# Sidebar configuration
with st.sidebar:
    st.header("Model Configuration")

    model_size = st.selectbox(
        "Whisper Model",
        ["tiny", "base", "small"],
        index=0,
        help="tiny = fastest (~1s), base = balanced, small = best quality"
    )

    language = st.selectbox(
        "Language",
        ["English", "Mongolian"],
        index=1
    )

    st.divider()
    st.header("Settings")

    auto_transcribe = st.toggle("Auto-transcribe", value=True, help="Automatically transcribe when recording stops")

    use_llm = st.toggle("LLM Correction", value=False)
    if use_llm:
        llm_model = st.selectbox("LLM", list(API_LLM_INFO.keys()), index=1)
    else:
        llm_model = None

    st.divider()
    info = MODEL_INFO["whisper"][model_size]
    st.metric("Speed", info["speed"])
    st.metric("Quality", info["quality"])

# Load model
@st.cache_resource
def get_whisper(size):
    return load_whisper_model(size)

with st.spinner(f"Loading Whisper {model_size}..."):
    whisper_model = get_whisper(model_size)
    st.success(f"Whisper {model_size} ready!")

# Initialize session state
if "transcripts" not in st.session_state:
    st.session_state.transcripts = []
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None

def transcribe_audio(audio_bytes):
    """Transcribe audio bytes"""
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Transcribe
        lang_code = LANGUAGE_CODES[language]["whisper"]
        segments, _ = whisper_model.transcribe(
            tmp_path,
            language=lang_code,
            beam_size=1,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300)
        )

        text = " ".join([s.text for s in segments]).strip()

        # Clean up temp file
        os.unlink(tmp_path)

        # Check for repetition
        if text and len(text) > 20:
            for plen in range(3, 15):
                pattern = text[:plen]
                if text.count(pattern) > 4:
                    text = pattern.strip()
                    break

        # LLM correction
        if use_llm and openai_client and text:
            with st.spinner("LLM correcting..."):
                corrected, _ = correct_with_llm(text, language, model_name=llm_model, temperature=0.2)
                if corrected:
                    text = corrected

        return text, None

    except Exception as e:
        return None, str(e)

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Recording")

    if RECORDER_AVAILABLE:
        # Use audio recorder component - allows continuous recording
        st.write("🎤 Click to start/stop recording")

        audio_bytes = audio_recorder(
            text="",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_name="microphone",
            icon_size="3x",
            pause_threshold=2.0,  # Auto-stop after 2s silence
            sample_rate=16000
        )

        if audio_bytes:
            # Check if this is new audio
            audio_hash = hash(audio_bytes)
            if audio_hash != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = audio_hash
                st.audio(audio_bytes, format="audio/wav")

                if auto_transcribe:
                    with st.spinner("Transcribing..."):
                        start = time.time()
                        text, error = transcribe_audio(audio_bytes)
                        elapsed = time.time() - start

                    if error:
                        st.error(f"Error: {error}")
                    elif text:
                        st.session_state.transcripts.append(text)
                        st.success(f"✓ ({elapsed:.1f}s) {text}")
                        log_usage(st.session_state.get("user_email"), "realtime", {
                            "model": model_size, "language": language
                        })
                        st.rerun()
                    else:
                        st.warning("No speech detected")
                else:
                    if st.button("Transcribe", type="primary"):
                        with st.spinner("Transcribing..."):
                            text, error = transcribe_audio(audio_bytes)
                        if error:
                            st.error(f"Error: {error}")
                        elif text:
                            st.session_state.transcripts.append(text)
                            st.success(f"✓ {text}")
                            st.rerun()
    else:
        # Fallback to standard audio input
        st.write("Record audio and it will be transcribed automatically")

        audio_value = st.audio_input("Click to record", key="recorder")

        if audio_value:
            audio_bytes = audio_value.getvalue()
            audio_hash = hash(audio_bytes)

            if audio_hash != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = audio_hash
                st.audio(audio_value)

                if auto_transcribe:
                    with st.spinner("Transcribing..."):
                        start = time.time()
                        text, error = transcribe_audio(audio_bytes)
                        elapsed = time.time() - start

                    if error:
                        st.error(f"Error: {error}")
                    elif text:
                        st.session_state.transcripts.append(text)
                        st.success(f"✓ ({elapsed:.1f}s) {text}")
                        st.rerun()
                else:
                    if st.button("Transcribe", type="primary"):
                        with st.spinner("Transcribing..."):
                            text, error = transcribe_audio(audio_bytes)
                        if text:
                            st.session_state.transcripts.append(text)
                            st.rerun()

with col2:
    st.subheader("Status")
    st.metric("Segments", len(st.session_state.transcripts))

    if st.session_state.transcripts:
        total_words = len(" ".join(st.session_state.transcripts).split())
        st.metric("Words", total_words)

st.divider()

# Transcript display
st.subheader("Full Transcript")

if st.session_state.transcripts:
    full_text = " ".join(st.session_state.transcripts)
    st.text_area("Transcript", full_text, height=200, key="full_transcript")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Clear All", use_container_width=True):
            st.session_state.transcripts = []
            st.session_state.last_audio_hash = None
            st.rerun()

    with col2:
        st.download_button(
            "Download",
            full_text,
            file_name="transcript.txt",
            use_container_width=True
        )

    with col3:
        if st.button("Copy (show)", use_container_width=True):
            st.code(full_text)
else:
    st.info("Start recording to see transcription here")

st.divider()

# Tips
with st.expander("Tips for best results"):
    st.markdown("""
    **For fastest transcription:**
    - Use **Whisper tiny** model (~1 second processing)
    - Disable LLM correction
    - Speak clearly and close to microphone

    **For best accuracy:**
    - Use **Whisper small** model
    - Enable LLM correction
    - Record in quiet environment

    **Recording tips:**
    - Wait for red indicator before speaking
    - Pause briefly between sentences
    - Auto-stop triggers after 2 seconds of silence
    """)

st.caption("TsuurAI Real-time | Whisper")
