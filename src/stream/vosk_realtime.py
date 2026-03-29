"""
TsuurAI Real-time with Vosk
True streaming speech-to-text (works offline, CPU-friendly)
"""

import streamlit as st
import numpy as np
import tempfile
import time
import sys
import os
import json
import queue
import threading
from pathlib import Path

# Add parent directory to path for imports
SRC_DIR = Path(__file__).parent.parent.resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import LANGUAGE_CODES, API_LLM_INFO
from common.llm import openai_client, correct_with_llm
from common.auth import show_login_page, show_user_sidebar, log_usage

# Check for Vosk
try:
    from vosk import Model, KaldiRecognizer, SetLogLevel
    SetLogLevel(-1)  # Suppress Vosk logs
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

# Check for audio recorder
try:
    from audio_recorder_streamlit import audio_recorder
    RECORDER_AVAILABLE = True
except ImportError:
    RECORDER_AVAILABLE = False

st.set_page_config(page_title="TsuurAI - Vosk Real-time", page_icon="🎙️", layout="wide")

# Check authentication
if not st.session_state.get("authenticated"):
    show_login_page()
    st.stop()

st.title("🎙️ TsuurAI - Vosk Real-time")
st.caption("True streaming transcription | Offline | CPU-friendly")

if not VOSK_AVAILABLE:
    st.error("Vosk not installed. Run:")
    st.code("pip install vosk")
    st.stop()

# Show user info in sidebar
show_user_sidebar()

# Vosk model paths
MODELS_DIR = SRC_DIR / "models" / "vosk"
VOSK_MODELS = {
    "Mongolian": {
        "path": MODELS_DIR / "vosk-model-mn-0.4",
        "url": "https://alphacephei.com/vosk/models/vosk-model-mn-0.4.zip",
        "size": "45 MB"
    },
    "English (small)": {
        "path": MODELS_DIR / "vosk-model-small-en-us-0.15",
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        "size": "40 MB"
    },
    "English (large)": {
        "path": MODELS_DIR / "vosk-model-en-us-0.22",
        "url": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
        "size": "1.8 GB"
    }
}

# Sidebar configuration
with st.sidebar:
    st.header("Vosk Configuration")

    language = st.selectbox(
        "Language / Model",
        list(VOSK_MODELS.keys()),
        index=0
    )

    model_info = VOSK_MODELS[language]
    st.caption(f"Model size: {model_info['size']}")

    st.divider()
    st.header("Settings")

    show_partial = st.toggle("Show partial results", value=True, help="Show text while speaking")

    use_llm = st.toggle("LLM Post-correction", value=False)
    if use_llm:
        llm_model = st.selectbox("LLM", list(API_LLM_INFO.keys()), index=1)
        st.caption("Corrects final result only")
    else:
        llm_model = None

    st.divider()
    st.header("Vosk Advantages")
    st.markdown("""
    ✅ **True real-time** streaming
    ✅ **Offline** - no internet needed
    ✅ **CPU-friendly** - no GPU required
    ✅ **Low latency** - instant feedback
    """)

# Check/download model
model_path = model_info["path"]

def download_vosk_model(url, dest_path):
    """Download and extract Vosk model"""
    import urllib.request
    import zipfile

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest_path.parent / "model.zip"

    st.info(f"Downloading model from {url}...")
    progress = st.progress(0)

    def report_progress(block_num, block_size, total_size):
        progress.progress(min(block_num * block_size / total_size, 1.0))

    urllib.request.urlretrieve(url, zip_path, report_progress)

    st.info("Extracting model...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_path.parent)

    os.unlink(zip_path)
    st.success("Model ready!")

if not model_path.exists():
    st.warning(f"Vosk model not found: {model_path}")
    st.info(f"Model URL: {model_info['url']}")

    if st.button(f"Download {language} model ({model_info['size']})"):
        with st.spinner("Downloading..."):
            try:
                download_vosk_model(model_info["url"], model_path)
                st.rerun()
            except Exception as e:
                st.error(f"Download failed: {e}")
                st.info("Manual download:")
                st.code(f"""
# Download manually:
wget {model_info['url']}
unzip vosk-model-*.zip -d {MODELS_DIR}
""")
    st.stop()

# Load Vosk model
@st.cache_resource
def load_vosk_model(path):
    return Model(str(path))

with st.spinner(f"Loading Vosk model..."):
    vosk_model = load_vosk_model(model_path)
    st.success("Vosk ready!")

# Initialize session state
if "transcripts" not in st.session_state:
    st.session_state.transcripts = []
if "partial_result" not in st.session_state:
    st.session_state.partial_result = ""
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None

def transcribe_with_vosk(audio_bytes, show_partial_results=False):
    """Transcribe audio using Vosk streaming"""
    import wave
    import io

    try:
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Open audio file
        wf = wave.open(tmp_path, "rb")

        # Check format
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            st.warning("Converting audio format...")
            # Convert using pydub if needed
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(tmp_path)
                audio = audio.set_channels(1).set_sample_width(2).set_frame_rate(16000)
                audio.export(tmp_path, format="wav")
                wf.close()
                wf = wave.open(tmp_path, "rb")
            except:
                pass

        sample_rate = wf.getframerate()

        # Create recognizer
        rec = KaldiRecognizer(vosk_model, sample_rate)
        rec.SetWords(True)

        results = []
        partial = ""

        # Process audio in chunks (streaming simulation)
        while True:
            data = wf.readframes(4000)  # ~250ms chunks
            if len(data) == 0:
                break

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if result.get("text"):
                    results.append(result["text"])
            else:
                partial_result = json.loads(rec.PartialResult())
                partial = partial_result.get("partial", "")

        # Get final result
        final = json.loads(rec.FinalResult())
        if final.get("text"):
            results.append(final["text"])

        wf.close()
        os.unlink(tmp_path)

        full_text = " ".join(results).strip()
        return full_text, None

    except Exception as e:
        return None, str(e)

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Recording")

    if RECORDER_AVAILABLE:
        st.write("🎤 Click to start/stop recording")

        audio_bytes = audio_recorder(
            text="",
            recording_color="#e74c3c",
            neutral_color="#27ae60",
            icon_name="microphone",
            icon_size="3x",
            pause_threshold=1.5,
            sample_rate=16000
        )

        if audio_bytes:
            audio_hash = hash(audio_bytes)
            if audio_hash != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = audio_hash
                st.audio(audio_bytes, format="audio/wav")

                with st.spinner("Vosk transcribing..."):
                    start = time.time()
                    text, error = transcribe_with_vosk(audio_bytes, show_partial)
                    elapsed = time.time() - start

                if error:
                    st.error(f"Error: {error}")
                elif text:
                    # LLM correction if enabled
                    if use_llm and openai_client:
                        with st.spinner("LLM correcting..."):
                            lang_name = "Mongolian" if "Mongolian" in language else "English"
                            corrected, _ = correct_with_llm(text, lang_name, model_name=llm_model, temperature=0.2)
                            if corrected:
                                st.caption(f"Raw: {text}")
                                text = corrected

                    st.session_state.transcripts.append(text)
                    st.success(f"✓ ({elapsed:.2f}s) {text}")

                    log_usage(st.session_state.get("user_email"), "vosk_realtime", {
                        "language": language, "processing_time": elapsed
                    })
                    st.rerun()
                else:
                    st.warning("No speech detected")
    else:
        st.write("Record audio:")
        audio_value = st.audio_input("Click to record")

        if audio_value:
            audio_bytes = audio_value.getvalue()
            audio_hash = hash(audio_bytes)

            if audio_hash != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = audio_hash
                st.audio(audio_value)

                with st.spinner("Vosk transcribing..."):
                    start = time.time()
                    text, error = transcribe_with_vosk(audio_bytes)
                    elapsed = time.time() - start

                if text:
                    if use_llm and openai_client:
                        lang_name = "Mongolian" if "Mongolian" in language else "English"
                        corrected, _ = correct_with_llm(text, lang_name, model_name=llm_model, temperature=0.2)
                        if corrected:
                            text = corrected

                    st.session_state.transcripts.append(text)
                    st.success(f"✓ ({elapsed:.2f}s) {text}")
                    st.rerun()

with col2:
    st.subheader("Status")
    st.metric("Segments", len(st.session_state.transcripts))

    if st.session_state.transcripts:
        total_words = len(" ".join(st.session_state.transcripts).split())
        st.metric("Words", total_words)

    st.divider()
    st.caption("🟢 Vosk: Offline ready")

st.divider()

# Transcript display
st.subheader("Full Transcript")

if st.session_state.transcripts:
    full_text = " ".join(st.session_state.transcripts)
    st.text_area("Transcript", full_text, height=200)

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
            file_name="vosk_transcript.txt",
            use_container_width=True
        )

    with col3:
        if st.button("Copy", use_container_width=True):
            st.code(full_text)
else:
    st.info("Start recording to see transcription")

st.divider()

# Comparison
with st.expander("Vosk vs Whisper"):
    st.markdown("""
    | Feature | Vosk | Whisper |
    |---------|------|---------|
    | **Real-time** | ✅ Native streaming | ❌ Batch only |
    | **Offline** | ✅ Yes | ✅ Yes |
    | **CPU** | ✅ Fast | ⚠️ Slow |
    | **GPU** | Optional | Recommended |
    | **Accuracy** | Good | Excellent |
    | **Mongolian** | ✅ Yes | ✅ Yes |
    | **Latency** | ~100ms | ~1-3s |

    **Use Vosk when:** Real-time matters, no GPU, offline required

    **Use Whisper when:** Accuracy matters most, have GPU
    """)

st.caption("TsuurAI Vosk | Offline Real-time")
