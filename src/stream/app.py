"""
TsuurAI Streaming App
Real-time speech-to-text transcription
"""

import streamlit as st
import tempfile
import os
import sys
import time
import queue
import threading

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from common.config import (
    LANGUAGE_CODES, MODEL_INFO, LOCAL_LLM_INFO, API_LLM_INFO
)
from common.models import (
    check_model_exists, load_whisper_model, load_mms_model, load_local_llm
)
from common.llm import (
    openai_client, correct_with_llm, refine_with_llm
)
from common.auth import show_login_page, show_user_sidebar, log_usage

st.set_page_config(page_title="TsuurAI - Stream", page_icon="🎙️", layout="wide")

# Check authentication
if not st.session_state.get("authenticated"):
    show_login_page()
    st.stop()

st.title("🎙️ TsuurAI - Real-time Transcription")
st.write("Live speech-to-text as you speak")

# Show user info in sidebar
show_user_sidebar()

# Sidebar for model configuration
with st.sidebar:
    st.header("Model Configuration")

    # For streaming, prefer faster models
    model_family = st.selectbox(
        "AI Model Family",
        ["Whisper (OpenAI)", "MMS (Meta)"],
        index=0  # Default: Whisper for streaming
    )

    if model_family == "Whisper (OpenAI)":
        model_key = "whisper"
        model_size = st.selectbox(
            "Model Size",
            ["tiny", "base", "small", "medium"],
            index=1,  # Default: base for streaming
            help="Smaller = faster, better for real-time"
        )
    else:
        model_key = "meta_mms"
        model_size = st.selectbox(
            "Model Size",
            ["mms-1b-all"],
            index=0
        )

    # Language selection
    language = st.selectbox(
        "Language",
        ["English", "Mongolian"],
        index=1
    )

    st.divider()

    # Streaming settings
    st.header("Streaming Settings")

    chunk_duration = st.slider(
        "Chunk duration (seconds)",
        min_value=2,
        max_value=10,
        value=5,
        help="Process audio every N seconds"
    )

    use_llm_correction = st.toggle("Enable LLM correction", value=True)

    if use_llm_correction:
        llm_model = st.selectbox(
            "LLM Model",
            list(API_LLM_INFO.keys()),
            index=1  # gpt-4o-mini for speed
        )
        st.caption(f"Using {llm_model} for real-time correction")

        # Only single-pass for streaming (speed)
        use_two_pass = st.toggle("Two-pass correction", value=False)
        if use_two_pass:
            st.warning("Two-pass adds latency")
    else:
        llm_model = None
        use_two_pass = False

    st.divider()

    # Model info
    st.header("Model Information")
    info = MODEL_INFO[model_key][model_size]
    st.metric("Speed", info["speed"])
    st.metric("Quality", info["quality"])

# Initialize session state
if "transcription_history" not in st.session_state:
    st.session_state.transcription_history = []
if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False

# Load ASR model
model_exists = check_model_exists(model_key, model_size)
loading_msg = f"Loading {model_size}..." if model_exists else f"Downloading {model_size}..."

with st.spinner(loading_msg):
    if model_key == "whisper":
        asr_model = load_whisper_model(model_size)
        st.success(f"Whisper '{model_size}' ready for streaming!")
    else:
        mms_processor, mms_model = load_mms_model()
        st.success("MMS ready for streaming!")

def transcribe_chunk(audio_data):
    """Transcribe a single audio chunk"""
    import torch

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        if model_key == "whisper":
            lang_code = LANGUAGE_CODES[language]["whisper"]
            segments, _ = asr_model.transcribe(
                tmp_path,
                language=lang_code,
                beam_size=3,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            text = " ".join([s.text for s in segments]).strip()
        else:
            import librosa
            audio, sr = librosa.load(tmp_path, sr=16000)

            lang_code = LANGUAGE_CODES[language]["mms"]
            mms_processor.tokenizer.set_target_lang(lang_code)
            mms_model.load_adapter(lang_code)

            inputs = mms_processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = mms_model(**inputs).logits

            ids = torch.argmax(outputs, dim=-1)[0]
            text = mms_processor.decode(ids)

        # Apply LLM correction if enabled
        if use_llm_correction and text and openai_client:
            corrected, error = correct_with_llm(
                text, language, model_name=llm_model, temperature=0.2
            )
            if not error:
                if use_two_pass:
                    refined, _ = refine_with_llm(
                        corrected, language, model_name=llm_model, temperature=0.2
                    )
                    text = refined
                else:
                    text = corrected

        return text

    except Exception as e:
        return f"[Error: {str(e)}]"

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Main content
st.markdown(f"**Settings:** {model_family} ({model_size}) | Language: {language} | Chunk: {chunk_duration}s")

# Real-time transcription display
st.subheader("Live Transcription")

# Transcription output area
transcription_container = st.container()

with transcription_container:
    if st.session_state.transcription_history:
        full_text = " ".join(st.session_state.transcription_history)
        st.text_area(
            "Transcription",
            full_text,
            height=200,
            key="live_transcription"
        )
    else:
        st.info("Start recording to see live transcription")

# Controls
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Clear History", use_container_width=True):
        st.session_state.transcription_history = []
        st.rerun()

with col2:
    if st.session_state.transcription_history:
        full_text = " ".join(st.session_state.transcription_history)
        st.download_button(
            "Download Text",
            full_text,
            file_name="transcription.txt",
            use_container_width=True
        )

with col3:
    if st.session_state.transcription_history:
        word_count = len(" ".join(st.session_state.transcription_history).split())
        st.metric("Words", word_count)

st.divider()

# Chunked recording mode
st.subheader("Record Audio Chunks")
st.write(f"Record {chunk_duration}-second chunks for near-real-time transcription")

audio_value = st.audio_input("Record a chunk", key="stream_recorder")

if audio_value:
    audio_bytes = audio_value.getvalue()
    st.audio(audio_value)

    if st.button("Transcribe Chunk", type="primary"):
        with st.spinner("Processing..."):
            start_time = time.time()
            result = transcribe_chunk(audio_bytes)
            elapsed = time.time() - start_time

        if result and not result.startswith("[Error"):
            st.session_state.transcription_history.append(result)
            st.success(f"Processed in {elapsed:.2f}s: {result}")

            log_usage(st.session_state.get("user_email"), "stream_transcription", {
                "model": model_key,
                "language": language,
                "chunk_duration": chunk_duration,
                "processing_time": elapsed
            })

            st.rerun()
        else:
            st.error(result)

st.divider()

# Continuous mode instructions
with st.expander("Continuous Recording Mode (Coming Soon)"):
    st.markdown("""
    ### WebSocket Streaming

    For true real-time streaming, we need:

    1. **WebSocket Server** - Continuous audio stream
    2. **VAD (Voice Activity Detection)** - Detect speech segments
    3. **Incremental Processing** - Process while recording

    Current implementation uses **chunked mode**:
    - Record 5-second chunks
    - Process each chunk immediately
    - Results appear in ~1-2 seconds

    **To enable WebSocket streaming:**
    ```bash
    pip install websockets sounddevice
    python -m stream.websocket_server
    ```
    """)

st.divider()
st.caption("TsuurAI Stream Mode - Real-time transcription")
