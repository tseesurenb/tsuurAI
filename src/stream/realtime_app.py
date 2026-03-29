"""
TsuurAI Real-time Streaming App
True real-time speech-to-text using WebRTC
"""

import streamlit as st
import numpy as np
import queue
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
SRC_DIR = Path(__file__).parent.parent.resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import LANGUAGE_CODES, MODEL_INFO, API_LLM_INFO
from common.models import check_model_exists, load_whisper_model
from common.llm import openai_client, correct_with_llm
from common.auth import show_login_page, show_user_sidebar, log_usage

# Check for streamlit-webrtc
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

st.set_page_config(page_title="TsuurAI - Real-time", page_icon="🎙️", layout="wide")

# Check authentication
if not st.session_state.get("authenticated"):
    show_login_page()
    st.stop()

st.title("🎙️ TsuurAI - Real-time Transcription")
st.write("Continuous speech-to-text as you speak")

# Show user info in sidebar
show_user_sidebar()

if not WEBRTC_AVAILABLE:
    st.error("streamlit-webrtc not installed. Run:")
    st.code("pip install streamlit-webrtc av")
    st.stop()

# Sidebar configuration
with st.sidebar:
    st.header("Model Configuration")

    # For real-time, only use fast Whisper models
    model_size = st.selectbox(
        "Whisper Model",
        ["tiny", "base", "small"],
        index=1,
        help="Smaller = faster, better for real-time"
    )

    language = st.selectbox(
        "Language",
        ["English", "Mongolian"],
        index=1
    )

    st.divider()

    st.header("Real-time Settings")

    vad_threshold = st.slider(
        "VAD Sensitivity",
        min_value=0.01,
        max_value=0.1,
        value=0.03,
        help="Lower = more sensitive to speech"
    )

    min_speech_duration = st.slider(
        "Min speech duration (sec)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        help="Minimum audio length before processing"
    )

    use_llm = st.toggle("LLM Correction", value=False)
    if use_llm:
        llm_model = st.selectbox("LLM", list(API_LLM_INFO.keys()), index=1)
        st.caption("Adds ~0.5s latency")
    else:
        llm_model = None

    st.divider()

    info = MODEL_INFO["whisper"][model_size]
    st.metric("Speed", info["speed"])
    st.metric("Quality", info["quality"])

# Load Whisper model
model_exists = check_model_exists("whisper", model_size)
loading_msg = f"Loading Whisper {model_size}..." if model_exists else f"Downloading Whisper {model_size}..."

with st.spinner(loading_msg):
    whisper_model = load_whisper_model(model_size)
    st.success(f"Whisper '{model_size}' ready for real-time!")

# Initialize session state
if "realtime_transcript" not in st.session_state:
    st.session_state.realtime_transcript = []
if "audio_buffer" not in st.session_state:
    st.session_state.audio_buffer = []

# Audio processor class for WebRTC
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert to numpy array
        audio = frame.to_ndarray()

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=0)

        # Resample to 16kHz if needed
        if frame.sample_rate != self.sample_rate:
            # Simple resampling (for production, use librosa)
            ratio = self.sample_rate / frame.sample_rate
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length).astype(int)
            audio = audio[indices]

        # Normalize
        audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0

        # Put in queue for processing
        self.audio_queue.put(audio)

        return frame

# Create columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Transcription")

    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    )

    # Transcription display
    transcript_container = st.container()

    with transcript_container:
        if st.session_state.realtime_transcript:
            full_text = " ".join(st.session_state.realtime_transcript)
            st.text_area(
                "Transcript",
                full_text,
                height=300,
                key="live_transcript_display"
            )
        else:
            st.info("Click START above to begin real-time transcription")

with col2:
    st.subheader("Controls")

    if st.button("Clear Transcript", use_container_width=True):
        st.session_state.realtime_transcript = []
        st.rerun()

    if st.session_state.realtime_transcript:
        full_text = " ".join(st.session_state.realtime_transcript)
        st.download_button(
            "Download",
            full_text,
            file_name="realtime_transcript.txt",
            use_container_width=True
        )

        word_count = len(full_text.split())
        st.metric("Words", word_count)

    st.divider()
    st.subheader("Status")

    if webrtc_ctx.state.playing:
        st.success("🔴 Recording...")
    else:
        st.info("⏸️ Stopped")

# Process audio in real-time
if webrtc_ctx.state.playing and webrtc_ctx.audio_receiver:
    st.info("Processing audio stream...")

    audio_buffer = []
    buffer_duration = 0
    sample_rate = 16000

    status_placeholder = st.empty()
    result_placeholder = st.empty()

    while webrtc_ctx.state.playing:
        try:
            # Get audio frames
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=0.1)

            for frame in audio_frames:
                # Convert to numpy
                audio = frame.to_ndarray()
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=0)

                audio = audio.astype(np.float32)
                if audio.max() > 1.0:
                    audio = audio / 32768.0

                audio_buffer.extend(audio.tolist())
                buffer_duration = len(audio_buffer) / sample_rate

            # Check if we have enough audio
            if buffer_duration >= min_speech_duration:
                audio_np = np.array(audio_buffer, dtype=np.float32)

                # Simple VAD - check energy
                energy = (audio_np ** 2).mean()

                if energy > vad_threshold:
                    status_placeholder.info(f"Processing {buffer_duration:.1f}s of audio...")

                    # Transcribe
                    import tempfile
                    import soundfile as sf

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        sf.write(tmp.name, audio_np, sample_rate)

                        lang_code = LANGUAGE_CODES[language]["whisper"]
                        segments, _ = whisper_model.transcribe(
                            tmp.name,
                            language=lang_code,
                            beam_size=1,  # Fast
                            temperature=0.0,
                            vad_filter=True
                        )

                        text = " ".join([s.text for s in segments]).strip()

                        if text:
                            # LLM correction if enabled
                            if use_llm and openai_client:
                                corrected, _ = correct_with_llm(
                                    text, language, model_name=llm_model, temperature=0.2
                                )
                                if corrected:
                                    text = corrected

                            st.session_state.realtime_transcript.append(text)
                            result_placeholder.success(f"✓ {text}")

                            log_usage(st.session_state.get("user_email"), "realtime_transcription", {
                                "model": model_size,
                                "language": language,
                                "duration": buffer_duration
                            })
                else:
                    status_placeholder.caption("Waiting for speech...")

                # Clear buffer
                audio_buffer = []
                buffer_duration = 0

        except queue.Empty:
            continue
        except Exception as e:
            st.error(f"Error: {e}")
            break

        time.sleep(0.05)

st.divider()
st.caption("TsuurAI Real-time Mode - Powered by Whisper + WebRTC")
