import streamlit as st
import tempfile
import os
from pathlib import Path

st.set_page_config(
    page_title="Mongolian Speech-to-Text",
    page_icon="🎤",
    layout="centered"
)

st.title("🎤 Mongolian Speech-to-Text")
st.markdown("Powered by OpenAI Whisper")

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None

# Sidebar
st.sidebar.header("Model Settings")

model_size = st.sidebar.selectbox(
    "Model Size",
    options=["tiny", "base", "small", "medium", "large"],
    index=2,  # default to "small"
    help="Larger models are more accurate but slower"
)

# Load model button
if st.sidebar.button("Load Model", type="primary"):
    with st.spinner(f"Loading Whisper {model_size}... (first time downloads the model)"):
        try:
            import whisper
            st.session_state.model = whisper.load_model(model_size)
            st.session_state.model_size = model_size
            st.sidebar.success(f"Model '{model_size}' loaded!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")

# Show model status
if st.session_state.model:
    st.sidebar.success(f"✅ Model ready ({st.session_state.model_size})")
else:
    st.sidebar.warning("⚠️ Please load the model first")

st.markdown("---")

# Audio input options
tab1, tab2 = st.tabs(["📁 Upload Audio", "🎙️ Record Audio"])

audio_file_path = None

with tab1:
    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
        help="Supported formats: WAV, MP3, M4A, OGG, FLAC, WEBM"
    )

    if uploaded_file:
        st.audio(uploaded_file)
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            audio_file_path = tmp.name

with tab2:
    st.info("Click the microphone to start recording. Click again to stop.")
    try:
        from audiorecorder import audiorecorder
        audio = audiorecorder("🎤 Click to record", "⏹️ Recording... Click to stop")

        if len(audio) > 0:
            st.audio(audio.export().read())
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                audio.export(tmp.name, format="wav")
                audio_file_path = tmp.name
    except ImportError:
        st.warning("Audio recording requires `streamlit-audiorecorder`. Install with: `pip install streamlit-audiorecorder`")

st.markdown("---")

# Language selection
language = st.selectbox(
    "Language",
    options=["mn", "en", "auto"],
    format_func=lambda x: {"mn": "Mongolian", "en": "English", "auto": "Auto-detect"}.get(x, x),
    index=0
)

# Task selection
task = st.radio(
    "Task",
    options=["transcribe", "translate"],
    format_func=lambda x: {"transcribe": "Transcribe (keep original language)", "translate": "Translate to English"}.get(x, x),
    horizontal=True
)

# Transcribe button
if st.button("🚀 Transcribe", type="primary", disabled=not st.session_state.model):
    if not audio_file_path:
        st.warning("Please upload or record an audio file first.")
    elif not st.session_state.model:
        st.warning("Please load the model first (use the sidebar).")
    else:
        with st.spinner("Transcribing..."):
            try:
                # Transcribe with Whisper
                options = {
                    "task": task,
                    "fp16": False,  # Use FP32 for CPU
                }
                if language != "auto":
                    options["language"] = language

                result = st.session_state.model.transcribe(audio_file_path, **options)

                st.success("Transcription complete!")

                # Show detected language if auto
                if language == "auto":
                    detected = result.get("language", "unknown")
                    st.info(f"Detected language: {detected}")

                st.markdown("### Result:")
                st.text_area(
                    "Transcription",
                    value=result["text"],
                    height=200,
                    label_visibility="collapsed"
                )

                # Show segments with timestamps
                with st.expander("Show timestamps"):
                    for segment in result["segments"]:
                        start = segment["start"]
                        end = segment["end"]
                        text = segment["text"]
                        st.text(f"[{start:.2f}s - {end:.2f}s] {text}")

            except Exception as e:
                st.error(f"Error during transcription: {e}")
            finally:
                if audio_file_path and os.path.exists(audio_file_path):
                    try:
                        os.unlink(audio_file_path)
                    except:
                        pass

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <small>
            Using <a href='https://github.com/openai/whisper'>OpenAI Whisper</a> - Supports 99 languages including Mongolian
        </small>
    </div>
    """,
    unsafe_allow_html=True
)
