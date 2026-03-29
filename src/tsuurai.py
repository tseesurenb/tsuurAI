import streamlit as st
import whisper_s2t
import tempfile
import os

st.set_page_config(page_title="Speech to Text", page_icon="🎤", layout="wide")

st.title("🎤 Speech to Text")
st.write("Powered by WhisperS2T on NVIDIA A2 GPU")

# Model selection
model_size = st.selectbox(
    "Select Whisper Model",
    ["base", "small", "medium", "large-v3"],
    index=1
)

@st.cache_resource
def load_model(model_id):
    return whisper_s2t.load_model(
        model_identifier=model_id,
        backend='CTranslate2',
        compute_type='float16'
    )

# Load model
with st.spinner(f"Loading {model_size} model..."):
    model = load_model(model_size)

st.success(f"Model '{model_size}' loaded!")

# Tabs for input method
tab1, tab2 = st.tabs(["🎙️ Record", "📁 Upload"])

def transcribe_audio(audio_data, file_ext=".wav"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    try:
        result = model.transcribe_with_vad([tmp_path])

        st.subheader("Transcription")
        for segment in result[0]:
            st.write(segment['text'])

        full_text = " ".join([seg['text'] for seg in result[0]])
        st.text_area("Full Text", full_text, height=200)
    finally:
        os.unlink(tmp_path)

with tab1:
    st.write("Click the microphone to start recording")
    audio_value = st.audio_input("Record your voice")

    if audio_value:
        st.audio(audio_value)
        if st.button("Transcribe Recording", type="primary", key="rec_btn"):
            with st.spinner("Transcribing..."):
                transcribe_audio(audio_value.getvalue())

with tab2:
    uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "flac", "ogg"])

    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("Transcribe File", type="primary", key="upload_btn"):
            with st.spinner("Transcribing..."):
                transcribe_audio(uploaded_file.getvalue(), os.path.splitext(uploaded_file.name)[1])

st.divider()
st.caption("Running on GPU server 160.187.40.172")
