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

# File upload
uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "flac", "ogg"])

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Transcribe", type="primary"):
        with st.spinner("Transcribing..."):
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                # Transcribe
                result = model.transcribe_with_vad([tmp_path])

                # Display results
                st.subheader("Transcription")
                for segment in result[0]:
                    st.write(segment['text'])

                # Full text
                full_text = " ".join([seg['text'] for seg in result[0]])
                st.text_area("Full Text", full_text, height=200)

            finally:
                os.unlink(tmp_path)

st.divider()
st.caption("Running on GPU server 160.187.40.172")
