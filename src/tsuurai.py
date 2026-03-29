import streamlit as st
import tempfile
import os
from pathlib import Path
from audio_recorder_streamlit import audio_recorder

st.set_page_config(page_title="TsuurAI - Speech to Text", page_icon="🎤", layout="wide")

# Setup persistent models directory
SCRIPT_DIR = Path(__file__).parent.resolve()
MODELS_DIR = SCRIPT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Set environment variables for model caching
os.environ["HF_HOME"] = str(MODELS_DIR / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR / "huggingface")
os.environ["XDG_CACHE_HOME"] = str(MODELS_DIR)

st.title("🎤 TsuurAI - Speech to Text")
st.write("Multi-model speech recognition on NVIDIA A2 GPU")

# Model information database
MODEL_INFO = {
    "whisper": {
        "tiny": {"params": "39M", "size_gb": 0.15, "speed": "~32x", "languages": 99, "quality": "Basic"},
        "base": {"params": "74M", "size_gb": 0.29, "speed": "~16x", "languages": 99, "quality": "Good"},
        "small": {"params": "244M", "size_gb": 0.97, "speed": "~6x", "languages": 99, "quality": "Better"},
        "medium": {"params": "769M", "size_gb": 3.1, "speed": "~2x", "languages": 99, "quality": "Great"},
        "large-v3": {"params": "1550M", "size_gb": 6.2, "speed": "~1x", "languages": 99, "quality": "Best"},
    },
    "meta_mms": {
        "mms-1b-all": {"params": "1000M", "size_gb": 4.0, "speed": "~2x", "languages": 1162, "quality": "Great"},
    },
    "meta_seamless": {
        "seamlessM4T-medium": {"params": "1200M", "size_gb": 4.8, "speed": "~1.5x", "languages": 100, "quality": "Great"},
    }
}

# Model comparison data
MODEL_COMPARISON = {
    "whisper": {
        "strength": "Best zero-shot accuracy",
        "weakness": "Hard to fine-tune",
        "best_for": "Quick deployment, general use"
    },
    "meta_mms": {
        "strength": "1162 language support",
        "weakness": "Inconsistent quality",
        "best_for": "Rare/low-resource languages"
    },
    "meta_seamless": {
        "strength": "Most advanced, translation",
        "weakness": "Less flexible",
        "best_for": "Translation apps"
    }
}

LANGUAGE_CODES = {
    "English": {"whisper": "en", "mms": "eng", "seamless": "eng"},
    "Mongolian": {"whisper": "mn", "mms": "mon", "seamless": "mon"},
}

# Helper function to check if model exists
def check_model_exists(model_key, model_size):
    if model_key == "whisper":
        whisper_cache = MODELS_DIR / "whisper" / model_size
        return whisper_cache.exists() and any(whisper_cache.iterdir()) if whisper_cache.exists() else False
    elif model_key == "meta_mms":
        mms_cache = MODELS_DIR / "huggingface" / "hub" / "models--facebook--mms-1b-all"
        return mms_cache.exists()
    else:
        seamless_cache = MODELS_DIR / "huggingface" / "hub" / "models--facebook--hf-seamless-m4t-medium"
        return seamless_cache.exists()

# Sidebar for model configuration
with st.sidebar:
    st.header("Model Configuration")

    # Model family selection
    model_family = st.selectbox(
        "AI Model Family",
        ["Whisper (OpenAI)", "MMS (Meta)", "SeamlessM4T (Meta)"],
        index=0
    )

    # Model size selection based on family
    if model_family == "Whisper (OpenAI)":
        model_key = "whisper"
        model_size = st.selectbox(
            "Model Size",
            ["tiny", "base", "small", "medium", "large-v3"],
            index=2
        )
    elif model_family == "MMS (Meta)":
        model_key = "meta_mms"
        model_size = st.selectbox(
            "Model Size",
            ["mms-1b-all"],
            index=0
        )
    else:
        model_key = "meta_seamless"
        model_size = st.selectbox(
            "Model Size",
            ["seamlessM4T-medium"],
            index=0
        )

    # Language selection
    language = st.selectbox(
        "Language",
        ["English", "Mongolian"],
        index=0
    )

    # Language-specific recommendation
    if language == "Mongolian":
        st.caption("For Mongolian: MMS has best coverage, Whisper is fastest")

    st.divider()

    # Model info display
    st.header("Model Information")
    info = MODEL_INFO[model_key][model_size]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Parameters", info["params"])
        st.metric("Size", f"{info['size_gb']} GB")
    with col2:
        st.metric("Speed", info["speed"])
        st.metric("Languages", info["languages"])

    st.info(f"Quality: **{info['quality']}**")

    # Show model download status
    model_exists = check_model_exists(model_key, model_size)
    if model_exists:
        st.success("Status: Downloaded")
    else:
        st.warning("Status: Will download on first use")

    # Show model strengths/weaknesses
    comparison = MODEL_COMPARISON[model_key]
    st.success(f"Strength: {comparison['strength']}")
    st.warning(f"Weakness: {comparison['weakness']}")
    st.caption(f"Best for: {comparison['best_for']}")

    # Additional model details
    with st.expander("More Details"):
        st.write(f"**Family:** {model_family}")
        st.write(f"**Model:** {model_size}")
        st.write(f"**Selected Language:** {language}")
        if model_key == "whisper":
            st.write("**Backend:** CTranslate2 (optimized)")
            st.write("**Precision:** FP16")
        elif model_key == "meta_mms":
            st.write("**Architecture:** Wav2Vec2")
            st.write("**Training:** 1100+ languages")
        else:
            st.write("**Architecture:** SeamlessM4T")
            st.write("**Features:** Speech-to-text, Translation")

    # Honest comparison
    with st.expander("Model Comparison"):
        st.markdown("""
| Model | Strength | Weakness | Best For |
|-------|----------|----------|----------|
| **Whisper** | Best zero-shot | Hard to fine-tune | Quick deployment |
| **MMS** | 1162 languages | Inconsistent quality | Rare languages |
| **SeamlessM4T** | Most advanced | Less flexible | Translation apps |
        """)

        # Contextual recommendation
        if model_key == "whisper":
            st.success("Recommended for: General use, English, quick setup")
        elif model_key == "meta_mms":
            if language == "Mongolian":
                st.success("Good choice for Mongolian - trained on 1100+ languages")
            else:
                st.info("Better for rare languages. For English, consider Whisper.")
        else:
            st.success("Best for: Speech translation, multilingual apps")

    # Storage info
    with st.expander("Storage Info"):
        st.write(f"**Models folder:** `{MODELS_DIR}`")
        if MODELS_DIR.exists():
            # Calculate total size
            total_size = sum(f.stat().st_size for f in MODELS_DIR.rglob('*') if f.is_file())
            size_gb = total_size / (1024**3)
            st.write(f"**Total cached:** {size_gb:.2f} GB")
        st.caption("Models are downloaded once and reused")

# Load model based on selection
@st.cache_resource
def load_whisper_model(model_id):
    import whisper_s2t
    whisper_cache = MODELS_DIR / "whisper"
    whisper_cache.mkdir(exist_ok=True)

    return whisper_s2t.load_model(
        model_identifier=model_id,
        backend='CTranslate2',
        compute_type='float16',
        cache_dir=str(whisper_cache)
    )

@st.cache_resource
def load_mms_model():
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    hf_cache = MODELS_DIR / "huggingface"
    hf_cache.mkdir(exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        "facebook/mms-1b-all",
        cache_dir=str(hf_cache)
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/mms-1b-all",
        cache_dir=str(hf_cache)
    )
    model = model.to("cuda")
    return processor, model

@st.cache_resource
def load_seamless_model():
    from transformers import AutoProcessor, SeamlessM4TModel
    hf_cache = MODELS_DIR / "huggingface"
    hf_cache.mkdir(exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        "facebook/hf-seamless-m4t-medium",
        cache_dir=str(hf_cache)
    )
    model = SeamlessM4TModel.from_pretrained(
        "facebook/hf-seamless-m4t-medium",
        cache_dir=str(hf_cache)
    )
    model = model.to("cuda")
    return processor, model

# Load selected model
model_exists = check_model_exists(model_key, model_size)
loading_msg = f"Loading {model_size}..." if model_exists else f"Downloading & loading {model_size} (first time only)..."

with st.spinner(loading_msg):
    if model_key == "whisper":
        model = load_whisper_model(model_size)
        st.success(f"Whisper '{model_size}' ready! (saved to models/whisper/)")
    elif model_key == "meta_mms":
        mms_processor, mms_model = load_mms_model()
        st.success("Meta MMS ready! (saved to models/huggingface/)")
    else:
        seamless_processor, seamless_model = load_seamless_model()
        st.success("Meta SeamlessM4T ready! (saved to models/huggingface/)")

def transcribe_audio(audio_data, file_ext=".wav"):
    import torch
    import numpy as np

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    try:
        if model_key == "whisper":
            lang_code = LANGUAGE_CODES[language]["whisper"]
            result = model.transcribe_with_vad([tmp_path], lang_codes=[lang_code])

            st.subheader("Transcription")
            for segment in result[0]:
                st.write(segment['text'])

            full_text = " ".join([seg['text'] for seg in result[0]])

        elif model_key == "meta_mms":
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
            full_text = mms_processor.decode(ids)

            st.subheader("Transcription")
            st.write(full_text)

        else:  # SeamlessM4T
            import librosa
            audio, sr = librosa.load(tmp_path, sr=16000)

            lang_code = LANGUAGE_CODES[language]["seamless"]
            inputs = seamless_processor(audios=audio, return_tensors="pt", sampling_rate=16000)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                output_tokens = seamless_model.generate(**inputs, tgt_lang=lang_code, generate_speech=False)

            full_text = seamless_processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

            st.subheader("Transcription")
            st.write(full_text)

        st.text_area("Full Text", full_text, height=150)
        return full_text

    finally:
        os.unlink(tmp_path)

# Main content - Tabs for input method
tab1, tab2 = st.tabs(["🎙️ Record", "📁 Upload"])

with tab1:
    st.write("Click microphone to start, click again to stop")

    audio_bytes = audio_recorder(
        text="",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_size="3x",
        pause_threshold=3.0,
        sample_rate=16000
    )

    if audio_bytes:
        st.info(f"Audio captured: {len(audio_bytes)} bytes")
        st.audio(audio_bytes, format="audio/wav")

        with st.spinner("Transcribing..."):
            transcribe_audio(audio_bytes)

with tab2:
    uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "flac", "ogg"])

    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("Transcribe", type="primary", key="upload_btn"):
            with st.spinner("Transcribing..."):
                transcribe_audio(uploaded_file.getvalue(), os.path.splitext(uploaded_file.name)[1])

st.divider()
st.caption("TsuurAI - Running on GPU server 160.187.40.172")
