import streamlit as st
import tempfile
import os
from pathlib import Path
from openai import OpenAI

st.set_page_config(page_title="TsuurAI - Speech to Text", page_icon="🎤", layout="wide")

# Load OpenAI API key
OPENAI_KEY_PATH = Path(__file__).parent.parent / "openai-key"
if OPENAI_KEY_PATH.exists():
    OPENAI_API_KEY = OPENAI_KEY_PATH.read_text().strip()
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    OPENAI_API_KEY = None
    openai_client = None

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
}

LANGUAGE_CODES = {
    "English": {"whisper": "en", "mms": "eng"},
    "Mongolian": {"whisper": "mn", "mms": "mon"},
}

# LLM Correction function
def correct_with_llm(raw_text, language, n_best_candidates=None):
    """Use GPT-4o-mini to correct ASR output"""
    if not openai_client:
        return raw_text, "OpenAI key not found"

    # Build prompt based on whether we have N-best candidates
    if n_best_candidates and len(n_best_candidates) > 1:
        candidates_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(n_best_candidates)])
        prompt = f"""You are an expert {language} language corrector for speech recognition output.

Given these ASR candidates (ranked by confidence):
{candidates_text}

Select the best candidate and correct any errors. Fix:
- Spelling mistakes
- Grammar issues
- Word boundaries (words incorrectly split or merged)
- Common ASR errors

Return ONLY the corrected {language} text, nothing else."""
    else:
        prompt = f"""You are an expert {language} language corrector for speech recognition output.

Raw ASR output:
{raw_text}

Correct any errors in this {language} text. Fix:
- Spelling mistakes
- Grammar issues
- Word boundaries (words incorrectly split or merged)
- Common ASR errors

Return ONLY the corrected {language} text, nothing else."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        corrected = response.choices[0].message.content.strip()
        return corrected, None
    except Exception as e:
        return raw_text, str(e)

# Helper function to check if model exists
def check_model_exists(model_key, model_size):
    if model_key == "whisper":
        # Check in huggingface cache (whisper_s2t uses HF cache)
        whisper_cache = MODELS_DIR / "huggingface" / "hub" / f"models--Systran--faster-whisper-{model_size}"
        return whisper_cache.exists()
    else:  # MMS
        mms_cache = MODELS_DIR / "huggingface" / "hub" / "models--facebook--mms-1b-all"
        return mms_cache.exists()

# Sidebar for model configuration
with st.sidebar:
    st.header("Model Configuration")

    # Model family selection
    model_family = st.selectbox(
        "AI Model Family",
        ["Whisper (OpenAI)", "MMS (Meta)"],
        index=1  # Default: MMS
    )

    # Model size selection based on family
    if model_family == "Whisper (OpenAI)":
        model_key = "whisper"
        model_size = st.selectbox(
            "Model Size",
            ["tiny", "base", "small", "medium", "large-v3"],
            index=2
        )
    else:  # MMS (Meta)
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
        index=1  # Default: Mongolian
    )

    # Language-specific recommendation
    if language == "Mongolian":
        st.caption("For Mongolian: MMS has best coverage, Whisper is fastest")

    st.divider()

    # LLM Correction toggle
    st.header("LLM Correction")
    if openai_client:
        use_llm_correction = st.toggle("Enable GPT-4o-mini correction", value=True)
        if use_llm_correction:
            st.caption("ASR output will be corrected by GPT-4o-mini")
    else:
        use_llm_correction = False
        st.warning("OpenAI key not found. Add 'openai-key' file to enable.")

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
        """)

        # Contextual recommendation
        if model_key == "whisper":
            st.success("Recommended for: General use, English, quick setup")
        else:  # MMS
            if language == "Mongolian":
                st.success("Good choice for Mongolian - trained on 1100+ languages")
            else:
                st.info("Better for rare languages. For English, consider Whisper.")

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
    # whisper_s2t uses HF_HOME/XDG_CACHE_HOME env vars set at top of file
    return whisper_s2t.load_model(
        model_identifier=model_id,
        backend='CTranslate2',
        compute_type='float16'
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

# Load selected model
model_exists = check_model_exists(model_key, model_size)
loading_msg = f"Loading {model_size}..." if model_exists else f"Downloading & loading {model_size} (first time only)..."

with st.spinner(loading_msg):
    if model_key == "whisper":
        model = load_whisper_model(model_size)
        st.success(f"Whisper '{model_size}' ready! (saved to models/)")
    else:  # MMS
        mms_processor, mms_model = load_mms_model()
        st.success("Meta MMS ready! (saved to models/huggingface/)")

def transcribe_audio(audio_data, file_ext=".wav"):
    import torch
    import numpy as np

    st.caption(f"Debug: transcribe_audio called, data size={len(audio_data)}, ext={file_ext}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        st.caption(f"Debug: Temp file created at {tmp_path}")

        if model_key == "whisper":
            lang_code = LANGUAGE_CODES[language]["whisper"]
            st.info(f"Transcribing with: **{model_family}** | Language: **{language}** ({lang_code})")
            st.caption("Debug: Calling whisper transcribe_with_vad...")
            result = model.transcribe_with_vad([tmp_path], lang_codes=[lang_code])
            st.caption(f"Debug: Whisper returned {len(result)} results")

            full_text = " ".join([seg['text'] for seg in result[0]])

        else:  # MMS
            import librosa
            st.caption("Debug: Loading audio with librosa...")
            audio, sr = librosa.load(tmp_path, sr=16000)
            st.caption(f"Debug: Audio loaded, shape={audio.shape}, sr={sr}")

            lang_code = LANGUAGE_CODES[language]["mms"]
            st.info(f"Transcribing with: **{model_family}** | Language: **{language}** ({lang_code})")
            st.caption(f"Debug: Setting MMS language to {lang_code}...")
            mms_processor.tokenizer.set_target_lang(lang_code)
            mms_model.load_adapter(lang_code)

            st.caption("Debug: Processing audio...")
            inputs = mms_processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            st.caption("Debug: Running inference...")
            with torch.no_grad():
                outputs = mms_model(**inputs).logits

            ids = torch.argmax(outputs, dim=-1)[0]
            full_text = mms_processor.decode(ids)
            st.caption(f"Debug: Transcription complete, length={len(full_text)}")

        # Display results
        st.subheader("Raw ASR Output")
        st.text_area("Raw transcription", full_text, height=100, key="raw_output")

        # LLM Correction
        if use_llm_correction and full_text:
            with st.spinner("Correcting with GPT-4o-mini..."):
                corrected_text, error = correct_with_llm(full_text, language)

            if error:
                st.warning(f"LLM correction failed: {error}")
                st.subheader("Final Text")
                st.text_area("Output", full_text, height=150, key="final_output")
            else:
                st.subheader("LLM Corrected Output")
                st.text_area("Corrected transcription", corrected_text, height=150, key="corrected_output")

                # Show diff if different
                if corrected_text != full_text:
                    with st.expander("Show changes"):
                        st.write("**Raw:**", full_text)
                        st.write("**Corrected:**", corrected_text)

                return corrected_text
        else:
            st.subheader("Final Text")
            st.text_area("Output", full_text, height=150, key="final_output")

        return full_text

    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Main content - Show current settings
st.markdown(f"**Current settings:** {model_family} ({model_size}) | Language: {language}")

# Tabs for input method
tab1, tab2 = st.tabs(["🎙️ Record", "📁 Upload File"])

with tab1:
    st.write("Click the microphone button to record your voice")

    try:
        # Use Streamlit's built-in audio input
        audio_value = st.audio_input("Record audio", key="audio_recorder")
        st.caption("Debug: audio_input component loaded")

        if audio_value:
            audio_bytes = audio_value.getvalue()
            st.success(f"Recording captured: {len(audio_bytes) / 1024:.1f} KB")
            st.caption(f"Debug: Audio type={type(audio_value)}, bytes length={len(audio_bytes)}")
            st.audio(audio_value)

            if st.button("Transcribe Recording", type="primary", key="transcribe_rec"):
                st.caption("Debug: Transcribe button clicked")
                st.caption(f"Debug: Saving {len(audio_bytes)} bytes to temp file...")
                with st.spinner("Transcribing recording..."):
                    transcribe_audio(audio_bytes, ".wav")
        else:
            st.caption("Debug: No audio recorded yet")

    except Exception as e:
        st.error(f"Recording error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

with tab2:
    uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "flac", "ogg"])

    if uploaded_file:
        st.success(f"File: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
        st.audio(uploaded_file)
        if st.button("Transcribe File", type="primary", key="transcribe_file"):
            with st.spinner(f"Transcribing {uploaded_file.name}..."):
                transcribe_audio(uploaded_file.getvalue(), os.path.splitext(uploaded_file.name)[1])

st.divider()
st.caption("TsuurAI - Running on GPU server 160.187.40.172")
