import streamlit as st
import tempfile
import os
from pathlib import Path
from openai import OpenAI

st.set_page_config(page_title="TsuurAI - Speech to Text", page_icon="🎤", layout="wide")

# Load OpenAI API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
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

# Load correction prompts from files
PROMPTS_DIR = SCRIPT_DIR / "prompts"
def load_prompt(language):
    """Load correction prompt from markdown file"""
    prompt_file = PROMPTS_DIR / f"{language.lower()}_correction.md"
    if prompt_file.exists():
        return prompt_file.read_text()
    return None

# Local LLM models info
LOCAL_LLM_INFO = {
    "Mongolian-Llama3": {
        "model_id": "Dorjzodovsuren/Mongolian_Llama3-v0.1",
        "base_model": "unsloth/llama-3-8b-bnb-4bit",
        "size_gb": 5.0,
        "languages": "Mongolian, English",
        "description": "First Mongolian instruction-tuned LLM"
    },
    "Qwen3-8B": {
        "model_id": "Qwen/Qwen3-8B",
        "base_model": None,
        "size_gb": 5.0,
        "languages": "119 languages",
        "description": "Best multilingual coverage"
    },
}

# Load local LLM
@st.cache_resource
def load_local_llm(model_name):
    """Load a local LLM with 4-bit quantization"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_info = LOCAL_LLM_INFO[model_name]

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    hf_cache = MODELS_DIR / "huggingface"

    if model_name == "Mongolian-Llama3":
        # Mongolian-Llama3 uses PEFT adapter
        from peft import PeftModel

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_info["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=str(hf_cache),
        )
        # Load tokenizer from adapter
        tokenizer = AutoTokenizer.from_pretrained(
            model_info["model_id"],
            cache_dir=str(hf_cache),
        )
        # Load PEFT adapter
        model = PeftModel.from_pretrained(
            base_model,
            model_info["model_id"],
            cache_dir=str(hf_cache),
        )
    else:
        # Standard model loading (Qwen, etc.)
        tokenizer = AutoTokenizer.from_pretrained(
            model_info["model_id"],
            cache_dir=str(hf_cache),
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_info["model_id"],
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=str(hf_cache),
            trust_remote_code=True,
        )

    return model, tokenizer

def correct_with_local_llm(raw_text, language, model, tokenizer, n_best_candidates=None):
    """Use local LLM to correct ASR output"""
    import torch

    # Load language-specific prompt from file
    system_prompt = load_prompt(language)
    if not system_prompt:
        system_prompt = f"You are an expert {language} language corrector for speech recognition output."

    # Build prompt
    if n_best_candidates and len(n_best_candidates) > 1:
        candidates_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(n_best_candidates)])
        prompt = f"""{system_prompt}

## ASR Candidates (ranked by confidence):
{candidates_text}

## Correct the text now:"""
    else:
        prompt = f"""{system_prompt}

## ASR Output to correct:
{raw_text}

## Corrected text:"""

    try:
        # Format prompt - use simple format that works with most models
        # Llama-3 instruct format
        input_text = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode and extract response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part (after the prompt)
        # Try different extraction methods
        if "<|eot_id|>" in full_response:
            # Llama-3 format - get text after last assistant header
            parts = full_response.split("assistant")
            if len(parts) > 1:
                corrected = parts[-1].strip()
            else:
                corrected = full_response.strip()
        elif "Assistant:" in full_response:
            corrected = full_response.split("Assistant:")[-1].strip()
        else:
            # Remove the input prompt from output
            corrected = full_response.split(prompt)[-1].strip()

        # Clean up any remaining special tokens or markers
        corrected = corrected.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()

        return corrected, None

    except Exception as e:
        return raw_text, str(e)

# LLM Correction function (OpenAI API)
def correct_with_llm(raw_text, language, model_name="gpt-4o-mini", n_best_candidates=None):
    """Use OpenAI LLM to correct ASR output"""
    if not openai_client:
        return raw_text, "OpenAI key not found"

    # Load language-specific prompt from file
    system_prompt = load_prompt(language)
    if not system_prompt:
        system_prompt = f"You are an expert {language} language corrector for speech recognition output."

    # Build prompt based on whether we have N-best candidates
    if n_best_candidates and len(n_best_candidates) > 1:
        candidates_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(n_best_candidates)])
        prompt = f"""{system_prompt}

## ASR Candidates (ranked by confidence):
{candidates_text}

## Correct the text now:"""
    else:
        prompt = f"""{system_prompt}

## ASR Output to correct:
{raw_text}

## Corrected text:"""

    try:
        # o1 models don't support temperature
        if model_name.startswith("o1"):
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1000
            )
        else:
            response = openai_client.chat.completions.create(
                model=model_name,
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
    use_llm_correction = st.toggle("Enable LLM correction", value=True)

    if use_llm_correction:
        # Choose between API and Local LLM
        llm_type = st.radio(
            "LLM Type",
            ["OpenAI API", "Local LLM (GPU)"],
            index=1 if language == "Mongolian" else 0,
            horizontal=True
        )

        if llm_type == "OpenAI API":
            use_local_llm = False
            if openai_client:
                llm_model = st.selectbox(
                    "API Model",
                    ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "o1-mini", "o1"],
                    index=0
                )
                # Show model info
                api_llm_info = {
                    "gpt-4o-mini": {"cost": "$0.15/$0.60 per 1M", "speed": "Very fast", "quality": "Good"},
                    "gpt-4o": {"cost": "$2.50/$10 per 1M", "speed": "Fast", "quality": "Excellent"},
                    "gpt-4-turbo": {"cost": "$10/$30 per 1M", "speed": "Medium", "quality": "Excellent"},
                    "o1-mini": {"cost": "$1.10/$4.40 per 1M", "speed": "Slower", "quality": "Best reasoning"},
                    "o1": {"cost": "$7.50/$30 per 1M", "speed": "Slow", "quality": "Most advanced"},
                }
                info = api_llm_info[llm_model]
                st.caption(f"{info['quality']} | {info['speed']} | {info['cost']}")
            else:
                llm_model = None
                st.warning("Set OPENAI_API_KEY environment variable")
        else:
            use_local_llm = True
            local_llm_name = st.selectbox(
                "Local Model",
                list(LOCAL_LLM_INFO.keys()),
                index=0 if language == "Mongolian" else 1
            )
            llm_model = local_llm_name
            # Show local model info
            local_info = LOCAL_LLM_INFO[local_llm_name]
            st.caption(f"{local_info['description']}")
            st.caption(f"Size: {local_info['size_gb']} GB | Languages: {local_info['languages']}")

        # Top-K candidates selection
        top_k = st.slider("Top-K candidates", min_value=1, max_value=10, value=5)
        st.caption(f"ASR will generate {top_k} candidates for LLM to choose from")
    else:
        llm_model = None
        top_k = 1
        use_local_llm = False

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
    from faster_whisper import WhisperModel
    # Load faster-whisper model directly for beam search support
    model = WhisperModel(
        model_id,
        device="cuda",
        compute_type="float16",
        download_root=str(MODELS_DIR / "huggingface")
    )
    return model

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

def get_mms_nbest(logits, processor, beam_width=5):
    """Get N-best candidates from MMS using CTC beam search"""
    import torch
    import numpy as np

    try:
        from pyctcdecode import build_ctcdecoder

        # Get vocabulary from processor
        vocab_dict = processor.tokenizer.get_vocab()
        # Sort by index to get correct order
        sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
        labels = [item[0] for item in sorted_vocab]

        # Build CTC decoder
        decoder = build_ctcdecoder(labels)

        # Convert logits to numpy and get log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs_np = log_probs.cpu().numpy()[0]  # Remove batch dimension

        # Decode with beam search
        beams = decoder.decode_beams(log_probs_np, beam_width=beam_width)

        # Extract text and scores
        candidates = []
        for beam in beams[:beam_width]:
            text = beam[0]  # The decoded text
            score = beam[-1]  # Log probability score
            candidates.append({"text": text, "score": float(score)})

        return candidates

    except Exception as e:
        # Fallback to greedy decoding if beam search fails
        ids = torch.argmax(logits, dim=-1)[0]
        text = processor.decode(ids)
        return [{"text": text, "score": 1.0}]

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

# Load local LLM if selected
local_llm_model = None
local_llm_tokenizer = None
if use_llm_correction and use_local_llm and llm_model:
    with st.spinner(f"Loading {llm_model}... (first time may take a few minutes)"):
        local_llm_model, local_llm_tokenizer = load_local_llm(llm_model)
        st.success(f"Local LLM '{llm_model}' ready!")

def transcribe_audio(audio_data, file_ext=".wav"):
    import torch
    import numpy as np

    st.caption(f"Debug: transcribe_audio called, data size={len(audio_data)}, ext={file_ext}")

    # Determine beam width based on settings
    beam_width = top_k if use_llm_correction else 1

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        st.caption(f"Debug: Temp file created at {tmp_path}")

        candidates = []

        if model_key == "whisper":
            lang_code = LANGUAGE_CODES[language]["whisper"]
            st.info(f"Transcribing with: **{model_family}** | Language: **{language}** ({lang_code}) | Beam: {beam_width}")

            # Use faster-whisper with beam search
            segments, info = model.transcribe(
                tmp_path,
                language=lang_code,
                beam_size=beam_width,
                best_of=beam_width,
                temperature=0.0,
            )

            # Collect segments into full text
            segment_texts = [segment.text for segment in segments]
            full_text = " ".join(segment_texts).strip()

            # For Whisper, we get one result but with beam search for better accuracy
            # To get actual N-best, we'd need multiple temperature samples
            if beam_width > 1 and use_llm_correction:
                # Generate alternative hypotheses using temperature sampling
                st.caption("Debug: Generating alternative hypotheses...")
                candidates.append({"text": full_text, "score": 1.0})

                for i, temp in enumerate([0.2, 0.4, 0.6, 0.8, 1.0][:beam_width-1]):
                    try:
                        alt_segments, _ = model.transcribe(
                            tmp_path,
                            language=lang_code,
                            beam_size=3,
                            temperature=temp,
                        )
                        alt_text = " ".join([s.text for s in alt_segments]).strip()
                        if alt_text and alt_text != full_text:
                            candidates.append({"text": alt_text, "score": 1.0 - temp})
                    except:
                        pass
            else:
                candidates = [{"text": full_text, "score": 1.0}]

        else:  # MMS
            import librosa
            st.caption("Debug: Loading audio with librosa...")
            audio, sr = librosa.load(tmp_path, sr=16000)
            st.caption(f"Debug: Audio loaded, shape={audio.shape}, sr={sr}")

            lang_code = LANGUAGE_CODES[language]["mms"]
            st.info(f"Transcribing with: **{model_family}** | Language: **{language}** ({lang_code}) | Beam: {beam_width}")
            st.caption(f"Debug: Setting MMS language to {lang_code}...")
            mms_processor.tokenizer.set_target_lang(lang_code)
            mms_model.load_adapter(lang_code)

            st.caption("Debug: Processing audio...")
            inputs = mms_processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            st.caption("Debug: Running inference...")
            with torch.no_grad():
                outputs = mms_model(**inputs).logits

            # Get N-best candidates using beam search
            if beam_width > 1 and use_llm_correction:
                st.caption(f"Debug: Getting top-{beam_width} candidates with beam search...")
                candidates = get_mms_nbest(outputs, mms_processor, beam_width=beam_width)
            else:
                ids = torch.argmax(outputs, dim=-1)[0]
                full_text = mms_processor.decode(ids)
                candidates = [{"text": full_text, "score": 1.0}]

            full_text = candidates[0]["text"] if candidates else ""
            st.caption(f"Debug: Transcription complete, {len(candidates)} candidates")

        # Get the best candidate as primary result
        full_text = candidates[0]["text"] if candidates else ""

        # Display results
        st.subheader("ASR Output")

        # Show N-best candidates if available
        if len(candidates) > 1:
            with st.expander(f"Top-{len(candidates)} ASR Candidates", expanded=True):
                for i, cand in enumerate(candidates):
                    score_pct = cand['score'] * 100 if cand['score'] <= 1 else cand['score']
                    st.write(f"**{i+1}.** {cand['text']}")
                    st.caption(f"Score: {score_pct:.1f}")
        else:
            st.text_area("Raw transcription", full_text, height=100, key="raw_output")

        # LLM Correction
        if use_llm_correction and full_text:
            candidate_texts = [c["text"] for c in candidates] if len(candidates) > 1 else None

            with st.spinner(f"Correcting with {llm_model}..."):
                if use_local_llm and local_llm_model is not None:
                    # Use local LLM
                    corrected_text, error = correct_with_local_llm(
                        full_text,
                        language,
                        local_llm_model,
                        local_llm_tokenizer,
                        n_best_candidates=candidate_texts
                    )
                else:
                    # Use OpenAI API
                    corrected_text, error = correct_with_llm(
                        full_text,
                        language,
                        model_name=llm_model,
                        n_best_candidates=candidate_texts
                    )

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
                        st.write("**Raw (best):**", full_text)
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
