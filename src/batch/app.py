"""
TsuurAI Batch Processing App
Record or upload audio, then process
"""

import streamlit as st
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
SRC_DIR = Path(__file__).parent.parent.resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.config import (
    LANGUAGE_CODES, MODEL_INFO, MODEL_COMPARISON,
    LOCAL_LLM_INFO, API_LLM_INFO
)
from common.models import (
    check_model_exists, load_whisper_model, load_mms_model,
    load_local_llm, get_mms_nbest
)
from common.llm import (
    openai_client, correct_with_llm, correct_with_local_llm,
    refine_with_llm, refine_with_local_llm
)
from common.auth import show_login_page, show_user_sidebar, log_usage

st.set_page_config(page_title="TsuurAI - Batch", page_icon="🎤", layout="wide")

# Check authentication
if not st.session_state.get("authenticated"):
    show_login_page()
    st.stop()

st.title("🎤 TsuurAI - Batch Processing")
st.write("Record or upload audio for transcription")

# Show user info in sidebar
show_user_sidebar()

# Sidebar for model configuration
with st.sidebar:
    st.header("Model Configuration")

    # Model family selection
    model_family = st.selectbox(
        "AI Model Family",
        ["Whisper (OpenAI)", "MMS (Meta)"],
        index=1  # Default: MMS
    )

    if model_family == "Whisper (OpenAI)":
        model_key = "whisper"
        model_size = st.selectbox(
            "Model Size",
            ["tiny", "base", "small", "medium", "large-v3"],
            index=2
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
        index=1  # Default: Mongolian
    )

    if language == "Mongolian":
        st.caption("For Mongolian: MMS has best coverage, Whisper is fastest")

    st.divider()

    # LLM Correction toggle
    st.header("LLM Correction")
    use_llm_correction = st.toggle("Enable LLM correction", value=True)

    if use_llm_correction:
        llm_type = st.radio(
            "LLM Type",
            ["OpenAI API", "Local LLM (GPU)"],
            index=0,
            horizontal=True
        )

        if llm_type == "OpenAI API":
            use_local_llm = False
            if openai_client:
                llm_model = st.selectbox(
                    "API Model",
                    list(API_LLM_INFO.keys()),
                    index=0
                )
                info = API_LLM_INFO[llm_model]
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
            local_info = LOCAL_LLM_INFO[local_llm_name]
            st.caption(f"{local_info['description']}")
            st.caption(f"Size: {local_info['size_gb']} GB | Languages: {local_info['languages']}")

        top_k = st.slider("Top-K candidates", min_value=1, max_value=10, value=5)
        st.caption(f"ASR will generate {top_k} candidates for LLM to choose from")

        use_two_pass = st.toggle("Two-pass correction", value=True)
        if use_two_pass:
            st.caption("Pass 1: Fix ASR errors → Pass 2: Context refinement")

        temp_option = st.radio(
            "Temperature",
            ["Precise (0.2)", "Balanced (0.5)"],
            index=0,
            horizontal=True
        )
        llm_temperature = 0.2 if temp_option == "Precise (0.2)" else 0.5
        st.caption("Lower = more consistent, Higher = more creative")
    else:
        llm_model = None
        top_k = 1
        use_local_llm = False
        use_two_pass = False
        llm_temperature = 0.2

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

# Load ASR model
model_exists = check_model_exists(model_key, model_size)
loading_msg = f"Loading {model_size}..." if model_exists else f"Downloading & loading {model_size} (first time only)..."

with st.spinner(loading_msg):
    if model_key == "whisper":
        model = load_whisper_model(model_size)
        st.success(f"Whisper '{model_size}' ready!")
    else:
        mms_processor, mms_model = load_mms_model()
        st.success("Meta MMS ready!")

# Load local LLM if selected
local_llm_model = None
local_llm_tokenizer = None
if use_llm_correction and use_local_llm and llm_model:
    with st.spinner(f"Loading {llm_model}..."):
        local_llm_model, local_llm_tokenizer = load_local_llm(llm_model)
        st.success(f"Local LLM '{llm_model}' ready!")

def detect_repetition(text, min_pattern_len=2, max_pattern_len=20):
    """Detect if text contains excessive repetition (ASR hallucination)"""
    if not text or len(text) < 10:
        return False, text

    # Check for repeating patterns
    for pattern_len in range(min_pattern_len, min(max_pattern_len, len(text) // 3)):
        pattern = text[:pattern_len]
        if pattern * 3 in text:  # Pattern repeats at least 3 times
            repeat_count = text.count(pattern)
            if repeat_count > 5:  # Too many repetitions
                return True, pattern.strip()

    return False, text

def transcribe_audio(audio_data, file_ext=".wav"):
    """Transcribe audio using selected model"""
    import torch
    import numpy as np

    beam_width = top_k if use_llm_correction else 1

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        candidates = []

        if model_key == "whisper":
            lang_code = LANGUAGE_CODES[language]["whisper"]
            st.info(f"Transcribing with: **{model_family}** | Language: **{language}** | Beam: {beam_width}")

            segments, info = model.transcribe(
                tmp_path,
                language=lang_code,
                beam_size=beam_width,
                best_of=beam_width,
                temperature=0.0,
            )

            segment_texts = [segment.text for segment in segments]
            full_text = " ".join(segment_texts).strip()

            if beam_width > 1 and use_llm_correction:
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
            audio, sr = librosa.load(tmp_path, sr=16000)

            lang_code = LANGUAGE_CODES[language]["mms"]
            st.info(f"Transcribing with: **{model_family}** | Language: **{language}** | Beam: {beam_width}")

            mms_processor.tokenizer.set_target_lang(lang_code)
            mms_model.load_adapter(lang_code)

            inputs = mms_processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = mms_model(**inputs).logits

            if beam_width > 1 and use_llm_correction:
                candidates = get_mms_nbest(outputs, mms_processor, beam_width=beam_width)
            else:
                ids = torch.argmax(outputs, dim=-1)[0]
                full_text = mms_processor.decode(ids)
                candidates = [{"text": full_text, "score": 1.0}]

            full_text = candidates[0]["text"] if candidates else ""

        full_text = candidates[0]["text"] if candidates else ""

        # Check for repetition loop (common ASR hallucination)
        is_repetitive, cleaned_text = detect_repetition(full_text)
        if is_repetitive:
            st.warning(f"Detected repetitive ASR output (hallucination). Original started with: '{full_text[:50]}...'")
            full_text = cleaned_text
            # Also clean candidates
            candidates = [{"text": cleaned_text, "score": 1.0}]

        # Display results
        st.subheader("ASR Output")

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

            # Pass 1
            with st.spinner(f"Pass 1: Correcting ASR errors with {llm_model}..."):
                if use_local_llm and local_llm_model is not None:
                    corrected_text, error1 = correct_with_local_llm(
                        full_text, language, local_llm_model, local_llm_tokenizer,
                        n_best_candidates=candidate_texts, temperature=llm_temperature
                    )
                else:
                    corrected_text, error1 = correct_with_llm(
                        full_text, language, model_name=llm_model,
                        n_best_candidates=candidate_texts, temperature=llm_temperature
                    )

            if error1:
                st.warning(f"Pass 1 failed: {error1}")
                corrected_text = full_text

            # Pass 2
            final_text = corrected_text
            error2 = None
            if use_two_pass and corrected_text:
                with st.spinner(f"Pass 2: Context refinement with {llm_model}..."):
                    if use_local_llm and local_llm_model is not None:
                        final_text, error2 = refine_with_local_llm(
                            corrected_text, language, local_llm_model,
                            local_llm_tokenizer, temperature=llm_temperature
                        )
                    else:
                        final_text, error2 = refine_with_llm(
                            corrected_text, language, model_name=llm_model,
                            temperature=llm_temperature
                        )

                if error2:
                    st.warning(f"Pass 2 failed: {error2}")
                    final_text = corrected_text

            # Display
            if error1 and error2:
                st.subheader("Final Text")
                st.text_area("Output", full_text, height=150, key="final_output")
            else:
                st.subheader("LLM Corrected Output")
                st.text_area("Final transcription", final_text, height=150, key="corrected_output")

                with st.expander("Show correction steps"):
                    st.write("**Raw ASR:**", full_text)
                    if corrected_text != full_text:
                        st.write("**Pass 1 (ASR fix):**", corrected_text)
                    if use_two_pass and final_text != corrected_text:
                        st.write("**Pass 2 (Context):**", final_text)

                log_usage(st.session_state.get("user_email"), "transcription", {
                    "model": model_key, "language": language,
                    "llm_corrected": True, "two_pass": use_two_pass
                })
                return final_text
        else:
            st.subheader("Final Text")
            st.text_area("Output", full_text, height=150, key="final_output")

        log_usage(st.session_state.get("user_email"), "transcription", {
            "model": model_key, "language": language, "llm_corrected": False
        })
        return full_text

    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Main content
st.markdown(f"**Current settings:** {model_family} ({model_size}) | Language: {language}")

tab1, tab2 = st.tabs(["🎙️ Record", "📁 Upload File"])

with tab1:
    st.write("Click the microphone button to record your voice")

    try:
        audio_value = st.audio_input("Record audio", key="audio_recorder")

        if audio_value:
            audio_bytes = audio_value.getvalue()
            st.success(f"Recording captured: {len(audio_bytes) / 1024:.1f} KB")
            st.audio(audio_value)

            if st.button("Transcribe Recording", type="primary", key="transcribe_rec"):
                with st.spinner("Transcribing recording..."):
                    transcribe_audio(audio_bytes, ".wav")
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
st.caption("TsuurAI Batch Mode - Running on GPU")
