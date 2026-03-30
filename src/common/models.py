"""
TsuurAI ASR Model Management
Load and manage speech recognition models
"""

import streamlit as st
from .config import MODELS_DIR, LOCAL_LLM_INFO

def check_model_exists(model_key, model_size):
    """Check if model is already downloaded"""
    if model_key == "whisper":
        whisper_cache = MODELS_DIR / "huggingface" / "hub" / f"models--Systran--faster-whisper-{model_size}"
        return whisper_cache.exists()
    else:  # MMS
        mms_cache = MODELS_DIR / "huggingface" / "hub" / "models--facebook--mms-1b-all"
        return mms_cache.exists()

@st.cache_resource
def load_whisper_model(model_id):
    """Load Whisper model with faster-whisper"""
    from faster_whisper import WhisperModel
    model = WhisperModel(
        model_id,
        device="cuda",
        compute_type="float16",
        download_root=str(MODELS_DIR / "huggingface")
    )
    return model

@st.cache_resource
def load_mms_model():
    """Load MMS model for ASR"""
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
def load_local_llm(model_name):
    """Load a local LLM with 4-bit quantization"""
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

    model_info = LOCAL_LLM_INFO[model_name]
    arch = model_info.get("arch", "causal")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    hf_cache = MODELS_DIR / "huggingface"

    if model_name == "Mongolian-Llama3":
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            model_info["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=str(hf_cache),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_info["model_id"],
            cache_dir=str(hf_cache),
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_info["model_id"],
            cache_dir=str(hf_cache),
        )
    elif arch == "seq2seq":
        tokenizer = AutoTokenizer.from_pretrained(
            model_info["model_id"],
            cache_dir=str(hf_cache),
            trust_remote_code=True,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_info["model_id"],
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=str(hf_cache),
            trust_remote_code=True,
        )
    else:
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

def get_mms_nbest(logits, processor, beam_width=5):
    """Get N-best candidates from MMS using CTC beam search"""
    import torch

    try:
        from pyctcdecode import build_ctcdecoder

        vocab_dict = processor.tokenizer.get_vocab()
        sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
        labels = [item[0] for item in sorted_vocab]

        decoder = build_ctcdecoder(labels)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs_np = log_probs.cpu().numpy()[0]

        beams = decoder.decode_beams(log_probs_np, beam_width=beam_width)

        candidates = []
        for beam in beams[:beam_width]:
            text = beam[0]
            score = beam[-1]
            candidates.append({"text": text, "score": float(score)})

        return candidates

    except Exception as e:
        ids = torch.argmax(logits, dim=-1)[0]
        text = processor.decode(ids)
        return [{"text": text, "score": 1.0}]
