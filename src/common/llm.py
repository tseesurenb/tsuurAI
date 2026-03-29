"""
TsuurAI LLM Correction
Functions for correcting ASR output using LLMs
"""

from openai import OpenAI
from .config import OPENAI_API_KEY
from .prompts import load_prompt, load_refinement_prompt

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def correct_with_local_llm(raw_text, language, model, tokenizer, n_best_candidates=None, temperature=0.2):
    """Use local LLM to correct ASR output"""
    import torch

    system_prompt = load_prompt(language)
    if not system_prompt:
        system_prompt = f"You are an expert {language} language corrector for speech recognition output."

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
        input_text = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "<|eot_id|>" in full_response:
            parts = full_response.split("assistant")
            if len(parts) > 1:
                corrected = parts[-1].strip()
            else:
                corrected = full_response.strip()
        elif "Assistant:" in full_response:
            corrected = full_response.split("Assistant:")[-1].strip()
        else:
            corrected = full_response.split(prompt)[-1].strip()

        corrected = corrected.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()

        return corrected, None

    except Exception as e:
        return raw_text, str(e)

def correct_with_llm(raw_text, language, model_name="gpt-4o", n_best_candidates=None, temperature=0.2):
    """Use OpenAI LLM to correct ASR output"""
    if not openai_client:
        return raw_text, "OpenAI key not found"

    system_prompt = load_prompt(language)
    if not system_prompt:
        system_prompt = f"You are an expert {language} language corrector for speech recognition output."

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
                temperature=temperature,
                max_tokens=1000
            )
        corrected = response.choices[0].message.content.strip()
        return corrected, None
    except Exception as e:
        return raw_text, str(e)

def refine_with_llm(text, language, model_name="gpt-4o", temperature=0.2):
    """Pass 2: Context refinement using OpenAI"""
    if not openai_client:
        return text, "OpenAI key not found"

    system_prompt = load_refinement_prompt(language)
    if not system_prompt:
        system_prompt = f"Review this {language} text and fix any words that don't fit the context."

    prompt = f"""{system_prompt}

## Text to refine:
{text}

## Refined text:"""

    try:
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
                temperature=temperature,
                max_tokens=1000
            )
        refined = response.choices[0].message.content.strip()
        return refined, None
    except Exception as e:
        return text, str(e)

def refine_with_local_llm(text, language, model, tokenizer, temperature=0.2):
    """Pass 2: Context refinement using local LLM"""
    import torch

    system_prompt = load_refinement_prompt(language)
    if not system_prompt:
        system_prompt = f"Review this {language} text and fix any words that don't fit the context."

    prompt = f"""{system_prompt}

## Text to refine:
{text}

## Refined text:"""

    try:
        input_text = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "<|eot_id|>" in full_response:
            parts = full_response.split("assistant")
            if len(parts) > 1:
                refined = parts[-1].strip()
            else:
                refined = full_response.strip()
        else:
            refined = full_response.split(prompt)[-1].strip()

        refined = refined.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
        return refined, None

    except Exception as e:
        return text, str(e)
