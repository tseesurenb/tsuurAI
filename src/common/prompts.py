"""
TsuurAI Prompt Management
Load and manage correction prompts
"""

from .config import PROMPTS_DIR, DOMAINS_DIR

def load_prompt(language, domains=None):
    """Load correction prompt from markdown file + domain prompts"""
    # Load main language prompt
    prompt_file = PROMPTS_DIR / f"{language.lower()}_correction.md"
    prompt = ""
    if prompt_file.exists():
        prompt = prompt_file.read_text()

    # Load all domain prompts
    if DOMAINS_DIR.exists():
        domain_prompts = []
        for domain_file in sorted(DOMAINS_DIR.glob("*.md")):
            domain_prompts.append(domain_file.read_text())
        if domain_prompts:
            prompt += "\n\n# Domain-Specific Knowledge\n" + "\n\n".join(domain_prompts)

    return prompt if prompt else None

def load_refinement_prompt(language):
    """Load context refinement prompt (pass 2)"""
    prompt_file = PROMPTS_DIR / f"{language.lower()}_context_refinement.md"
    if prompt_file.exists():
        return prompt_file.read_text()
    return None
