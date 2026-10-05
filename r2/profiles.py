"""Which original runner produced each canonical model's results (verified by hash, spec 2.2)."""
OPENAI_FC_MODELS = {
    "gpt-4.1-2025-04-14", "gpt-4.1-nano-2025-04-14",
    "gpt-4o-mini-2024-07-18", "gpt-5-nano-2025-08-07",
}
PROFILES = {
    "openai_fc": "src/runner.py",
    "gemini_inline7": "scripts/run_gemini_experiment.py",
    "deepseek_chat_inline7": "scripts/run_deepseek_experiment.py",
    "deepseek_reasoner_inline7": "scripts/run_deepseek_reasoner.py",
    "together_inline6": "scripts/run_together_experiment.py",
}


def profile_for(model):
    if model in OPENAI_FC_MODELS:
        return "openai_fc"
    if model == "gemini-2.5-flash-lite":
        return "gemini_inline7"
    if model == "deepseek-chat":
        return "deepseek_chat_inline7"
    if model == "deepseek-reasoner":
        return "deepseek_reasoner_inline7"
    return "together_inline6"
