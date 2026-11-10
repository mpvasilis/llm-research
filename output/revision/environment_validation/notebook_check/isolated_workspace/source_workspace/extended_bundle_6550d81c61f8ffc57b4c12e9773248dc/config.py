"""Central config: token loading, corpus indexes, instruction dataset ids."""
import os
from pathlib import Path

ROOT = Path(__file__).parent


def _from_env_or_dotenv(key: str) -> str:
    v = os.environ.get(key, "").strip()
    if v:
        return v
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def load_token() -> str:
    """HF read token from env var, else from .env file. Never hardcoded."""
    tok = _from_env_or_dotenv("HF_TOKEN")
    if tok:
        return tok
    raise SystemExit(
        "No HF token. Set HF_TOKEN env var or copy .env.example -> .env and fill it in."
    )


def load_anthropic_key() -> str:
    """Anthropic API key for optional Claude generation; '' if absent."""
    return _from_env_or_dotenv("ANTHROPIC_API_KEY")


# --- Pretraining corpora (infini-gram indexes) ---------------------------
# https://infini-gram.io  — full list of indexes in their docs.
INFINIGRAM_API = "https://api.infini-gram.io/"
PRETRAIN_INDEXES = {
    # model family -> infini-gram index name (corpus the model was trained on).
    # OLMo-2 (both the 1124 and 0425 lines) was pretrained on OLMo-Mix-1124
    # (Stage 1) + Dolmino-Mix-1124 (Stage 2 mid-training/anneal) -- NOT Dolma v1.7
    # (a distinct earlier OLMo-1.7-era corpus). infini-gram hosts a live index for
    # the Stage-1 corpus; Stage-2 Dolmino-Mix-1124 has NO standalone index, so a
    # complete two-stage pretraining trace is not achievable by a single index.
    "olmo2": "v4_olmo-mix-1124_llama",   # OLMo-2 pretraining Stage 1 (OLMo-Mix-1124)
    "pythia": "v4_piletrain_llama",      # Pythia pretraining = The Pile
    "claude": "v4_olmo-mix-1124_llama",  # PROXY: Claude's data is private; an open
                                          # web corpus standing in for "the public
                                          # web Claude likely also saw".
}

# Models whose training data is NOT public: the corpus above is a *plausible
# source* proxy, not actual provenance. Reports/UI flag this loudly.
PROXY_MODELS = {"claude"}

# --- Instruction-tuning (SFT) datasets ------------------------------------
# OLMo-2 Instruct used model-SPECIFIC Tulu-3 SFT mixtures, NOT the generic
# allenai/tulu-3-sft-mixture: the 1124-7B line uses tulu-3-sft-olmo-2-mixture and
# the 0425-1B line uses the reduced/re-decontaminated tulu-3-sft-olmo-2-mixture-0225.
# oasst1 is already an internal subset (~7,132 prompts) of every Tulu-3 mixture, so
# it is NOT searched separately (doing so double-counts those prompts).
DATASETS_SERVER = "https://datasets-server.huggingface.co"

# Per released model id -> the SFT mixture that model was actually trained on.
SFT_BY_MODEL = {
    "allenai/OLMo-2-1124-7B-Instruct":  "allenai/tulu-3-sft-olmo-2-mixture",
    "allenai/OLMo-2-1124-13B-Instruct": "allenai/tulu-3-sft-olmo-2-mixture",
    "allenai/OLMo-2-0425-1B-Instruct":  "allenai/tulu-3-sft-olmo-2-mixture-0225",
}

# Post-SFT stages (all public/ungated): DPO preference mixes and RLVR datasets,
# plus the intermediate weight checkpoints for a stagewise Base->SFT->DPO->
# RLVR->Instruct behavioral eval (no retraining needed).
DPO_BY_MODEL = {
    "allenai/OLMo-2-1124-7B-Instruct": "allenai/olmo-2-1124-7b-preference-mix",
    "allenai/OLMo-2-0425-1B-Instruct": "allenai/olmo-2-0425-1b-preference-mix",
}
RLVR_BY_MODEL = {
    "allenai/OLMo-2-1124-7B-Instruct": ["allenai/RLVR-GSM"],
    "allenai/OLMo-2-0425-1B-Instruct": ["allenai/RLVR-MATH",
                                        "allenai/RLVR-GSM-MATH-IF-Mixed-Constraints"],
}
STAGE_CHECKPOINTS = {
    "1B": {"base": "allenai/OLMo-2-0425-1B", "sft": "allenai/OLMo-2-0425-1B-SFT",
           "dpo": "allenai/OLMo-2-0425-1B-DPO", "rlvr": "allenai/OLMo-2-0425-1B-RLVR1",
           "instruct": "allenai/OLMo-2-0425-1B-Instruct"},
    "7B": {"base": "allenai/OLMo-2-1124-7B", "sft": "allenai/OLMo-2-1124-7B-SFT",
           "dpo": "allenai/OLMo-2-1124-7B-DPO",
           "instruct": "allenai/OLMo-2-1124-7B-Instruct"},
}

INSTRUCT_DATASETS = {
    # model family -> list of (dataset_id, config, split, text_column_hint).
    # Default olmo2 entry = the 7B-line mixture; per-model selection uses SFT_BY_MODEL.
    "olmo2": [
        ("allenai/tulu-3-sft-olmo-2-mixture", "default", "train", "messages"),
    ],
    "pythia": [
        ("OpenAssistant/oasst1", "default", "train", "text"),
    ],
    # proxy: public instruction data resembling RLHF-style assistant behavior
    "claude": [
        ("allenai/tulu-3-sft-olmo-2-mixture", "default", "train", "messages"),
    ],
}

# --- Stage 0: local generation (open-data models aren't on HF inference) ------
# These run locally via transformers. Kept small (1B) so CPU inference is viable.
GEN_MODELS = {
    "olmo2": "allenai/OLMo-2-0425-1B-Instruct",
    "pythia": "EleutherAI/pythia-1.4b",  # base model, no chat template
    "claude": "claude-haiku-4-5-20251001",  # via Anthropic API (needs key)
}

DEFAULT_MODEL = "olmo2"
