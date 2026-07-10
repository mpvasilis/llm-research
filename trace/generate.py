"""Stage 0: generate the answer locally with the open-data model itself.

Open-data models (OLMo, Pythia) are not hosted on any HF inference provider, so
to auto-produce the answer we run them locally via transformers. We use the 1B
OLMo 2 so CPU inference is viable. torch/transformers are imported lazily here
so the rest of the pipeline stays dependency-free.
"""
from functools import lru_cache
from config import GEN_MODELS, PROXY_MODELS, load_anthropic_key

_HAS_CHAT = {"olmo2": True, "pythia": False}


def _generate_claude(model_id: str, question: str, max_tokens: int) -> str:
    """Closed model: generate via the Anthropic API (needs ANTHROPIC_API_KEY)."""
    key = load_anthropic_key()
    if not key:
        raise ValueError(
            "Claude generation needs an API key. Set ANTHROPIC_API_KEY in .env, "
            "or paste a Claude answer manually instead."
        )
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=model_id, max_tokens=max_tokens,
        messages=[{"role": "user", "content": question}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


@lru_cache(maxsize=2)
def _load(model_id: str):
    import torch  # noqa: F401  (lazy, heavy)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    model.eval()
    return tok, model


def generate(model: str, question: str, max_new_tokens: int = 256,
             temperature: float = 0.7) -> str:
    """Generate the answer: Claude via API, open-data models locally."""
    model_id = GEN_MODELS[model]
    if model in PROXY_MODELS:  # closed model -> API
        return _generate_claude(model_id, question, max_new_tokens)
    tok, m = _load(model_id)

    if _HAS_CHAT.get(model):
        msgs = [{"role": "user", "content": question}]
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
    else:  # base model: plain prompt
        enc = tok(question, return_tensors="pt")

    prompt_len = enc["input_ids"].shape[-1]
    import torch
    with torch.no_grad():
        out = m.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
    return text.strip()
