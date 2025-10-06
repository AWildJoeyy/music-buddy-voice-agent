from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList


# -------------------- paths & knobs --------------------
_DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "model" / "models" / "Qwen2.5-7B-Instruct-GPTQ-Int4"
)
MODEL_DIR = Path(os.getenv("HF_LOCAL_MODEL_DIR", os.getenv("AGENT_MODEL_DIR", str(_DEFAULT_MODEL_DIR))))

_MAX_NEW_TOKENS = int(os.getenv("AGENT_MAX_NEW_TOKENS", "128"))
_DO_SAMPLE = os.getenv("AGENT_DO_SAMPLE", "0").lower() in ("1", "true", "yes")
_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.7"))
_TOP_P = float(os.getenv("AGENT_TOP_P", "0.9"))
_TOP_K = int(os.getenv("AGENT_TOP_K", "0"))
_REP_PENALTY = float(os.getenv("AGENT_REPETITION_PENALTY", "1.15"))

_DEVICE_MAP = os.getenv("AGENT_DEVICE_MAP", "auto")
_TORCH_DTYPE = os.getenv("AGENT_TORCH_DTYPE", "auto")  # auto|float16|bfloat16

_GPTQ_BASENAME_ENV = os.getenv("AGENT_GPTQ_BASENAME", "").strip()

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[torch.nn.Module] = None
_loaded_via_gptq: bool = False


# -------------------- helpers --------------------
def _resolve_dtype():
    m = _TORCH_DTYPE.lower()
    if m == "float16": return torch.float16
    if m == "bfloat16": return torch.bfloat16
    return "auto"

def _ensure_quantize_config(model_dir: Path) -> None:
    """Write a minimal config if missing (avoids unknown-key warnings)."""
    qpath = model_dir / "quantize_config.json"
    if qpath.exists():
        return
    cfg = {
        "bits": 4,
        "group_size": 128,
        "desc_act": False,
        "sym": True,
        "damp_percent": 0.01,
        "true_sequential": True
    }
    try:
        qpath.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"[llm] wrote default quantize_config.json → {qpath}")
    except Exception as e:
        print(f"[llm] warn: could not write quantize_config.json: {e}")

_SHARD_RE = re.compile(r"^(?P<base>.+)-00001-of-\d+$", re.I)

def _find_gptq_basename(folder: Path) -> Optional[str]:
    if _GPTQ_BASENAME_ENV:
        return _GPTQ_BASENAME_ENV
    cands = sorted([p.stem for p in folder.glob("*.safetensors")])
    if not cands:
        return None
    stem = cands[0]
    m = _SHARD_RE.match(stem)
    return (m.group("base") if m else stem)

def _qwen_eos_ids(tok: AutoTokenizer) -> List[int]:
    ids: List[int] = []
    if tok.eos_token_id is not None:
        if isinstance(tok.eos_token_id, list): ids.extend(tok.eos_token_id)
        else: ids.append(int(tok.eos_token_id))
    try:
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end >= 0:
            ids.append(im_end)
    except Exception:
        pass
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i); out.append(i)
    return out

def _ensure_qwen_chat_template(tok: AutoTokenizer) -> None:
    tpl = getattr(tok, "chat_template", None)
    if tpl and ("<|im_start|>" in str(tpl) or "im_start" in str(tpl)):
        return
    tok.chat_template = (
        "{{ bos_token }}"
        "{% for m in messages %}"
        "{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}"
        "{% endfor %}"
        "{{ '<|im_start|>assistant\\n' }}"
    )


# -------------------- load --------------------
def preload() -> None:
    """Load tokenizer/model once; prefer AutoGPTQ on CUDA, else Transformers."""
    global _tokenizer, _model, _loaded_via_gptq
    if _tokenizer is not None and _model is not None:
        return
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model dir not found: {MODEL_DIR}")

    _ensure_quantize_config(MODEL_DIR)

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True, local_files_only=True)
    _ensure_qwen_chat_template(tok)

    loaded = False
    if torch.cuda.is_available():
        try:
            from auto_gptq import AutoGPTQForCausalLM
            basename = _find_gptq_basename(MODEL_DIR)
            model = AutoGPTQForCausalLM.from_quantized(
                model_name_or_path=str(MODEL_DIR),
                model_basename=basename,
                device="cuda:0",
                use_triton=False,
                trust_remote_code=True,
                use_safetensors=True,
            )
            model.eval()
            _model = model
            _loaded_via_gptq = True
            loaded = True
            print(f"[llm] AutoGPTQ loaded on CUDA (basename={basename})")
        except Exception as e:
            print(f"[llm] AutoGPTQ CUDA path failed ({e}); falling back.")

    if not loaded:
        dtype = _resolve_dtype()
        _model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            device_map=_DEVICE_MAP,
            torch_dtype=(None if dtype == "auto" else dtype),
            trust_remote_code=True,
            local_files_only=True,
        ).eval()
        _loaded_via_gptq = False
        dev = "auto" if _DEVICE_MAP == "auto" else str(_model.device)
        print(f"[llm] Transformers path loaded (device_map={dev})")

    _tokenizer = tok


# -------------------- build inputs --------------------
def _build_inputs(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    assert _tokenizer is not None
    try:
        text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _tokenizer(text, return_tensors="pt")
    except Exception:
        # simple fallback
        joined = []
        for m in messages:
            joined.append(f"{m.get('role','user').upper()}:\n{m.get('content','')}\n")
        joined.append("ASSISTANT:\n")
        inputs = _tokenizer("".join(joined), return_tensors="pt")
    if hasattr(_model, "device"):
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    return inputs


# -------------------- generation --------------------
class _ChatMLStop(StoppingCriteria):
    def __init__(self, stop_ids: List[int]): self.stop_ids = set(stop_ids)
    def __call__(self, input_ids, scores, **kwargs): return int(input_ids[0, -1]) in self.stop_ids

def complete(history: List[Dict[str, str]], max_new_tokens: Optional[int] = None) -> str:
    preload()
    assert _tokenizer is not None and _model is not None

    inputs = _build_inputs(history)

    gen: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens or _MAX_NEW_TOKENS,
        pad_token_id=_tokenizer.eos_token_id,
        no_repeat_ngram_size=6,
        repetition_penalty=_REP_PENALTY,
        do_sample=_DO_SAMPLE,
    )
    if _DO_SAMPLE:
        gen["temperature"] = _TEMPERATURE
        gen["top_p"] = _TOP_P
        if _TOP_K > 0:
            gen["top_k"] = _TOP_K
    else:
        gen["do_sample"] = False

    stop_ids = _qwen_eos_ids(_tokenizer)
    stopping = StoppingCriteriaList([_ChatMLStop(stop_ids)]) if stop_ids else None
    if stop_ids:
        gen["eos_token_id"] = stop_ids if len(stop_ids) > 1 else stop_ids[0]

    with torch.inference_mode():
        out = _model.generate(**inputs, **({**gen, "stopping_criteria": stopping} if stopping else gen))

    prompt_len = int(inputs["input_ids"].shape[1])
    text = _tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    return text.strip()
