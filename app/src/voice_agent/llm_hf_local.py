from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Optional, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "model" / "models" / "Qwen2.5-7B-Instruct-GPTQ-Int4"
)
MODEL_DIR = Path(os.getenv("HF_LOCAL_MODEL_DIR", os.getenv("AGENT_MODEL_DIR", str(_DEFAULT_MODEL_DIR))))

# Optional explicit GPTQ basename override (e.g., "model-GPTQ-4bit-128g")
_GPTQ_BASENAME = os.getenv("AGENT_GPTQ_BASENAME", "").strip()

_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[torch.nn.Module] = None
_loaded_via_gptq: bool = False

_MAX_NEW_TOKENS = int(os.getenv("AGENT_MAX_NEW_TOKENS", "96"))
_DO_SAMPLE = os.getenv("AGENT_DO_SAMPLE", "0").lower() in ("1", "true", "yes")
_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.7"))  # used only if _DO_SAMPLE
_TOP_P = float(os.getenv("AGENT_TOP_P", "0.9"))               # used only if _DO_SAMPLE
_TOP_K = int(os.getenv("AGENT_TOP_K", "0"))                   # 0 = disabled unless set
_REP_PENALTY = float(os.getenv("AGENT_REPETITION_PENALTY", "1.05"))

_DEVICE_MAP = os.getenv("AGENT_DEVICE_MAP", "auto")           # used for non-GPTQ path
_TORCH_DTYPE = os.getenv("AGENT_TORCH_DTYPE", "auto")         # "auto","float16","bfloat16"

def _resolve_dtype():
    m = _TORCH_DTYPE.lower()
    if m == "float16": return torch.float16
    if m == "bfloat16": return torch.bfloat16
    return "auto"

def _find_gptq_basename(folder: Path) -> Optional[str]:
    """Try to infer AutoGPTQ model_basename from safetensors files."""
    if _GPTQ_BASENAME:
        return _GPTQ_BASENAME
    if not folder.exists(): return None
    cands = sorted([p.stem for p in folder.glob("*.safetensors")])
    for stem in cands:
        if "gptq" in stem.lower():
            return stem
    # fallback: first .safetensors
    return cands[0] if cands else None

def preload() -> None:
    """
    Load tokenizer/model from local folder.
    Prefer AutoGPTQ on GPU; otherwise fall back to Transformers path.
    """
    global _tokenizer, _model, _loaded_via_gptq

    if _tokenizer is not None and _model is not None:
        return

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model dir not found: {MODEL_DIR}")

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True, local_files_only=True)
    if not getattr(tok, "chat_template", None):
        tok.chat_template = (
            "{% if messages[0]['role'] != 'system' %}"
            "{% set _=messages.insert(0, {'role':'system','content':'You are a helpful assistant.'}) %}"
            "{% endif %}{{ bos_token }}"
            "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}"
            "{% endfor %}"
        )

    loaded = False
    if torch.cuda.is_available():
        try:
            from auto_gptq import AutoGPTQForCausalLM 
            basename = _find_gptq_basename(MODEL_DIR)
            kwargs = dict(
                model_name_or_path=str(MODEL_DIR),
                model_basename=basename,            
                device="cuda:0",
                use_triton=False,                  
                trust_remote_code=True,
                use_safetensors=True,
            )
            model = AutoGPTQForCausalLM.from_quantized(**kwargs)
            model.eval()
            _model = model
            _loaded_via_gptq = True
            loaded = True
            print(f"[llm] AutoGPTQ loaded on CUDA (basename={basename})", flush=True)
        except Exception as e:
            print(f"[llm] AutoGPTQ GPU load failed ({e}); falling back.", flush=True)

    if not loaded:
        dtype = _resolve_dtype()
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            device_map=_DEVICE_MAP,
            torch_dtype=(None if dtype == "auto" else dtype),
            trust_remote_code=True,
            local_files_only=True,
        )
        model.eval()
        _model = model
        _loaded_via_gptq = False
        dev = "auto" if _DEVICE_MAP == "auto" else str(_model.device)
        print(f"[llm] Transformers path loaded (device_map={dev})", flush=True)

    _tokenizer = tok

def _qwen_eos_ids(tok: AutoTokenizer) -> list[int]:
    ids: list[int] = []
    if tok.eos_token_id is not None:
        if isinstance(tok.eos_token_id, list):
            ids.extend(tok.eos_token_id)
        else:
            ids.append(tok.eos_token_id)
    try:
        qwen_end = tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(qwen_end, int) and qwen_end >= 0 and qwen_end not in ids:
            ids.append(qwen_end)
    except Exception:
        pass
    # dedupe
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            out.append(i); seen.add(i)
    return out

def _build_inputs(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    assert _tokenizer is not None
    try:
        text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _tokenizer(text, return_tensors="pt")
    except Exception:
        joined = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            joined += f"{role.upper()}: {content}\n"
        joined += "ASSISTANT:"
        inputs = _tokenizer(joined, return_tensors="pt")

    # Move to model device
    if hasattr(_model, "device"):
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    return inputs

def complete(history: List[Dict[str, str]], max_new_tokens: Optional[int] = None) -> str:
    preload()
    assert _tokenizer is not None and _model is not None

    inputs = _build_inputs(history)

    gen: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens or _MAX_NEW_TOKENS,
        repetition_penalty=_REP_PENALTY,
        pad_token_id=_tokenizer.eos_token_id,
    )

    eos_ids = _qwen_eos_ids(_tokenizer)
    if eos_ids:
        gen["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]

    if _DO_SAMPLE:
        gen["do_sample"] = True
        gen["temperature"] = _TEMPERATURE
        gen["top_p"] = _TOP_P
        if _TOP_K > 0:
            gen["top_k"] = _TOP_K
    else:
        gen["do_sample"] = False
        beams = int(os.getenv("AGENT_BEAMS", "1"))
        if beams > 1:
            gen["num_beams"] = beams

    with torch.inference_mode():
        out = _model.generate(**inputs, **gen)

    prompt_len = inputs["input_ids"].shape[1]
    text = _tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)
    return text.strip()
