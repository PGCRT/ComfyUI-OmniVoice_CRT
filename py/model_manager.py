import gc
import os
import threading
from dataclasses import dataclass
from typing import Optional

import folder_paths
import torch


@dataclass(frozen=True)
class OmniVoiceModelKey:
    model_path: str
    device: str
    dtype: str


_MODEL_CACHE: dict[OmniVoiceModelKey, object] = {}
_CACHE_LOCK = threading.Lock()
DEFAULT_MODEL_ID = "k2-fsa/OmniVoice"


def _log(level: str, message: str) -> None:
    print(f"[OmniVoice][{level}] {message}")


def resolve_device(device: str) -> str:
    d = (device or "auto").strip().lower()
    if d != "auto":
        return d
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str, resolved_device: str) -> torch.dtype:
    d = (dtype or "auto").strip().lower()
    if d == "float32":
        return torch.float32
    if d == "bfloat16":
        return torch.bfloat16
    if d == "float16":
        return torch.float16

    if resolved_device.startswith("cuda"):
        return torch.float16
    if resolved_device == "mps":
        return torch.float16
    return torch.float32


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return "float32"


def _attention_candidates(resolved_device: str) -> list[str]:
    if resolved_device.startswith("cuda"):
        return ["sdpa", "eager"]
    return ["eager"]


def ensure_model(model_id: str, model_cache_root: str) -> str:
    model_id = (model_id or DEFAULT_MODEL_ID).strip()
    safe_name = model_id.replace("/", "--")
    local_dir = os.path.join(model_cache_root, safe_name)
    config_path = os.path.join(local_dir, "config.json")

    if os.path.isfile(config_path):
        return local_dir

    os.makedirs(model_cache_root, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        _log("INFO", f"Downloading model '{model_id}' to {local_dir}")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to download OmniVoice model from Hugging Face. "
            f"repo={model_id} error={e}"
        ) from e
    finally:
        pass

    if not os.path.isfile(config_path):
        raise RuntimeError(
            f"Model download finished but config.json was not found at: {config_path}"
        )

    return local_dir


def get_model_cache_root(custom_root: str = "") -> str:
    if custom_root and custom_root.strip():
        return os.path.normpath(custom_root.strip())
    return os.path.join(folder_paths.models_dir, "omnivoice")


def get_or_load_model(
    model_id: str,
    model_cache_root: str,
    device: str,
    dtype: str,
    load_asr: bool,
    asr_model_name: str,
):
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)
    model_path = ensure_model(model_id=model_id, model_cache_root=model_cache_root)

    key = OmniVoiceModelKey(
        model_path=os.path.normpath(model_path),
        device=resolved_device,
        dtype=_dtype_name(resolved_dtype),
    )

    with _CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            _log(
                "INFO",
                f"Loading OmniVoice model from {model_path} on {resolved_device} ({key.dtype})",
            )
            try:
                from omnivoice import OmniVoice
            except Exception as e:
                raise RuntimeError(
                    "Python package 'omnivoice' is not installed in this ComfyUI environment. "
                    "Install it with: python -m pip install --no-deps omnivoice>=0.1.0"
                ) from e

            load_error = None
            model = None
            selected_attention = None
            for candidate in _attention_candidates(resolved_device):
                try:
                    model = OmniVoice.from_pretrained(
                        model_path,
                        device_map=resolved_device,
                        dtype=resolved_dtype,
                        load_asr=False,
                        attn_implementation=candidate,
                    )
                    selected_attention = candidate
                    break
                except Exception as e:
                    load_error = e
                    _log(
                        "WARN",
                        f"Attention backend '{candidate}' failed, trying fallback",
                    )

            if model is None:
                raise RuntimeError(
                    f"Failed to load OmniVoice with available attention backends. Last error: {load_error}"
                )

            if selected_attention is not None:
                _log("INFO", f"Using attention backend: {selected_attention}")
            model._crt_attention_backend = selected_attention
            _MODEL_CACHE[key] = model

    if getattr(model, "_crt_attention_backend", None) is None:
        model._crt_attention_backend = "eager"

    if load_asr and getattr(model, "_asr_pipe", None) is None:
        _log("INFO", f"Loading ASR model '{asr_model_name}'")
        model.load_asr_model(model_name=asr_model_name)

    return key, model


def unload_all_models() -> int:
    with _CACHE_LOCK:
        count = len(_MODEL_CACHE)
        _MODEL_CACHE.clear()

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    return count
