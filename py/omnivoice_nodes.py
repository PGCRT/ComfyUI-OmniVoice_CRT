from typing import Any

import numpy as np

import torch

from .model_manager import (
    DEFAULT_MODEL_ID,
    get_model_cache_root,
    get_or_load_model,
    unload_all_models,
)


def _as_audio_dict(waveform: torch.Tensor, sample_rate: int) -> dict[str, Any]:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 3:
        raise RuntimeError(
            f"Unexpected waveform shape from OmniVoice: {tuple(waveform.shape)}"
        )
    return {
        "waveform": waveform.detach().cpu().float(),
        "sample_rate": int(sample_rate),
    }


def _audio_input_to_tuple(audio: dict[str, Any] | None):
    if not audio:
        return None
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0) or 0)
    if waveform is None or sample_rate <= 0:
        return None
    if waveform.dim() == 3:
        waveform = waveform[0]
    if waveform.dim() == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    return (waveform.detach().cpu().float(), sample_rate)


def _transcribe_ref_audio(model, ref_audio_obj):
    try:
        return model.transcribe(ref_audio_obj)
    except Exception as e:
        msg = str(e)
        if "return_timestamps=True" not in msg:
            raise

        waveform, sr = ref_audio_obj
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        audio_input = {
            "array": waveform.squeeze(0).cpu().numpy().astype(np.float32),
            "sampling_rate": int(sr),
        }
        result = model._asr_pipe(audio_input, return_timestamps=True)
        return (result.get("text") or "").strip()


def _compose_instruct(style_gender, style_age, style_pitch, style_accent) -> str:
    parts = []
    for token in (style_gender, style_age, style_pitch, style_accent):
        t = (token or "none").strip().lower()
        if t and t != "none":
            parts.append(t)
    return ", ".join(parts)


class OmniVoiceLoadModel:
    CATEGORY = "OmniVoice/Load"
    FUNCTION = "load"
    RETURN_TYPES = ("OMNIVOICE_PIPE",)
    RETURN_NAMES = ("pipe",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (
                    ["auto", "cuda", "cpu", "mps"],
                    {"default": "auto", "tooltip": "Model execution device"},
                ),
                "dtype": (
                    ["auto", "float16", "bfloat16", "float32"],
                    {"default": "auto", "tooltip": "Model tensor dtype"},
                ),
                "offload_after_generate": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Unload OmniVoice from memory after generation",
                    },
                ),
                "asr_model_name": (
                    [
                        "openai/whisper-large-v3-turbo",
                        "openai/whisper-medium",
                        "openai/whisper-small",
                    ],
                    {
                        "default": "openai/whisper-large-v3-turbo",
                        "tooltip": "Whisper model used when transcription override is empty",
                    },
                ),
            },
            "optional": {
                "reference audio (optional)": (
                    "AUDIO",
                    {"tooltip": "Reference speaker audio for voice clone"},
                ),
                "transcription override": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": True,
                        "tooltip": "Optional transcript for reference audio; ignored if no reference audio",
                    },
                ),
            },
        }

    def load(
        self,
        device,
        dtype,
        offload_after_generate,
        asr_model_name,
        **kwargs,
    ):
        root = get_model_cache_root("")
        ref_audio_obj = _audio_input_to_tuple(kwargs.get("reference audio (optional)"))
        clean_ref_text = (kwargs.get("transcription override") or "").strip()

        # If transcription override is connected but no reference audio is present,
        # ignore it safely as requested.
        if ref_audio_obj is None and clean_ref_text:
            clean_ref_text = ""

        needs_asr = ref_audio_obj is not None and not clean_ref_text
        key, model = get_or_load_model(
            model_id=DEFAULT_MODEL_ID,
            model_cache_root=root,
            device=device,
            dtype=dtype,
            load_asr=needs_asr,
            asr_model_name=asr_model_name,
        )

        if needs_asr:
            try:
                clean_ref_text = _transcribe_ref_audio(model, ref_audio_obj)
            except Exception as e:
                raise RuntimeError(
                    f"[OmniVoice][ERROR] Failed to auto-transcribe reference audio. {e}"
                ) from e

        pipe = {
            "key": key,
            "sample_rate": int(getattr(model, "sampling_rate", 24000) or 24000),
            "model_path": key.model_path,
            "device": key.device,
            "dtype": key.dtype,
            "asr_model_name": asr_model_name,
            "offload_after_generate": bool(offload_after_generate),
            "ref_audio": ref_audio_obj,
            "ref_text": clean_ref_text,
        }
        return (pipe,)


class OmniVoiceGenerate:
    CATEGORY = "OmniVoice/Process"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "status")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": (
                    "OMNIVOICE_PIPE",
                    {"tooltip": "Pipe output from OmniVoice Load Model"},
                ),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Hello from OmniVoice.",
                        "tooltip": "Target speech text",
                    },
                ),
                "language": (
                    [
                        "auto",
                        "English",
                        "Chinese",
                        "Japanese",
                        "Korean",
                        "French",
                        "Spanish",
                        "German",
                        "Portuguese",
                        "Russian",
                        "Arabic",
                        "Hindi",
                    ],
                    {
                        "default": "auto",
                        "tooltip": "Language hint for TTS; auto lets model infer language",
                    },
                ),
                "style_gender": (
                    ["none", "male", "female"],
                    {"default": "none", "tooltip": "no ref audio only"},
                ),
                "style_age": (
                    [
                        "none",
                        "child",
                        "teenager",
                        "young adult",
                        "middle-aged",
                        "elderly",
                    ],
                    {"default": "none", "tooltip": "no ref audio only"},
                ),
                "style_pitch": (
                    [
                        "none",
                        "very low pitch",
                        "low pitch",
                        "moderate pitch",
                        "high pitch",
                        "very high pitch",
                        "whisper",
                    ],
                    {"default": "none", "tooltip": "no ref audio only"},
                ),
                "style_accent": (
                    [
                        "none",
                        "american accent",
                        "australian accent",
                        "british accent",
                        "canadian accent",
                        "chinese accent",
                        "indian accent",
                        "japanese accent",
                        "korean accent",
                        "portuguese accent",
                        "russian accent",
                    ],
                    {"default": "none", "tooltip": "no ref audio only"},
                ),
                "num_step": (
                    "INT",
                    {
                        "default": 32,
                        "min": 4,
                        "max": 128,
                        "tooltip": "Diffusion sampling steps (higher = slower, often cleaner)",
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Classifier-free guidance strength",
                    },
                ),
                "t_shift": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Diffusion timestep shift; lower tends to favor low-SNR detail",
                    },
                ),
                "layer_penalty_factor": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Penalty encouraging earlier codebook layers to unmask first",
                    },
                ),
                "position_temperature": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Temperature for position selection during generation",
                    },
                ),
                "class_temperature": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": "Token class sampling temperature (0 = greedy)",
                    },
                ),
                "speed": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2147483647,
                        "tooltip": "Random seed for reproducibility",
                    },
                ),
                "use_duration": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable fixed output duration override",
                    },
                ),
                "duration": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 0.0,
                        "max": 120.0,
                        "step": 0.1,
                        "tooltip": "Target output duration in seconds (used only when enabled)",
                    },
                ),
                "postprocess_output": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Apply output postprocessing (trim/fade/pad cleanup)",
                    },
                ),
            }
        }

    def generate(
        self,
        pipe,
        text,
        language,
        style_gender,
        style_age,
        style_pitch,
        style_accent,
        num_step,
        guidance_scale,
        t_shift,
        layer_penalty_factor,
        position_temperature,
        class_temperature,
        speed,
        seed,
        use_duration,
        duration,
        postprocess_output,
    ):
        clean_text = (text or "").strip()
        if not clean_text:
            raise RuntimeError("[OmniVoice][ERROR] Text input is empty")

        if pipe is None or not isinstance(pipe, dict):
            raise RuntimeError(
                "[OmniVoice][ERROR] Connect 'OmniVoice Load Model' pipe output to this node"
            )

        model_path = (pipe.get("model_path") or "").strip()
        device = (pipe.get("device") or "auto").strip()
        dtype = (pipe.get("dtype") or "auto").strip()
        if not model_path:
            raise RuntimeError("[OmniVoice][ERROR] Invalid pipe: missing model_path")

        _, ov_model = get_or_load_model(
            model_id=DEFAULT_MODEL_ID,
            model_cache_root=get_model_cache_root(""),
            device=device,
            dtype=dtype,
            load_asr=False,
            asr_model_name=pipe.get("asr_model_name", "openai/whisper-large-v3-turbo"),
        )
        sample_rate = int(pipe.get("sample_rate", 24000))
        offload_after_generate = bool(pipe.get("offload_after_generate", False))
        ref_audio_obj = pipe.get("ref_audio")
        ref_text = (pipe.get("ref_text") or "").strip()

        kwargs: dict[str, Any] = {
            "text": clean_text,
            "num_step": int(num_step),
            "guidance_scale": float(guidance_scale),
            "t_shift": float(t_shift),
            "layer_penalty_factor": float(layer_penalty_factor),
            "position_temperature": float(position_temperature),
            "class_temperature": float(class_temperature),
            "speed": float(speed),
            "postprocess_output": bool(postprocess_output),
        }

        if language != "auto":
            kwargs["language"] = language.strip()

        if use_duration and duration > 0:
            kwargs["duration"] = float(duration)

        seed = int(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        clean_instruct = _compose_instruct(
            style_gender, style_age, style_pitch, style_accent
        )
        if ref_audio_obj is not None:
            mode = "voice_clone"
            kwargs["ref_audio"] = ref_audio_obj
            if ref_text:
                kwargs["ref_text"] = ref_text
        else:
            mode = "voice_design"
            kwargs["instruct"] = clean_instruct if clean_instruct else "male"

        audios = ov_model.generate(**kwargs)
        if not audios:
            raise RuntimeError("[OmniVoice][ERROR] OmniVoice returned no audio")

        output = _as_audio_dict(audios[0], sample_rate)
        status = (
            f"Generated {output['waveform'].shape[-1]} samples at {sample_rate} Hz "
            f"using mode={mode} | seed={seed}"
        )
        if offload_after_generate:
            unload_all_models()
            status += " | model offloaded"
            del ov_model
        return (output, status)
