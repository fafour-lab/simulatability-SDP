"""Qwen generation client for GPU-backed Transformers inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any


@dataclass
class GenerationConfig:
    model_id: str
    max_new_tokens: int = 1000
    temperature: float = 0.0
    top_p: float = 1.0
    dtype: str = "auto"
    device_map: str = "auto"
    local_files_only: bool = False
    require_cuda: bool = True
    cache_dir: str | None = None

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "GenerationConfig":
        return cls(
            model_id=str(obj["model_id"]),
            max_new_tokens=int(obj.get("max_new_tokens", 1000)),
            temperature=float(obj.get("temperature", 0.0)),
            top_p=float(obj.get("top_p", 1.0)),
            dtype=str(obj.get("dtype", "auto")),
            device_map=str(obj.get("device_map", "auto")),
            local_files_only=bool(obj.get("local_files_only", False)),
            require_cuda=bool(obj.get("require_cuda", True)),
            cache_dir=obj.get("cache_dir"),
        )


class QwenClient:
    """Thin wrapper around Qwen-8B generation on GPU."""

    def __init__(self, generation_cfg: GenerationConfig) -> None:
        self.cfg = generation_cfg
        self._tokenizer = None
        self._model = None

    def generate(self, system: str, user: str) -> str:
        if self._model is None or self._tokenizer is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._validate_local_model_id()
            if self.cfg.require_cuda and not torch.cuda.is_available():
                raise RuntimeError("Qwen local inference requires CUDA, but torch.cuda.is_available() is false.")

            self._print_load_settings()
            dtype = "auto"
            if self.cfg.dtype == "float16":
                dtype = torch.float16
            elif self.cfg.dtype == "bfloat16":
                dtype = torch.bfloat16
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.model_id,
                trust_remote_code=True,
                local_files_only=self.cfg.local_files_only,
                cache_dir=self.cfg.cache_dir,
            )
            device_map: str | dict[str, str] = self.cfg.device_map
            if self.cfg.device_map.lower() in {"cuda", "gpu"}:
                device_map = {"": "cuda"}
            self._model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_id,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
                local_files_only=self.cfg.local_files_only,
                cache_dir=self.cfg.cache_dir,
            )
            self._model.eval()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if hasattr(self._tokenizer, "apply_chat_template"):
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
		enable_thinking=False
            )
        else:
            text = f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"
        input_device = next(self._model.parameters()).device
        inputs = self._tokenizer([text], return_tensors="pt").to(input_device)
        do_sample = self.cfg.temperature > 0.0
        generation_kwargs = {
            **inputs,
            "max_new_tokens": self.cfg.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.cfg.temperature
            generation_kwargs["top_p"] = self.cfg.top_p
        outputs = self._model.generate(**generation_kwargs)
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def _validate_local_model_id(self) -> None:
        model_id = self.cfg.model_id
        if "$" in model_id or "{" in model_id or "}" in model_id:
            raise ValueError(
                "Qwen model path is not resolved. Set QWEN_MODEL_PATH to a local Qwen-8B checkpoint "
                "directory, or set qwen.model_id in the YAML config to an existing path. If the model "
                "should be cached/downloaded by Hugging Face, use qwen.model_id: Qwen/Qwen3-8B."
            )
        if model_id == "/path/to/local/Qwen3-8B" or model_id.startswith("/path/to/"):
            raise ValueError(
                "qwen.model_id is still the placeholder path. Replace it with a Qwen-8B checkpoint "
                "directory."
            )
        path = Path(model_id).expanduser()
        if path.is_absolute() or model_id.startswith(".") or model_id.startswith("~"):
            if not path.exists():
                raise FileNotFoundError(
                    f"Configured qwen.model_id does not exist: {path}. Set qwen.model_id to the actual "
                    "local Qwen-8B checkpoint directory, or export QWEN_MODEL_PATH before running step 4."
                )
            if not path.is_dir():
                raise NotADirectoryError(f"Configured qwen.model_id is not a directory: {path}")

    def _print_load_settings(self) -> None:
        cache_text = self.cfg.cache_dir if self.cfg.cache_dir else "default Hugging Face cache"
        mode_text = "cache only" if self.cfg.local_files_only else "cache first, download if missing"
        print(
            f"[QwenClient] Loading {self.cfg.model_id} on GPU ({mode_text}; cache_dir={cache_text})",
            file=sys.stderr,
            flush=True,
        )
