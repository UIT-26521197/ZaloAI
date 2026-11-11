"""Local PhoWhisper and Qwen inference for LifeMate."""

import os
import wave
from time import monotonic


CONTEXT_LIMIT = 16_384
ACTION_TOKENS = 192


def remaining_seconds(deadline):
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("Per-sample time budget exhausted")
    return remaining


class LocalModels:
    def __init__(self, llm_path, asr_path, device, seed):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import numpy as np
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            AutoTokenizer,
            set_seed,
        )

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Enable a GPU in Colab or use --device cpu.")
        set_seed(seed, deterministic=True)
        dtype = torch.float32
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
        self.device = device
        self.dtype = dtype
        self.np = np
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, dtype=dtype, local_files_only=True
        ).to(device).eval()
        if asr_path is not None:
            self.processor = AutoProcessor.from_pretrained(asr_path, local_files_only=True)
            self.asr = AutoModelForSpeechSeq2Seq.from_pretrained(
                asr_path, dtype=dtype, local_files_only=True
            ).to(device).eval()

    def transcribe(self, audio_path, deadline):
        with wave.open(str(audio_path), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16_000):
                raise ValueError(f"Expected 16 kHz mono PCM16 WAV: {audio_path.name}")
            pcm = audio.readframes(audio.getnframes())
        samples = self.np.frombuffer(pcm, dtype="<i2").astype(self.np.float32) / 32768
        long_form = len(samples) > 30 * 16_000
        inputs = self.processor(
            samples, sampling_rate=16_000, return_tensors="pt",
            return_attention_mask=True, truncation=not long_form,
            padding="longest" if long_form else "max_length",
        )
        with self.torch.inference_mode():
            output = self.asr.generate(
                input_features=inputs.input_features.to(self.device, dtype=self.dtype),
                attention_mask=inputs.attention_mask.to(self.device),
                language="vi", task="transcribe", do_sample=False, num_beams=1,
                return_timestamps=long_form, max_new_tokens=256,
                max_time=remaining_seconds(deadline),
            )
        return self.processor.batch_decode(output, skip_special_tokens=True)[0].strip()

    def next_action(self, messages, deadline):
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)
        input_length = inputs.input_ids.shape[1]
        budget = min(ACTION_TOKENS, CONTEXT_LIMIT - input_length)
        if budget <= 0:
            raise ValueError("Prompt exceeds the 16,384-token context budget")
        with self.torch.inference_mode():
            output = self.llm.generate(
                **inputs, do_sample=False, max_new_tokens=budget,
                pad_token_id=self.tokenizer.pad_token_id,
                max_time=remaining_seconds(deadline),
            )
        return self.tokenizer.decode(output[0, input_length:], skip_special_tokens=True).strip()
