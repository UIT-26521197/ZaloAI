def load_qwen(source, device, offline):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    cuda = device.startswith("cuda")
    dtype = torch.float32
    if cuda:
        with torch.cuda.device(device):
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=offline)
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=dtype, local_files_only=offline,
    ).to(device)
    return tokenizer, model


class QwenModel:
    def __init__(self, source, device, offline, context_tokens, new_tokens, adapter=None):
        tokenizer, model = load_qwen(source, device, offline)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.context_tokens = context_tokens
        self.new_tokens = new_tokens

    def generate(self, messages, seconds: float):
        import torch
        from time import monotonic
        from ..core.types import ContextLimit

        started = monotonic()
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        size = inputs["input_ids"].shape[-1]
        if size + self.new_tokens > self.context_tokens:
            raise ContextLimit("Prompt exceeds Qwen context budget")
        remaining = seconds - (monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Qwen deadline elapsed during tokenization")
        with torch.inference_mode():
            result = self.model.generate(
                **inputs, do_sample=False, num_beams=1, max_new_tokens=self.new_tokens,
                max_time=remaining, pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(result[0, size:], skip_special_tokens=True).strip()
