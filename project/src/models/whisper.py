from ..utils.audio import read_audio


class WhisperModel:
    def __init__(self, source: str, device: str, offline: bool):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(source, dtype=dtype, local_files_only=offline)
        processor = AutoProcessor.from_pretrained(source, local_files_only=offline)
        self.asr = pipeline(
            "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor, device=device, chunk_length_s=30,
        )

    def transcribe(self, path):
        result = self.asr(
            read_audio(path),
            generate_kwargs={"language": "vi", "task": "transcribe", "do_sample": False, "num_beams": 1},
        )
        return result["text"].strip()
