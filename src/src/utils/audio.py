from pathlib import Path


def audio_path(data_root: Path, split: str, raw: str) -> Path:
    split_dir = data_root / split
    audio_root = (split_dir / "audio").resolve()
    path = Path(raw)
    for base in (data_root, split_dir, audio_root):
        candidate = (base / path).resolve()
        if candidate.is_relative_to(audio_root) and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Audio outside {audio_root} or missing: {raw}")


def read_audio(path: Path):
    import soundfile
    from math import gcd
    from scipy.signal import resample_poly

    waveform, rate = soundfile.read(path, dtype="float32", always_2d=True)
    if len(waveform) == 0:
        raise ValueError(f"Empty audio: {path}")
    mono = waveform.mean(axis=1)
    if rate != 16000:
        factor = gcd(rate, 16000)
        mono = resample_poly(mono, 16000 // factor, rate // factor)
    return {"raw": mono, "sampling_rate": 16000}
