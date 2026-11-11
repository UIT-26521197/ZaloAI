"""Stage Drive-mounted model files to fast local storage once per runtime."""

import os
from pathlib import Path
import shutil


def stage_models(source, destination, report=print):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        return source
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)

    files = []
    for model_name in ("PhoWhisper-medium", "Qwen3-4B-Instruct-2507"):
        model_dir = source / model_name
        if not model_dir.is_dir():
            continue
        for base, dirs, names in os.walk(model_dir):
            dirs[:] = [name for name in dirs if name not in {".cache", "__pycache__", ".git"}]
            for name in names:
                if name.endswith((".lock", ".pyc", ".part")):
                    continue
                path = Path(base) / name
                if path.is_file():
                    files.append(path)
    files.sort()
    if not files:
        raise FileNotFoundError(f"No model files in {source}")

    missing_bytes = 0
    for path in files:
        target = destination / path.relative_to(source)
        size = path.stat().st_size
        if target.is_file() and target.stat().st_size == size:
            continue
        partial = target.with_name(target.name + ".part")
        copied = partial.stat().st_size if partial.exists() else 0
        missing_bytes += size - copied if copied <= size else size
    if shutil.disk_usage(destination).free < missing_bytes + 512 * 1024**2:
        report("Not enough local disk for model cache; loading directly from Drive.")
        return source

    for index, path in enumerate(files, 1):
        target = destination / path.relative_to(source)
        size = path.stat().st_size
        if target.is_file() and target.stat().st_size == size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        copied = partial.stat().st_size if partial.exists() else 0
        if copied > size:
            partial.unlink()
            copied = 0
        report(f"Caching model file {index}/{len(files)}: {path.name} "
               f"({copied / 1024**2:.0f}/{size / 1024**2:.0f} MiB)")
        with path.open("rb") as src, partial.open("ab") as dst:
            src.seek(copied)
            next_report = copied + 256 * 1024**2
            while copied < size:
                block = src.read(min(16 * 1024**2, size - copied))
                if not block:
                    raise IOError(f"Incomplete model file: {path}")
                dst.write(block)
                copied += len(block)
                if copied >= next_report or copied == size:
                    report(f"  {path.name}: {copied / 1024**2:.0f}/{size / 1024**2:.0f} MiB "
                           f"({copied / size:.0%})")
                    next_report = copied + 256 * 1024**2
            dst.flush()
            os.fsync(dst.fileno())
        partial.replace(target)
    report(f"Model cache ready: {destination}")
    return destination
