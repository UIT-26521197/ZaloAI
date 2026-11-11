"""Stage a bounded LifeMate train baseline from Google Drive in the marimo runtime.

This is setup code. The inference agent itself accesses app state only via ToolEnv.
"""

import json
from pathlib import Path, PurePosixPath


def prepare_baseline(drive_fs, repo, work, limit=10, chunk_mb=64):
    repo, work = Path(repo), Path(work)
    train, models = work / "data" / "train", work / "models"
    if not (repo / "evaluate.py").is_file():
        raise FileNotFoundError(repo / "evaluate.py")
    train.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    step = chunk_mb * 1024 * 1024

    def download(remote, local):
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        size = int(drive_fs.info(remote)["size"])
        if local.is_file() and local.stat().st_size == size:
            print(f"Cached {remote} ({size / 1024**2:.1f} MiB)", flush=True)
            return
        part = local.with_name(local.name + ".part")
        copied = part.stat().st_size if part.exists() else 0
        if copied > size:
            part.unlink()
            copied = 0
        print(f"Download {remote}: {copied / 1024**2:.1f}/{size / 1024**2:.1f} MiB", flush=True)
        with drive_fs.open(remote, "rb") as source, part.open("ab") as target:
            if copied:
                source.seek(copied)
            next_report = copied + step
            while copied < size:
                block = source.read(min(8 * 1024 * 1024, size - copied))
                if not block:
                    raise IOError(f"Download interrupted: {remote} ({copied}/{size})")
                target.write(block)
                copied += len(block)
                if copied >= next_report or copied == size:
                    print(f"  {local.name}: {copied / 1024**2:.1f}/{size / 1024**2:.1f} MiB ({copied / size:.0%})", flush=True)
                    next_report = copied + step
        part.replace(local)

    def folder(remote, local):
        root = PurePosixPath(remote)
        paths = [path for path in drive_fs.find(remote) if drive_fs.isfile(path)
                 and not any(piece in {".cache", "__pycache__"} for piece in PurePosixPath(path).parts)
                 and not path.endswith((".pyc", ".lock", ".metadata", ".DS_Store"))]
        if not paths:
            raise FileNotFoundError(remote)
        print(f"Stage {remote}: {len(paths)} files", flush=True)
        for i, path in enumerate(paths, 1):
            relative = PurePosixPath(path).relative_to(root)
            print(f"File {i}/{len(paths)}", flush=True)
            download(path, local.joinpath(*relative.parts))

    print("Stage 1/4: manifests and tools", flush=True)
    download("ZAI/data/train/train.jsonl", train / "train.jsonl")
    download("ZAI/data/train/train_tools_catalog.json", train / "train_tools_catalog.json")
    if not (train / "zaic_tools" / "__init__.py").is_file():
        folder("ZAI/data/train/zaic_tools", train / "zaic_tools")

    samples = []
    with (train / "train.jsonl").open(encoding="utf-8-sig") as manifest:
        for line in manifest:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) >= limit:
                break
    if len(samples) != limit:
        raise ValueError(f"Only {len(samples)} train rows available, requested {limit}")

    print(f"Stage 2/4: {limit} sample environments", flush=True)
    for i, sample in enumerate(samples, 1):
        sample_id = sample["id"]
        target = train / "train_environment" / sample_id
        if not target.is_dir() or not any(target.iterdir()):
            folder(f"ZAI/data/train/train_environment/{sample_id}", target)
        print(f"Environment {i}/{limit}: {sample_id}", flush=True)

    audio_paths = set()
    for sample in samples:
        for turn in [sample, *(sample.get("history") or [])]:
            if turn is not sample and turn.get("role") != "user":
                continue
            path = turn.get("audio_path") or turn.get("audio")
            if isinstance(path, dict):
                path = path.get("path")
            if isinstance(path, str):
                audio_paths.add(path)
    print(f"Stage 3/4: {len(audio_paths)} audio files", flush=True)
    for i, path in enumerate(sorted(audio_paths), 1):
        print(f"Audio {i}/{len(audio_paths)}", flush=True)
        download(f"ZAI/data/train/{path}", train / path)

    print("Stage 4/4: pretrained models", flush=True)
    for name in ("PhoWhisper-medium", "Qwen3-4B-Instruct-2507"):
        folder(f"ZAI/models/{name}", models / name)
    return {"repo": repo, "work": work, "train": train, "models": models,
            "limit": limit, "sample_ids": [s["id"] for s in samples]}
