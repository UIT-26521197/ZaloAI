from datetime import datetime, timezone
from uuid import uuid4


def new_run_directory(root):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"{stamp}-{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path
