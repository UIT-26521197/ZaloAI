# Baseline LifeMate

Baseline dùng PhoWhisper-medium để nhận dạng tiếng Việt và Qwen3-4B-Instruct-2507 để chọn từng tool theo ngữ cảnh. Mỗi mẫu dùng một `ToolEnv` của BTC. Runner kiểm tra tham số bằng `validate` rồi mới `call`, lấy kết quả cho bước tiếp theo và ghi JSONL từ log cùng hash do môi trường trả về.

Runner đọc checkpoint cục bộ, giải mã greedy, giới hạn prompt và output ở tổng 16.384 token, và theo dõi ngân sách 30 giây mỗi mẫu. Giới hạn thời gian của `generate` được kiểm tra trong quá trình sinh token. Cần đo độ trễ trên GPU chấm thi. Cần xác nhận với BTC việc dùng giới hạn context ở runtime cho checkpoint Qwen gốc.

## Kiểm thử trên máy

```powershell
python -m unittest discover -s tests -v
```

Smoke test dùng ToolEnv và câu trả lời LLM giả lập. Nó kiểm tra lọc tool, phản hồi lỗi schema, tra cứu trước khi ghi, và JSONL cùng hash. Chạy với dữ liệu thật cần devkit LifeMate theo từng split.

## Chạy trên Colab

Mở `LifeMate_Colab.ipynb`, chọn GPU và bấm **Run all** để mount Drive, đồng bộ code từ GitHub và cài thư viện. Notebook dùng `MyDrive/ZAI/data`, `MyDrive/ZAI/models` và `MyDrive/ZAI/runs`.

### Lượt baseline đầu tiên trên train

Sau khi setup xong, thêm và chạy cell sau để replay nhãn rồi đánh giá baseline trên cùng các mẫu `train`.

```python
subprocess.run([
    sys.executable, str(REPO_DIR / "run_train_baseline.py"),
    "--data-root", str(DATA_DIR),
    "--models-dir", str(MODELS_DIR),
    "--runs-dir", str(RUNS_DIR),
], check=True)
```

Lệnh chạy toàn bộ `train`. Thêm `"--limit", "10"` vào danh sách đối số nếu muốn thử nhanh 10 mẫu. File `baseline-train.metrics.json` trong `runs/` ghi tỷ lệ khớp chuỗi tool calls, tỷ lệ khớp hash và độ trễ. Lượt chạy giới hạn thêm số mẫu vào tên các file.

### Chạy runner trực tiếp

Dùng Python và PyTorch GPU có sẵn trong Colab. PhoWhisper-medium phát hành trọng số dạng PyTorch `.bin`, cần PyTorch 2.6 trở lên với phiên bản Transformers đã ghim.

### Lấy code và cài thư viện

```python
import subprocess
import sys

if not REPO_DIR.exists():
    subprocess.run([
        "git", "clone", "--branch", "codex/lifemate-baseline",
        "https://github.com/UIT-26521197/ZaloAI.git", str(REPO_DIR),
    ], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], check=True)
subprocess.run([
    sys.executable, "-m", "pip", "install", "-r", str(REPO_DIR / "requirements-colab.txt")
], check=True)
```

Nếu repo đặt private, xác thực GitHub trong Colab trước khi clone. Khi cần cập nhật bản clone hiện có, chạy `git pull --ff-only` trong repo và ghi lại commit đã chạy.

### Chuẩn bị devkit và model

Chép hoặc giải nén devkit từ Drive vào ổ Colab. Đặt `DEVKIT` ở thư mục trực tiếp chứa `train/` và `public_test/`. Trong từng split, runner dùng manifest `<split>.jsonl`, catalog `<split>_tools_catalog.json`, `audio/` và package `zaic_tools/` tương ứng. Dữ liệu và model nằm trên Drive hoặc ổ Colab. Git lưu source và cấu hình.

```python
DEVKIT = DATA_DIR  # Hoặc thư mục chứa hai split trên ổ Colab.
ASR = MODELS_DIR / "PhoWhisper-medium"
LLM = MODELS_DIR / "Qwen3-4B-Instruct-2507"

subprocess.run(["hf", "download", "vinai/PhoWhisper-medium", "--local-dir", str(ASR)], check=True)
subprocess.run(["hf", "download", "Qwen/Qwen3-4B-Instruct-2507", "--local-dir", str(LLM)], check=True)
```

### Replay train rồi chạy baseline

```python
runner = [
    sys.executable, str(REPO_DIR / "lifemate_baseline.py"),
    "--data-root", str(DEVKIT), "--split", "train", "--limit", "30",
]
gold = RUNS_DIR / "gold-train-30.jsonl"
subprocess.run(runner + ["--replay-gold", "--output", str(gold)], check=True)
subprocess.run(runner + [
    "--asr", str(ASR), "--llm", str(LLM), "--gold-replay", str(gold),
    "--output", str(RUNS_DIR / "baseline-train-30.jsonl"),
], check=True)
```

Gửi `baseline-train-30.metrics.json` để xem tỷ lệ khớp hash, tỷ lệ khớp toàn bộ chuỗi tool calls và độ trễ. Tỷ lệ khớp chuỗi so sánh cả thứ tự và tham số. Điểm chính thức dùng F1 tool calls và các quy tắc fuzzy của BTC.

Dùng `--train-transcripts` trên cùng các mẫu train để đo chênh lệch khi thay transcript ASR bằng transcript có sẵn. Với public test, chọn `--split public_test`, bỏ `--gold-replay` và `--limit`, và đặt output là `result/submission.jsonl`.
