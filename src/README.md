# LifeMate với PhoWhisper, Laya và Qwen

`ZAI/src/` chứa codebase độc lập. Chạy các lệnh từ thư mục này. Baseline và inference dùng chung pipeline trong package Python `src`, và mỗi mẫu mở một ToolEnv của bộ dữ liệu.

1. PhoWhisper-medium chuyển âm thanh tiếng Việt thành transcript.
2. Laya multilingual chọn hướng gọi trực tiếp, tra cứu, hỏi lại hoặc xử lý chung. Nếu độ tin cậy thấp, pipeline chuyển hướng xử lý chung cho Qwen.
3. Qwen3-4B-Instruct-2507 nhận transcript gốc, hướng xử lý, schema và kết quả trước đó. Mỗi lượt sinh một tool call hoặc yêu cầu dừng.
4. Ứng dụng kiểm tra công cụ khả dụng và gọi `ToolEnv.validate()` trước khi thực thi. Submission lấy call log và state hash trực tiếp từ chính ToolEnv đó.

## Chuẩn bị runtime

Dùng Python 3.11 hoặc 3.12 trên Colab/MolAB và PyTorch tương thích với CUDA của runtime.

```bash
python -m pip install -r requirements.txt
```

Bố trí thư mục Drive:

```text
ZAI/
├── src/                               # thư mục chạy lệnh
│   └── src/                           # package Python
├── data/
│   ├── train/
│   │   ├── train.jsonl
│   │   ├── train_tools_catalog.json
│   │   ├── audio/
│   │   ├── train_environment/
│   │   └── zaic_tools/                 # gói chính thức
│   ├── public_test/                   # cùng bố cục theo tên split
│   └── private_test/                  # khi bộ dữ liệu được cung cấp
├── models/
│   ├── PhoWhisper-medium/              # vinai/PhoWhisper-medium
│   ├── laya-multilingual/              # convaiinnovations/laya-multilingual
│   └── Qwen3-4B-Instruct-2507/          # Qwen/Qwen3-4B-Instruct-2507
└── checkpoints/
    └── qwen/
```

Ba cấu hình mặc định dùng trọng số local. Đặt `ZAI_ROOT` hoặc truyền `--workspace` nếu clone code ngoài Drive. Có thể dùng `--data-root` để chỉ định thư mục chứa các split. Các model có thể trỏ đến thư mục tuyệt đối hoặc đường dẫn tương đối với workspace. Để tải model từ Hub khi phát triển, đặt `offline: false` và dùng tiền tố `hf:` trước ID model trong YAML.

## Chạy baseline, huấn luyện và inference

```bash
python -m src.baseline.run --limit 10
python -m src.training.trainer
python -m src.inference.run --split public_test
```

Có thể dùng `python -m scripts.run_baseline`, `python -m scripts.train`, `python -m scripts.run_inference` hoặc `python -m src.main <baseline|training|inference>` từ cùng thư mục. Các tệp Python trong `notebooks/` là cell ví dụ để chạy từ thư mục dự án.

Baseline ghi `submission.jsonl`, `trace.jsonl`, `metrics.json` và `errors.jsonl` vào `outputs/baseline/`. Metric khớp tool call so sánh nghiêm ngặt theo thứ tự. Metric khớp hash chỉ tính trên những dòng có gold hash. Đây là các metric chẩn đoán, điểm fuzzy chính thức do BTC tính.

Training dành riêng một phần tập train làm validation theo seed. Với từng gold call, training tạo prompt cùng định dạng inference rồi dùng ToolEnv của **tập train** để lấy kết quả cho bước kế tiếp. Loss chỉ áp dụng lên token đầu ra. Mỗi lượt lưu checkpoint, adapter tốt nhất theo validation loss, metrics và `validation_ids.json` vào một thư mục mới dưới `ZAI/checkpoints/qwen/`. PhoWhisper và Laya dùng checkpoint đã cấu hình.

Để thử adapter, điền `qwen_adapter` trong `configs/inference.yaml` bằng đường dẫn đến thư mục `adapter` của lượt train. So sánh hai pipeline trên cùng mẫu holdout:

```bash
python -m src.baseline.run --ids /path/to/validation_ids.json
python -m src.inference.run --split train --ids /path/to/validation_ids.json --output outputs/experiments/qwen_validation
```

Inference mặc định ghi `result/submission.jsonl`, mỗi dòng gồm `id`, `tool_calls` và `state_hash`. Thời gian mỗi mẫu được ghi trong `trace.jsonl`. Giới hạn thời gian trong pipeline được kiểm tra giữa các bước, còn thời gian thực tế phụ thuộc phần cứng và thời lượng âm thanh.
