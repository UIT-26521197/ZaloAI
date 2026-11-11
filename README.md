# LifeMate pretrained baseline

Mã chạy PhoWhisper-medium và Qwen3-4B-Instruct-2507 để đánh giá baseline. Chưa có bước huấn luyện hay tinh chỉnh.

## Bố cục Google Drive

```text
ZAI/
├── data/
│   ├── train/
│   └── public_test/
├── models/                    # chấp nhận tên model/
│   ├── PhoWhisper-medium/
│   └── Qwen3-4B-Instruct-2507/
├── src/
│   └── ZaloAI/                # repo sau khi bạn clone
├── runs/                      # tự tạo khi đánh giá train
└── result/                    # tự tạo khi chạy test
```

Trong `data/train/` cần có `train.jsonl`, `train_tools_catalog.json`, `zaic_tools/`, `train_environment/` và `audio/` đúng cấu trúc bộ dữ liệu. Mã suy luận truy cập trạng thái ứng dụng qua `ToolEnv`, không đọc `state.json` trực tiếp.

Sau khi bạn mount Drive trong Colab, tự tạo `ZAI/src` và clone repo vào đó. Đứng ở `ZAI/src` thì chạy:

```bash
python -u ZaloAI/baseline.py --limit 10
```

Mã tự tìm `ZAI/data` và `ZAI/models` theo vị trí repo. Một lệnh chạy nạp model **một lần** rồi xử lý lần lượt số mẫu bạn chọn. Kết quả train, progress, checkpoint và metrics được ghi trong `ZAI/runs/`. Bỏ `--limit` để chạy toàn bộ train. Bạn có thể chọn bất kỳ số mẫu nào bằng `--limit N`.

```bash
# Toàn bộ train có nhãn
python -u ZaloAI/baseline.py

# Tiếp tục lượt 10 mẫu sau khi bị ngắt
python -u ZaloAI/baseline.py --limit 10 --resume

# Tạo submission public test; không thể tính accuracy tại máy vì không có nhãn
python -u ZaloAI/baseline.py --split public_test

# Tiếp tục public test
python -u ZaloAI/baseline.py --split public_test --resume
```

Nếu đã `cd ZAI/src/ZaloAI`, bỏ tiền tố `ZaloAI/` trong các lệnh trên. Nếu bố cục Drive khác, dùng `--project-root /duong/dan/ZAI`.

## Tăng tốc nạp model trong Colab

Khi nhận ra Drive đã mount tại `/content/drive`, launcher tự sao chép hai model vào `/content/zai-model-cache`. Các lệnh tiếp theo trong **cùng runtime Colab** dùng bản cache này, tránh đọc lại trọng số từ Drive. Bản copy dở có thể tiếp tục; khi Colab cấp runtime mới, cache tạm sẽ phải tạo lại. Nếu ổ tạm không đủ chỗ, launcher dùng model trên Drive. Có thể chọn cache riêng bằng `--cache-dir /duong/dan/cache` hoặc tắt cache bằng `--no-cache-models`.

## Môi trường và metrics

Cần PyTorch có CUDA và các package trong `requirements.txt`. Có thể cài phần phụ thuộc bổ sung bằng `pip install -r ZaloAI/requirements.txt`. Nên dùng một GPU có ít nhất 24 GB VRAM cho hai model cùng lúc; 16 GB có thể thiếu khi prompt dài.

`state_hash_match_rate` là tỷ lệ trạng thái cuối khớp gold trên train. `exact_ordered_tool_calls_rate` là so khớp nghiêm ngặt theo thứ tự, không phải F1 fuzzy chính thức. Public/private test chỉ có dự đoán tại `ZAI/result/submission.jsonl` và metrics latency/status. Giữ nguyên dữ liệu, model, split, limit và đường dẫn khi dùng `--resume`.

Data, models, runs, result và `Đề ZAI.docx` được Git bỏ qua. Không thêm các tệp này vào commit.
