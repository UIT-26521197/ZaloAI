# Khảo sát dữ liệu và thiết kế input/output cho LifeMate

Khảo sát toàn bộ manifest `train` và `public_test`, hai catalog và header của tất cả WAV lượt cuối trong devkit tải về. Các số liệu dưới đây mô tả dữ liệu, chưa phải kết quả chạy model.

| Chỉ số | `train` | `public_test` |
|---|---:|---:|
| Mẫu | 1.545 | 634 |
| Mẫu có lịch sử | 1.018 (65,9%) | 419 (66,1%) |
| Audio lượt user trong lịch sử | 3.491 | 1.299 |
| Thời lượng audio lượt cuối, trung vị / p95 | 6,94 / 11,84 giây | 7,21 / 13,47 giây |
| App / tool trong catalog | 46 / 147 | 36 / 122 |
| Transcript lượt cuối và nhãn tool call | Có | Không |

Tất cả WAV lượt cuối đều tồn tại, mono 16 kHz. Lịch sử dài nhất có 24 lượt ở cả hai split. Trong `train`, 1.065 mẫu (68,9%) cần ít nhất hai tool call, 715 mẫu (46,3%) dùng tool từ nhiều app, 217 mẫu gọi `sys.ask_missing_info`, và 111 mẫu có chuỗi gọi rỗng. Catalog `public_test` có 34 tên tool vắng trong catalog `train`.

## Đầu vào và đầu ra của từng bước

| Thành phần | Input | Output |
|---|---|---|
| PhoWhisper-medium | WAV lượt cuối và WAV của từng lượt user trước đó, giữ thứ tự và vai trò | Transcript tiếng Việt theo từng lượt. Giữ số, tên riêng, thời gian và địa chỉ vì các trường này đi vào tham số tool. |
| Lọc tool theo mẫu | `context.installed_apps` và catalog của split đang chạy | Schema của các tool được phép, kèm `sys.ask_missing_info`. Giữ mô tả, kiểu dữ liệu, trường bắt buộc và tham chiếu `x-ref`. |
| Qwen 4B | Transcript lượt cuối, lịch sử có vai trò, `current_time`, `timezone`, vị trí, hồ sơ, schema tool và phản hồi từ các tool đã gọi trong lượt hiện tại | Một JSON mỗi bước, dạng `{"name":"<tool>","arguments":{...}}` hoặc `{"done":true}`. |
| Runner + `ToolEnv` | Tên tool và tham số Qwen đề xuất | Kết quả hoặc lỗi đưa lại cho Qwen. Khi kết thúc, lấy `id`, `tool_calls`, `state_hash` để ghi JSONL. |

Ví dụ, với yêu cầu “Bật điều hòa phòng khách rồi đặt xe về nhà”, Qwen đọc schema của cả hai app, gọi tool tra thiết bị để lấy ID, dùng ID đó ở bước bật điều hòa và dùng vị trí hiện tại cùng địa chỉ nhà để đặt xe. Nếu thiếu tham số mà ngữ cảnh và tool tra cứu không cung cấp được, gọi `sys.ask_missing_info` với đúng schema. Lịch sử chỉ cung cấp ngữ cảnh vì các hành động trước đã được áp dụng. Mỗi mẫu dùng một `ToolEnv` và agent lấy dữ liệu app qua tool của môi trường.

## Thiết kế dữ liệu train cho từng model

1. **ASR.** Đo PhoWhisper trên audio validation có `audio_text`. Ngoài WER, kiểm tra lỗi ở số tiền, ngày giờ, tên riêng và địa chỉ. Fine-tune ASR nếu các lỗi này thực sự kéo điểm tool call xuống. Nếu fine-tune, ghép từng WAV với transcript đúng lượt và chia tập theo mẫu trước khi tách WAV lịch sử.
2. **Qwen.** Nếu fine-tune, chuyển mỗi mẫu train thành chuỗi quyết định. Ở bước i, input gồm ngữ cảnh, schema và các cặp call/kết quả từ bước 1 đến i−1. Target là `label.tool_calls[i]`. Thêm một bước `done` sau call cuối. Với nhãn rỗng, target đầu tiên là `done`. Dùng `label.tool_results` làm phản hồi trong chuỗi train. Khi chạy validation và public test, phản hồi đến từ `ToolEnv` thật. Giữ schema trong prompt để model có thể đọc cả những tool mới xuất hiện ở public test.
3. **Router tùy chọn.** Ở lượt đầu nên giữ bộ lọc `installed_apps` hiện có. Nếu cần giảm token hoặc độ trễ, thử router chạy local xếp hạng schema khả dụng bằng transcript và ngữ cảnh. Dùng toàn bộ app xuất hiện trong chuỗi nhãn làm tín hiệu đa nhãn. Đo tỷ lệ giữ đủ tool cho cả chuỗi và ưu tiên mở rộng danh sách khi router thiếu chắc chắn. Với tool mới, xếp hạng theo mô tả schema giúp router tiếp nhận được tool ngoài tập nhãn cố định.

## Thứ tự thử nghiệm

1. Giữ Qwen 4B làm baseline. Tách validation từ `train` trước khi tạo dữ liệu theo bước. Cố gắng giữ các nhóm không gọi tool, thiếu thông tin, nhiều bước và nhiều app trong validation.
2. Chạy cùng mẫu với `--train-transcripts` và với audio thật để đo phần suy giảm do ASR. Replay nhãn để lấy hash tham chiếu vì manifest `train` cung cấp `final_state` nhưng không có `state_hash` trong `label`.
3. Nếu phần lớn lỗi vẫn xảy ra với transcript chuẩn, ưu tiên cải thiện prompt hoặc fine-tune Qwen theo từng bước. Nếu độ trễ tập trung ở lịch sử dài, đo riêng thời gian ASR và Qwen trước khi thử rút gọn ngữ cảnh.
4. Thử model 1–2B hoặc router sau khi có mốc 4B. So sánh bằng F1 tool calls nếu có bộ chấm phù hợp, tỷ lệ khớp hash, tỷ lệ khớp toàn bộ chuỗi và độ trễ p95 đầu cuối. Kiểm tra riêng mẫu nhiều app, lịch sử dài và tool ít gặp. Chạy `public_test` với cấu hình đã chọn để sinh bài nộp.

Tham chiếu triển khai trong `lifemate_baseline.py`, `lifemate_models.py` và hướng dẫn Colab trong `README.md`.
