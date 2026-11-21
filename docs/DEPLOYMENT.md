# Triển khai realtime trên Kali Linux

## 1. Cài đặt

```bash
sudo apt update && sudo apt install -y python3-venv python3-dev libpcap-dev tcpdump
git clone <repo> network-ids && cd network-ids        # hoặc copy thư mục dự án
python3 -m venv ~/.venvs/nids && source ~/.venvs/nids/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Mô hình triển khai nằm trong `models/deployment/` (`bundle.joblib`, `bundle.json`). Có thể copy thư mục này từ máy huấn luyện, hoặc tạo lại:

```bash
python scripts/preprocess_dataset.py --stage all
python scripts/train.py --task multiclass --models rf xgb lgbm
python scripts/train_anomaly.py
python scripts/botnet_analysis.py          # tuỳ chọn: host-level botnet model
python scripts/build_bundle.py
```

## 2. Quyền bắt gói

Bắt gói cần quyền root hoặc capability `CAP_NET_RAW` + `CAP_NET_ADMIN`. Cách tối thiểu quyền (không chạy cả chương trình bằng root):

```bash
sudo setcap cap_net_raw,cap_net_admin=eip "$(readlink -f ~/.venvs/nids/bin/python3)"
```

Nếu không có quyền, chương trình dừng với thông báo `capture error: permission denied ...` thay vì treo.

## 3. Chạy

```bash
python scripts/run_realtime.py --list-interfaces
python scripts/run_realtime.py --interface eth0                      # chạy tới khi Ctrl+C
python scripts/run_realtime.py --interface eth0 --duration 600 --sqlite
python scripts/run_realtime.py --pcap lab_capture.pcap --speed 1.0    # replay đúng nhịp thời gian gốc
python scripts/run_realtime.py --pcap lab_capture.pcap --speed 0      # replay nhanh nhất có thể
```

Dừng an toàn: `Ctrl+C` (SIGINT) hoặc `SIGTERM` → dừng sniffer, xả toàn bộ flow đang mở, chạy phát hiện lần cuối, ghi `run_summary.json`.

Chạy nền như service:

```ini
# /etc/systemd/system/nids.service
[Unit]
Description=ML NIDS
After=network-online.target
[Service]
WorkingDirectory=/opt/network-ids
ExecStart=/root/.venvs/nids/bin/python scripts/run_realtime.py --interface eth0 --quiet --sqlite
Restart=on-failure
[Install]
WantedBy=multi-user.target
```

## 4. Cấu hình (`config/config.yaml`)

| Khoá | Ý nghĩa | Mặc định |
|---|---|---|
| `flow.flow_timeout_s` | Active timeout – flow dài hơn bị cắt (như CICFlowMeter) | 120 |
| `flow.idle_timeout_s` | Đóng flow không có gói mới | 60 |
| `flow.max_active_flows` | Giới hạn RAM: vượt quá → đóng flow cũ nhất | 200000 |
| `realtime.detection_interval_s` | Chu kỳ chấm điểm các flow đã đóng | 5 |
| `realtime.host_window_s` | Cửa sổ trượt cho đặc trưng host / beaconing / đồ thị | 300 |
| `realtime.queue_size` | Hàng đợi gói giữa sniffer và bộ xử lý; đầy → đếm `dropped` | 100000 |
| `realtime.bpf_filter` | Bộ lọc BPF của libpcap | `ip` |
| `classifier.uncertain_threshold` | τ: dưới ngưỡng → nhãn `Uncertain` | 0.6 |
| `botnet.*` | Tham số beaconing (số kết nối tối thiểu, ngưỡng CV, chu kỳ tối thiểu, port whitelist) | |

## 5. Đầu ra

* Console: một dòng cho mỗi cảnh báo (mức `LOW` trở lên).
* `logs/alerts/alerts.jsonl`: mỗi dòng một JSON theo schema `timestamp, src_ip, dst_ip, src_port, dst_port, protocol, detection_module, attack_type, confidence, severity, anomaly_score, reason, relevant_features`.
* `logs/alerts/alerts.csv`: lịch sử cảnh báo; `alerts.db` (SQLite) khi dùng `--sqlite`.
* `logs/alerts/run_summary.json`: số gói, gói lỗi, gói bị drop, số flow, độ trễ (p50/p95), thời gian suy luận theo lô, CPU, RAM.

Cảnh báo lặp lại cùng (src, dst, loại, module) trong 30 s được gộp và đếm, tránh hàng triệu dòng khi bị DoS.

## 6. Feature parity offline / realtime

| Hạng mục | Cách đảm bảo |
|---|---|
| Tên, thứ tự đặc trưng | `CIC_FEATURE_COLUMNS` duy nhất; classifier đọc đúng danh sách đã lưu trong bundle |
| Đơn vị | micro-giây cho thời gian, byte payload L4 cho độ dài – như CICFlowMeter |
| Logic gom flow | cùng class `FlowBuilder` (offline: `build_flows`; realtime: `RealtimeProcessor`) |
| Giải mã gói | cùng `decode_frame()`; Scapy chỉ cung cấp byte thô |
| Giá trị thiếu/Inf | cùng `FlowCleaner` đã fit trên train (lưu trong bundle) |
| Scaling | cùng `LogScaler` đã fit (Gaussian) – lưu trong bundle |
| Cửa sổ thời gian | cùng `host_features()` cho CTU-13, PCAP và realtime |

Kiểm thử `tests/test_pipeline_integration.py::test_realtime_replay_matches_offline` xác nhận số flow realtime (replay) bằng số flow offline trên cùng PCAP.

### 6.1 Kiểm chứng parity với CICFlowMeter trên PCAP gốc của CICIDS2018

Dùng chung code chưa đủ: các đặc trưng phải *bằng* giá trị CICFlowMeter đã tính khi tạo dữ liệu huấn luyện. Kiểm chứng (`scripts/fetch_cic_pcap_member.py`, `scripts/parity_check.py`, kết quả `reports/metrics/feature_parity_check.json`): tải riêng một file PCAP gốc trong archive 44.8 GB của ngày 02/03/2018 (đọc thư mục ZIP bằng HTTP Range) – máy 172.31.69.26, một nạn nhân Bot theo tài liệu CIC – rồi so sánh flow do dự án dựng với các dòng CSV nhãn Bot.

* Không ghép được từng dòng: CSV ngày này không có cột IP, timestamp dùng **đồng hồ 12 giờ không có AM/PM** (chỉ có giờ 01–12; dự án quy ước 01–07 là buổi chiều) và flow C&C lặp lại giống hệt nhau mỗi vài giây.
* So sánh theo "chữ ký kích thước" (số gói, tổng byte, độ dài header, TCP window đầu): cùng cách tách kết nối như CICFlowMeter – mỗi kết nối C&C thành một flow chính (3/4 gói, 326/129 byte, đóng bởi FIN) và một flow "đuôi" sau FIN (2/0 gói) – đúng hiện tượng *TCP appendix* đã được mô tả cho CICFlowMeter. Với cùng chữ ký, các đặc trưng **không khớp** là: mọi bộ đếm cờ (FIN/SYN/RST/PSH/ACK/CWE, Fwd/Bwd PSH), `Init Bwd Win Byts` (CICFlowMeter 219 so với 26883 của SYN-ACK), `Flow/Fwd/Bwd IAT Min`, `Pkt Len Mean/Var`. Kết hợp với việc trong CSV `SYN Flag Cnt` ≡ `Fwd PSH Flags` (r = 1), các cột cờ của bản CSV không thể tái tạo từ gói tin.
* **58/78 đặc trưng khớp** → `models/preprocessing/parity_safe_features.json`. Mô hình triển khai (`scripts/train_deployment.py`) chỉ dùng 24 đặc trưng = 30 đặc trưng đã chọn ∩ 58 đặc trưng khớp (bỏ `Pkt Len Mean`, `RST Flag Cnt`, `Flow IAT Min`, `Init Bwd Win Byts`, `PSH Flag Cnt`, `ACK Flag Cnt`).
* CICFlowMeter **không xuất flow 1 gói** (0 trong 1.78 triệu dòng) và **không có idle timeout** (chỉ cắt flow 120 s sau gói đầu). Dự án đặt `idle_timeout_s = flow_timeout_s = 120` và không chấm điểm flow < 2 gói ở mức flow (vẫn dùng ở mức host – SYN không phản hồi là tín hiệu quét).

Kết quả end-to-end trên PCAP nạn nhân Bot (classifier/mixture chỉ học từ CSV):

| Chỉ số | Trước khi sửa parity | Sau khi sửa |
|---|---|---|
| Flow C&C (TCP → 8080) phân loại Botnet | 99.99% | 99.99% (29,734/29,737) |
| Flow C&C bị Gaussian coi là bất thường | 100% | 0.3% |
| Flow còn lại (bình thường) bị Gaussian coi là bất thường | 99.3% | 16.6% (2,403 flow được chấm; 15,492 flow 1 gói không chấm) |
| Flow còn lại phân loại BENIGN | 96.8% | 99.0% |

Kết luận: bộ phân loại cây chuyển giao tốt từ CSV sang flow tự dựng; mô hình mật độ nhạy với sai khác nhỏ hơn nhiều, và ngay trên chính testbed vẫn báo động 16.6% thay vì 1% mục tiêu. **Trước khi tin cảnh báo `Unknown` trên một mạng thật, phải hiệu chỉnh lại ε** bằng traffic bình thường của mạng đó:

```bash
python scripts/calibrate_anomaly.py normal_traffic.pcap --target-fpr 0.01
```

(mô hình giữ nguyên, chỉ dịch ngưỡng ε tới phân vị yêu cầu; lịch sử hiệu chỉnh lưu trong `bundle.json`). Trên PCAP bài tập 2026 (mạng hoàn toàn khác, không nhãn), mixture chưa hiệu chỉnh đánh dấu 158/521 flow được chấm (30%).

## 7. Giới hạn đã biết

* Python + Scapy xử lý khoảng vài chục nghìn gói/giây trên một core; trên đường truyền tốc độ cao cần tăng `queue_size`, dùng BPF để lọc, hoặc thay Scapy bằng AF_PACKET/PF_RING. Gói bị drop được đếm và báo cáo, không bị bỏ qua âm thầm.
* Một flow chỉ được chấm điểm khi đã đóng (FIN/RST hoặc timeout) – đây là độ trễ cố hữu của phân loại mức flow (đo riêng là `flow_close_wait_s`).
* Mô hình học trên traffic CICIDS2018 (mạng mô phỏng của CIC/AWS năm 2018). Trên mạng khác, phân phối "bình thường" khác: cần hiệu chỉnh lại ε của Gaussian bằng traffic bình thường của chính mạng đó trước khi tin vào cảnh báo `Unknown`.
