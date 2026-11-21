# Web demo

Giao diện web cho hệ thống NIDS. Web **không có mô hình riêng**: mọi phân tích đi qua đúng mã của dự án
(`pcap_parser` → `FlowBuilder` → `feature_extractor` → `DetectionPipeline` với bundle triển khai
`models/deployment`: LightGBM 24 đặc trưng + Gaussian mixture K = 16 + mô hình host CTU-13).

## Chạy

```powershell
# Windows (từ thư mục network-ids)
& "$env:USERPROFILE\.venvs\nids\Scripts\python.exe" -m pip install fastapi "uvicorn[standard]" python-multipart httpx
& "$env:USERPROFILE\.venvs\nids\Scripts\python.exe" web/make_samples.py          # tạo mẫu demo (một lần)
& "$env:USERPROFILE\.venvs\nids\Scripts\python.exe" -m uvicorn web.app:app --host 127.0.0.1 --port 8000
```

Mở <http://127.0.0.1:8000>. Lần đầu mỗi mẫu được phân tích rồi lưu vào `web/cache/`, các lần sau mở ngay.
Kết quả cache tự bị bỏ khi bundle mô hình được dựng lại.

```bash
# Kali Linux – bật thêm bắt gói trực tiếp trên card mạng
source ~/.venvs/nids/bin/activate
sudo -E NIDS_ALLOW_LIVE=1 $(which python) -m uvicorn web.app:app --host 0.0.0.0 --port 8000
```

Bắt gói cần quyền root hoặc `CAP_NET_RAW`/`CAP_NET_ADMIN` (xem [DEPLOYMENT.md](DEPLOYMENT.md)). Khi không đặt
`NIDS_ALLOW_LIVE=1`, chế độ "Card mạng" bị tắt – web chỉ đọc file và phát lại PCAP.
Chỉ mở `--host 0.0.0.0` trong mạng lab: web không có đăng nhập.

## Các trang

| Trang | Làm gì | Mã |
|---|---|---|
| Tổng quan | Kiến trúc pipeline, thông số bundle, lối tắt tới các mẫu | `GET /api/info`, `/api/samples` |
| Phân tích lưu lượng | Tải lên `.pcap/.pcapng/.csv` (≤ 300 MB, ≤ 3 triệu gói) hoặc chọn mẫu → số flow, dự đoán theo họ, dòng thời gian, cảnh báo gộp (bấm để xem lý do và đặc trưng), IP nguồn, điểm botnet theo host (cửa sổ 300 s), beaconing, topology, bảng flow. CSV có cột `Label` (tên nhãn CICIDS2018) hoặc `Family` → ma trận nhầm lẫn, Macro-F1, FPR, FNR | `POST /api/analyze/upload`, `POST /api/analyze/sample/{id}`, `GET /api/jobs/{id}` |
| Thử một flow | Lấy ngẫu nhiên một flow thật của **tập test** theo họ, sửa đặc trưng, xem xác suất 7 lớp, đóng góp SHAP (TreeSHAP gốc của LightGBM), điểm −log p(x) so với ngưỡng ε, và cảnh báo mà bộ quyết định sinh ra | `GET /api/flow/random`, `POST /api/flow/predict` |
| Giám sát realtime | Phát lại PCAP qua `RealtimeProcessor` (×10, ×50, ×200, tối đa) hoặc bắt gói trực tiếp; số gói/flow/cảnh báo theo thời gian, luồng cảnh báo | WebSocket `/ws/realtime` |
| Kết quả mô hình | Đọc `reports/`: validation 7 lớp và nhị phân, test độc lập, chia theo thời gian, ablation, detector bất thường, LOFO, CTU-13, mô hình triển khai, toàn bộ hình | `GET /api/results` |

## Mẫu demo (`web/samples/samples.json`)

| id | Nội dung | Kết quả mong đợi |
|---|---|---|
| `synthetic` | Tổng hợp, biết trước đáp án: 5 máy duyệt web/DNS bình thường, 192.168.10.66 quét 1.024 cổng, 192.168.10.67 quét 120 host cổng 445, 192.168.10.7 gọi về 185.199.10.9:8080 mỗi 30 s | PortScan, HostSweep, Beaconing (CV = 0) và flow C&C bị phân loại Botnet; web bình thường không bị cảnh báo |
| `cic_bot` | PCAP gốc CICIDS2018 02/03/2018, máy nạn nhân Bot 172.31.69.26 | 29.734 / 29.737 flow C&C → Botnet (khớp mục 4.10 báo cáo) |
| `ctu_neris` | CTU-13 kịch bản 1, phút 74–94 (bot bắt đầu hoạt động ở phút 75), chỉ gói của các host có nhãn | Mô hình host cho 147.32.84.165 điểm botnet 1,0. Ở mức flow có báo nhầm (DoS/Unknown) trên mạng 2011 – minh hoạ dịch chuyển phân phối |
| `cic_csv` | 9.128 flow phân tầng của tập test (≤ 1.500 dòng/họ), có cột `Family` | Macro-F1 ≈ 0,885, FPR ≈ 1,3%; Infiltration recall ≈ 0,41 |
| `exercise` | PCAP bài tập phân tích traffic (mạng khác, không nhãn) | Ít tấn công; GMM báo "Unknown" nhiều hơn 1% → cần `scripts/calibrate_anomaly.py` |

Dựng lại mẫu: `python web/make_samples.py` (cần PCAP CTU-13 theo `config.yaml`; thiếu thì mẫu đó bị bỏ qua).

## Ghi chú thiết kế

* Phân tích chạy trong luồng nền; trình duyệt hỏi `/api/jobs/{id}` để cập nhật tiến độ.
* Khi có rất nhiều flow bị gắn cờ, chỉ 1.500 flow (lấy đều theo họ) được dựng thành cảnh báo đầy đủ có giải thích.
  Các số đếm và biểu đồ vẫn tính trên toàn bộ flow.
* Phân tích host offline dùng cửa sổ 300 s cho mô hình host và luật quét (giống lúc huấn luyện trên CTU-13).
  Beaconing và topology dùng cả capture, vì chu kỳ C&C 120 s cần hơn 300 s mới đủ 8 kết nối.
* Realtime phát lại theo **thời gian gói**: phát hiện chạy mỗi 5 s thời gian gói, kể cả ở tốc độ tối đa
  (sửa trong `RealtimeProcessor.consume`: trước đây một lô 5.000 gói có thể nhảy vài phút, làm flow quét
  rời cửa sổ host trước khi được phân tích).
* Kiểm thử: `python -m pytest tests/test_web_demo.py` (API, dự đoán một flow, mẫu tổng hợp phát hiện đủ quét/beacon).
