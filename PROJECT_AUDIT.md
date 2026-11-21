# PROJECT AUDIT – Khảo sát workspace ban đầu

Ngày khảo sát: 2026-11-21. Thư mục: `ATTT/` (OneDrive).

## 1. Hiện trạng trước khi triển khai

| Hạng mục | Tình trạng ban đầu |
|---|---|
| Source code | **Không có.** Không có file `.py`, notebook, cấu hình hay git repository. |
| Dataset CSE-CIC-IDS2018 | **Không có.** |
| Dataset CTU-13 | Chỉ có PCAP `capture20110810.truncated.pcap` (6.18 GB, pcapng, 71,829,068 gói, 2011-08-10 14:46:53 → 20:54:07, 22,034 s). Đây là capture của kịch bản 1 (CTU-Malware-Capture-Botnet-42, botnet Neris). Gói tin bị cắt (snaplen 42–66 byte) – chỉ còn header. **Không có file nhãn `.binetflow`.** |
| PCAP khác | `2026-10-01-traffic-analysis-exercise.pcap` (36.8 MB, 42,497 gói, 2,431 s) – bài tập phân tích traffic, không có nhãn; dùng để kiểm thử parser. |
| Model | Không có. |
| Python | Python 3.12.3 (hệ thống) có pandas 3.0, numpy 1.26, scikit-learn 1.4, torch 2.6 CPU; **thiếu** xgboost, lightgbm, catboost, scapy, dpkt, imbalanced-learn, shap, pytest. |
| Công cụ | Wireshark 4.6.9 (tshark, capinfos, editcap) đã cài. |
| Phần cứng | 4 CPU core, 16 GB RAM, GPU NVIDIA MX130 2 GB (không dùng được với bản torch CPU; quá yếu cho huấn luyện), ổ C còn ~270 GB. |

## 2. Những gì đang hoạt động / lỗi hiện tại
- Không có thành phần nào để chạy. Không có lỗi code vì chưa có code.
- Máy chủ phân phối CTU-13 (`mcfp.felk.cvut.cz`) **không truy cập được** trong suốt phiên làm việc (DNS phân giải được nhưng kết nối TCP timeout, thử lại 6 lần × 120 s). Do đó không tải được file nhãn `.binetflow` của CTU-13.
- Một lần cài đặt thư viện thất bại do mất DNS tạm thời; cài lại thành công.

## 3. Thành phần có thể tái sử dụng
- PCAP CTU-13 kịch bản 1: dùng trực tiếp cho PCAP parser, flow reconstruction, phân tích đồ thị và beaconing.
- PCAP bài tập: dữ liệu kiểm thử nhỏ, thực tế.
- tshark/capinfos: kiểm tra độc lập (cross-check) số gói, thời gian của parser tự viết.

## 4. Quyết định kiến trúc rút ra từ khảo sát
1. **Dự án mới** tại `ATTT/network-ids/`; PCAP gốc của người dùng **không bị di chuyển hay sửa**, chỉ được tham chiếu qua `config.yaml` (`paths.external_pcaps`).
2. **Virtualenv riêng** ở `C:\Users\KIM AN\.venvs\nids` (ngoài OneDrive để tránh đồng bộ hàng nghìn file thư viện; không làm thay đổi Python hệ thống).
3. **Lấy mẫu CICIDS2018**: ~16 triệu dòng × 80 cột không vừa 16 GB RAM khi huấn luyện → đọc theo chunk, thống kê toàn bộ dữ liệu, rồi lấy mẫu có kiểm soát trên các dòng *duy nhất* (chi tiết trong `docs/DATA_PREPROCESSING.md`).
4. **Feature parity**: CICIDS2018 được sinh bằng CICFlowMeter, nên bộ trích xuất flow của dự án tái hiện đúng tên, thứ tự và đơn vị (micro-giây, byte payload) của 80 cột CICFlowMeter. Cùng một bộ giải mã gói `decode_frame` dùng cho cả file PCAP (dpkt) và bắt gói trực tiếp (Scapy).
5. **CTU-13 không có nhãn flow** → nhãn mức host theo tài liệu CTU-13 cho kịch bản 1, được **kiểm chứng lại bằng hành vi trong chính PCAP** (xem `docs/DATASET_DOCUMENTATION.md`). Giới hạn này được ghi rõ trong báo cáo.
6. Huấn luyện trên CPU (4 core). MLP dùng PyTorch CPU.

## 5. Kế hoạch hoàn thiện (thứ tự thực hiện)
1. Tải CICIDS2018 (S3 công khai, có resume) → kiểm tra toàn bộ dữ liệu → tài liệu dataset.
2. Pipeline tiền xử lý: làm sạch, nhãn, chia 70/15/15 (stratified + temporal), chọn đặc trưng, scaling, mất cân bằng.
3. EDA notebook.
4. RF, XGBoost, LightGBM, CatBoost, MLP, baseline (LR, DT) – tuning trên validation.
5. Gaussian anomaly detection (univariate, multivariate) + Isolation Forest + One-Class SVM; thí nghiệm unknown attack.
6. CTU-13: PCAP → flow → host features → đồ thị → centralized/P2P → beaconing → đánh giá.
7. Đánh giá độc lập trên test set, so sánh, feature importance/SHAP, lặp seed, temporal split, ablation.
8. Pipeline đa tầng + chương trình realtime (Scapy) + replay PCAP + kịch bản lab.
9. Unit/integration test, tài liệu, báo cáo.

Tiến độ thực tế được theo dõi trong `PROJECT_STATUS.md`.
