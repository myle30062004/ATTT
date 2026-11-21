# Tài liệu dataset

Mọi con số trong tài liệu này được tính từ dữ liệu thực tế bởi `scripts/preprocess_dataset.py --stage inspect` (CICIDS2018, `reports/metrics/cicids_inspection.json`, `cicids_column_stats_raw.csv`) và `scripts/process_pcap.py` (CTU-13, `data/processed/ctu13_s42_flows/parse_stats.json`, `reports/metrics/ctu13_label_verification.json`).

---

## 1. CSE-CIC-IDS2018 (dataset chính)

### 1.1 Nguồn

* Canadian Institute for Cybersecurity & Communications Security Establishment, https://www.unb.ca/cic/datasets/ids-2018.html.
* Bản dùng: *Processed Traffic Data for ML Algorithms* – 10 file CSV do CICFlowMeter-V3 sinh ra từ PCAP, tải từ bucket S3 công khai `s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/` bằng `scripts/download_dataset.py cicids2018` (Range request, tải tiếp khi gián đoạn, kiểm tra đúng kích thước báo bởi S3, SHA-256 lưu trong `data/raw/cicids2018/MANIFEST.json`). Tổng 6.93 GB.
* Môi trường sinh dữ liệu: mạng mô phỏng trên AWS (nhiều phân đoạn nạn nhân, nhóm máy tấn công), 10 ngày làm việc từ 14/02 đến 02/03/2018, mỗi ngày tiến hành một số loại tấn công trong khung giờ xác định; nhãn được gán theo IP máy tấn công/nạn nhân và khoảng thời gian.

### 1.2 Cấu trúc file

| File | Ngày | Số dòng dữ liệu | Dòng header lặp | Kích thước (MB) | Cột bổ sung |
|---|---|---|---|---|---|
| `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` | 2018-02-14 | 1,048,575 | 0 | 341.6 | – |
| `Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv` | 2018-02-15 | 1,048,575 | 0 | 358.5 | – |
| `Friday-16-02-2018_TrafficForML_CICFlowMeter.csv` | 2018-02-16 | 1,048,574 | 1 | 318.3 | – |
| `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv` | 2018-02-20 | 7,948,748 | 0 | 3,867.1 | Flow ID, Src IP, Src Port, Dst IP |
| `Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv` | 2018-02-21 | 1,048,575 | 0 | 313.7 | – |
| `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` | 2018-02-22 | 1,048,575 | 0 | 364.9 | – |
| `Friday-23-02-2018_TrafficForML_CICFlowMeter.csv` | 2018-02-23 | 1,048,575 | 0 | 365.1 | – |
| `Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv` | 2018-02-28 | 613,071 | 33 | 199.6 | – |
| `Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv` | 2018-03-01 | 331,100 | 25 | 102.8 | – |
| `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv` | 2018-03-02 | 1,048,575 | 0 | 336.0 | – |
| **Tổng** | | **16,232,943** | **59** | **6,568** | |

* Mỗi file có 80 cột chung: `Dst Port`, `Protocol`, `Timestamp`, 76 đặc trưng thống kê flow và `Label`. Do đó có **78 đặc trưng số** (gồm `Dst Port`, `Protocol`); `Timestamp` không phải đặc trưng. Danh sách đầy đủ, theo đúng thứ tự: `CIC_FEATURE_COLUMNS` trong `src/data_processing/feature_extractor.py`.
* File ngày 20/02 có thêm 4 cột định danh (`Flow ID`, `Src IP`, `Src Port`, `Dst IP`); các file khác không có. Các cột này **không được dùng** làm đặc trưng (định danh môi trường, gây leakage và không tồn tại ở các ngày khác).
* Đơn vị: thời gian (`Flow Duration`, `*IAT*`, `Active*`, `Idle*`) tính bằng micro-giây; độ dài gói là byte payload tầng 4; `*/s` là tốc độ trên giây.

### 1.3 Nhãn và phân phối lớp (toàn bộ 16.2 triệu dòng)

| Nhãn gốc | Nhãn chuẩn hoá | Nhóm (đa lớp) | Số dòng | Tỉ lệ | Dòng duy nhất | Tỉ lệ trùng |
|---|---|---|---|---|---|---|
| `Benign` | Benign | BENIGN | 13,484,708 | 83.070% | 10,637,200 | 21.1% |
| `DDOS attack-HOIC` | DDoS-HOIC | DDoS | 686,012 | 4.226% | 198,861 | 71.0% |
| `DDoS attacks-LOIC-HTTP` | DDoS-LOIC-HTTP | DDoS | 576,191 | 3.550% | 575,364 | 0.1% |
| `DoS attacks-Hulk` | DoS-Hulk | DoS | 461,912 | 2.846% | 145,199 | 68.6% |
| `Bot` | Bot | Botnet | 286,191 | 1.763% | 144,535 | 49.5% |
| `FTP-BruteForce` | BruteForce-FTP | BruteForce | 193,360 | 1.191% | **54** | 100.0% |
| `SSH-Bruteforce` | BruteForce-SSH | BruteForce | 187,589 | 1.156% | 94,048 | 49.9% |
| `Infilteration` | Infiltration | Infiltration | 161,934 | 0.998% | 140,610 | 13.2% |
| `DoS attacks-SlowHTTPTest` | DoS-SlowHTTPTest | DoS | 139,890 | 0.862% | **55** | 100.0% |
| `DoS attacks-GoldenEye` | DoS-GoldenEye | DoS | 41,508 | 0.256% | 41,406 | 0.2% |
| `DoS attacks-Slowloris` | DoS-Slowloris | DoS | 10,990 | 0.068% | 9,908 | 9.8% |
| `DDOS attack-LOIC-UDP` | DDoS-LOIC-UDP | DDoS | 1,730 | 0.011% | 1,730 | 0.0% |
| `Brute Force -Web` | Web-BruteForce | WebAttack | 611 | 0.004% | 555 | 9.2% |
| `Brute Force -XSS` | Web-XSS | WebAttack | 230 | 0.001% | 228 | 0.9% |
| `SQL Injection` | Web-SQLInjection | WebAttack | 87 | 0.001% | 84 | 3.4% |

Ý nghĩa nhãn:

* **BENIGN**: traffic người dùng mô phỏng (B-Profile: web, email, SSH, FTP...).
* **DoS** (một nguồn): GoldenEye, Hulk (lũ HTTP GET với tham số ngẫu nhiên), SlowHTTPTest và Slowloris (giữ kết nối HTTP mở lâu, gửi header chậm).
* **DDoS** (nhiều nguồn): LOIC-HTTP, LOIC-UDP, HOIC.
* **BruteForce**: dò mật khẩu FTP (Patator) và SSH.
* **WebAttack**: dò mật khẩu form web, XSS, SQL injection (DVWA).
* **Botnet**: Zeus/Ares – máy bị nhiễm liên lạc C&C, chụp màn hình, lấy dữ liệu.
* **Infiltration**: máy nội bộ bị nhiễm qua tài liệu độc hại, sau đó quét nội bộ (Nmap) từ máy bị nhiễm.
* **Không có lớp `PortScan`** trong CICIDS2018 (khác CICIDS2017); dự án không tự tạo nhãn này. Quét mạng chỉ xuất hiện như một phần của Infiltration.

Phân phối theo ngày:

| Ngày | Nhãn (số dòng) |
|---|---|
| 2018-02-14 | Benign 667,626, BruteForce-FTP 193,360, BruteForce-SSH 187,589 |
| 2018-02-15 | Benign 996,077, DoS-GoldenEye 41,508, DoS-Slowloris 10,990 |
| 2018-02-16 | DoS-Hulk 461,912, Benign 446,772, DoS-SlowHTTPTest 139,890 |
| 2018-02-20 | Benign 7,372,557, DDoS-LOIC-HTTP 576,191 |
| 2018-02-21 | DDoS-HOIC 686,012, Benign 360,833, DDoS-LOIC-UDP 1,730 |
| 2018-02-22 | Benign 1,048,213, Web-BruteForce 249, Web-XSS 79, Web-SQLInjection 34 |
| 2018-02-23 | Benign 1,048,009, Web-BruteForce 362, Web-XSS 151, Web-SQLInjection 53 |
| 2018-02-28 | Benign 544,200, Infiltration 68,871 |
| 2018-03-01 | Benign 238,037, Infiltration 93,063 |
| 2018-03-02 | Benign 762,384, Bot 286,191 |

Mỗi loại tấn công chỉ xuất hiện trong 1–2 ngày. Vì vậy **không thể** dùng cách chia "train trên một số ngày, test trên ngày khác" cho bài toán đa lớp (lớp ở tập test sẽ không có trong tập train); thay vào đó dự án dùng chia phân tầng ngẫu nhiên và **chia theo thời gian trong từng nhãn** (xem `DATA_PREPROCESSING.md`).

### 1.4 Chất lượng dữ liệu

| Vấn đề | Số liệu | Xử lý |
|---|---|---|
| Dòng header lặp giữa file | 59 dòng (3 file) | loại khi đọc |
| `NaN` | 59,721 ô, tất cả ở `Flow Byts/s` | xem DATA_PREPROCESSING |
| `+Infinity` | 131,799 ô: `Flow Pkts/s` 95,760; `Flow Byts/s` 36,039 | xem DATA_PREPROCESSING |
| `-Infinity` | 0 | – |
| Thời gian âm | 14–15 dòng ở `Flow Duration`, `Flow IAT *`, `Fwd IAT *` (tới −9.47·10¹¹ µs) – lỗi đồng hồ | đặt về 0 |
| Giá trị −1 | `Init Fwd Win Byts` 4,432,608 dòng; `Init Bwd Win Byts` 8,255,549 dòng | giữ: là giá trị quy ước "không quan sát được TCP window" |
| Cột hằng số (toàn bộ dữ liệu) | `Bwd PSH Flags`, `Bwd URG Flags`, `Fwd Byts/b Avg`, `Fwd Pkts/b Avg`, `Fwd Blk Rate Avg`, `Bwd Byts/b Avg`, `Bwd Pkts/b Avg`, `Bwd Blk Rate Avg` | loại |
| Dòng trùng lặp (đặc trưng + nhãn) | 4,243,106 / 16,232,943 (26.1%) | khử trùng lặp trước khi chia |
| Vector đặc trưng mang >1 nhãn | 22,359 vector, 423,290 dòng | loại khỏi tập làm việc (xem dưới) |
| `Timestamp` dùng đồng hồ 12 giờ, không có AM/PM | mọi file chỉ có giờ 01–12 (không có 00, 13–23) | quy ước: giờ 01–07 là buổi chiều (+12 h), 08–12 là buổi sáng; kiểm tra: dải thời gian sau chuyển đổi 08:28–17:30 mỗi ngày, khớp giờ làm việc của testbed |
| File bị cắt ở giới hạn Excel | 7/10 file có đúng 1,048,575 (hoặc 1,048,574 + 1 header) dòng = 1,048,576 − 1 dòng tiêu đề | không khắc phục được; ghi nhận là giới hạn |

Nhận xét quan trọng:

1. **FTP-BruteForce và DoS-SlowHTTPTest gần như không chứa thông tin**: 193,360 dòng FTP chỉ có 54 vector khác nhau; 139,890 dòng SlowHTTPTest chỉ có 55. Đây là các flow rất ngắn, gần như giống hệt nhau (tấn công thất bại/không có payload). Nếu không khử trùng lặp, các lớp này chiếm hàng trăm nghìn dòng "dễ" và thổi phồng accuracy.
2. **7 trên 10 file có đúng 1,048,575 dòng dữ liệu** – đúng bằng giới hạn 1,048,576 dòng của Microsoft Excel trừ dòng tiêu đề. Các file này nhiều khả năng đã bị cắt khi lưu qua Excel; phần traffic sau dòng đó của các ngày tương ứng không có trong bộ CSV công bố. Hệ quả: phân phối theo thời gian của các ngày này không đầy đủ.
3. **Benign và Infiltration trùng vector**: trong tập làm việc, 1,894 vector đặc trưng xuất hiện với cả nhãn Benign lẫn Infiltration (flow 1–3 gói, ví dụ DNS). Infiltration được gán nhãn theo IP máy bị nhiễm và thời gian, nên traffic bình thường của chính máy đó cũng mang nhãn tấn công – nhiễu nhãn vốn có của dataset.

### 1.5 Thống kê mô tả

Bảng đầy đủ (count, NaN, ±Inf, số giá trị âm/0, min, max, mean, std trên toàn bộ dữ liệu; median trên mẫu 2%): `reports/metrics/cicids_column_stats_raw.csv` và mục 5 của `notebooks/01_data_exploration.ipynb`.

### 1.6 Cách sử dụng trong dự án

Toàn bộ 16.2 triệu dòng được duyệt một lần để thống kê. Tập làm việc (để vừa RAM 16 GB) được lấy **trên các dòng duy nhất** của từng nhãn chi tiết: giữ toàn bộ nhãn hiếm, giới hạn 150,000 dòng/nhãn tấn công và 900,000 dòng Benign; lựa chọn là hàm tất định của hash dòng (seed 42). Kết quả: 1,779,198 dòng duy nhất; sau khi loại 3,833 dòng thuộc 1,914 vector xung đột nhãn còn 1,775,365 dòng, chia 70/15/15 thành 1,242,755 / 266,305 / 266,305 – chi tiết trong `DATA_PREPROCESSING.md`.

---

## 2. CTU-13 (dataset bổ sung – botnet)

### 2.1 Nguồn

* Stratosphere Lab, CTU Prague: https://www.stratosphereips.org/datasets-ctu13 – 13 kịch bản bắt traffic thật trong mạng đại học CTU (2011), mỗi kịch bản chạy một mẫu malware trên máy ảo Windows. Giấy phép CC BY.
* Dự án dùng **kịch bản 1** (`CTU-Malware-Capture-Botnet-42`, botnet **Neris**: IRC C&C, gửi spam, click-fraud), file `capture20110810.truncated.pcap` đã có sẵn trong workspace.
* File nhãn flow chính thức (`capture20110810.binetflow`) và các kịch bản khác **không tải được**: máy chủ `mcfp.felk.cvut.cz` không phản hồi trong toàn bộ phiên làm việc (kết nối TCP timeout, thử lại nhiều lần). Mirror chính thức trên Mega yêu cầu tải thủ công. `scripts/download_dataset.py ctu13 --scenarios 42 50 51 53` sẵn sàng để tải khi máy chủ hoạt động.

### 2.2 Cấu trúc PCAP

| Thuộc tính | Giá trị (capinfos) |
|---|---|
| Định dạng | pcapng, Ethernet, độ phân giải µs; gộp từ 3 file tcp/udp/icmp (mergecap) |
| Số gói | 71,829,068 |
| Thời gian | 2011-08-10 14:46:53 → 20:54:07 (22,034 s ≈ 6.1 giờ) |
| Kích thước | 6.18 GB trên đĩa; 54.4 GB dữ liệu gốc (gói bị cắt còn 42–66 byte – chỉ giữ header) |

Kết quả dựng flow bằng `scripts/process_pcap.py` (flow timeout 120 s, idle timeout 60 s): 71,829,042 gói giải mã được (26 khung hỏng bị bỏ qua – số gói khớp capinfos), **4,998,700 flow hai chiều**, 555,933 IP nguồn khác nhau, thời gian xử lý 2,444 s (~29,400 gói/s, một luồng Python). Vì gói bị cắt, độ dài payload được tính từ trường *IP total length*, không phải độ dài bắt được.

### 2.3 Nhãn (ground truth mức host)

Theo phương pháp gán nhãn của CTU-13: flow từ/tới máy bị nhiễm là *botnet*; flow từ các máy bình thường được kiểm soát là *normal*; mọi thứ khác là *background* (không biết nhãn). Danh sách IP trong `config/config.yaml` (`ctu13.infected_hosts`, `ctu13.normal_hosts`) theo tài liệu CTU-13 của kịch bản 1. Vì không có file nhãn chính thức, danh sách này được **kiểm chứng bằng hành vi trong chính PCAP** (`reports/metrics/ctu13_label_verification.json`):

| Host | Vai trò | Flow khởi tạo | IP đích khác nhau | Tỉ lệ flow không có phản hồi | Port đích phổ biến |
|---|---|---|---|---|---|
| 147.32.84.165 | **nhiễm Neris** | 21,695 | 4,190 | 56.0% | 25 (8,367), 53 (7,126), 80 (2,140), **6667 (1,349)**, 65500 (885), 4506 (862) |
| 147.32.84.170 | normal | 26,420 | 303 | 23.6% | 53, 80, 443, 5222 |
| 147.32.84.134 | normal | 5,858 | 135 | 19.6% | 53, 80, 443, 5222, 1863, 993 |
| 147.32.84.164 | normal | 11,349 | 239 | 17.1% | 53, 80, 443, 5222 |
| 147.32.87.36 | normal | 2,488 | 175 | 87.4% | 53, ephemeral |
| 147.32.80.9 | normal | 155 | 17 | 96.8% | 0 (ICMP) |
| 147.32.87.11 | normal | 268 | 2 | 99.3% | 0 (ICMP) |

Bằng chứng: 147.32.84.165 là host gửi nhiều flow tới cổng IRC 6667 nhất trong toàn bộ capture (1,349; host kế tiếp 362) và gửi SMTP nhiều nhất (8,367; host kế tiếp 1,831) – khớp với hành vi Neris (C&C qua IRC, phát tán spam); host này chỉ bắt đầu hoạt động lúc 1312967064 (≈ 78 phút sau khi bắt đầu capture, khi malware được chạy). Các host normal có hồ sơ duyệt web/DNS/XMPP thông thường; 147.32.80.9 và 147.32.87.11 hầu như chỉ là đích (máy chủ hạ tầng), khởi tạo rất ít flow.

**Giới hạn**: nhãn mức host chứ không phải nhãn flow chính thức; chỉ 1 host bị nhiễm và 6 host normal; host background 147.32.84.166 cũng có 362 flow tới cổng 6667 nhưng không có nhãn nên không được coi là âm tính hay dương tính.

### 2.4 Cách sử dụng

PCAP → flow (cùng code với realtime) → đặc trưng host theo cửa sổ 5 phút + độ đo đồ thị → so sánh bộ phát hiện, beaconing, phân tích topology (`scripts/botnet_analysis.py`). Không thực hiện cross-dataset CICIDS2018 → CTU-13 cho bộ phân loại đa lớp, vì tập nhãn không tương thích (CTU-13 kịch bản 1 chỉ có Neris/IRC/spam; không có lớp tương ứng trong CICIDS2018 ngoài "Botnet" là Zeus/Ares với hành vi khác).

---

## 3. PCAP bổ trợ

`2026-10-01-traffic-analysis-exercise.pcap` (36.8 MB, 42,497 gói, 2,431 s, không có nhãn) – bài tập phân tích traffic: dùng kiểm thử parser và pipeline trên traffic thật (42,344 gói giải mã được, 153 khung không phải IP bị bỏ qua, 779 flow). Bộ PCAP NETRESEC có thể dùng tương tự qua `scripts/process_pcap.py`.
