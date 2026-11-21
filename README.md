# Ứng dụng học máy phát hiện tấn công mạng (ML-based NIDS)

Hệ thống phát hiện xâm nhập mạng dựa trên học máy, gồm ba bài toán và một chương trình giám sát realtime:

1. **Phát hiện bất thường** bằng Gaussian Anomaly Detection (đơn biến và đa biến), so sánh với Isolation Forest và One-Class SVM.
2. **Phân loại tấn công** (nhị phân BENIGN/ATTACK và đa lớp theo nhóm tấn công) bằng Random Forest, XGBoost, LightGBM, CatBoost, MLP (PyTorch), cùng baseline Logistic Regression và Decision Tree.
3. **Phát hiện botnet** trên CTU-13: đặc trưng mức host, đồ thị giao tiếp (NetworkX), phân tích centralized/P2P, phát hiện beaconing.
4. **Pipeline đa tầng + realtime**: PCAP hoặc gói tin trực tiếp (Scapy) → flow → đặc trưng → Gaussian → classifier → phân tích botnet → cảnh báo (console/JSON/CSV/SQLite).

Kết quả thực nghiệm: [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) · So sánh mô hình: [MODEL_COMPARISON.md](MODEL_COMPARISON.md) · Báo cáo: [REPORT.md](REPORT.md) · Tiến độ: [PROJECT_STATUS.md](PROJECT_STATUS.md)

## Kiến trúc

Chi tiết: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
PCAP (dpkt) ─┐                                    ┌─ Gaussian anomaly ─┐
             ├─ decode_frame ─ FlowBuilder ─ 80 CIC features ─ FlowCleaner ─┤                    ├─ DecisionEngine ─ AlertManager
Live (Scapy)─┘                                    └─ Attack classifier ┘          │
                                     └─ host window ─ host features / beaconing / graph topology ─┘
```

Offline và realtime dùng **cùng** bộ giải mã gói, bộ dựng flow, công thức đặc trưng, cleaner và scaler (feature parity).

## Web demo

```bash
python web/make_samples.py                                  # mẫu demo (một lần)
python -m uvicorn web.app:app --host 127.0.0.1 --port 8000  # mở http://127.0.0.1:8000
```

Phân tích PCAP/CSV (tải lên hoặc mẫu có sẵn), thử một flow kèm giải thích SHAP, giám sát realtime (phát lại PCAP; bắt gói trực tiếp trên Kali với `NIDS_ALLOW_LIVE=1`), và trang kết quả mô hình. Web dùng đúng `DetectionPipeline` + bundle triển khai. Chi tiết: [docs/WEB_DEMO.md](docs/WEB_DEMO.md).

## Cấu trúc thư mục

```
config/config.yaml          cấu hình (đường dẫn, lấy mẫu, timeout, ngưỡng)
src/data_processing/        pcap_parser, flow_builder, feature_extractor, dataset_loader, preprocessing, host_aggregator
src/models/                 random_forest, xgboost_model, lightgbm_model, catboost_model, baselines, mlp, gaussian_detector
src/graph/                  graph_builder, topology_analyzer, beaconing_detector
src/detection/              decision_engine, alert_manager, detection_pipeline, bundle
src/realtime/               packet_capture (Scapy / PCAP replay), realtime_processor, monitor
src/utils/                  config, logger, metrics, visualization
web/                        web demo: app.py (FastAPI), services.py, realtime.py, static/, make_samples.py
scripts/                    download_dataset, preprocess_dataset, train, train_anomaly, evaluate, botnet_analysis,
                            build_bundle, process_pcap, run_realtime, lab/ (kịch bản lab)
notebooks/                  01_data_exploration.ipynb (+ build_notebooks.py để sinh lại)
tests/                      unit + integration tests (pytest)
reports/figures|metrics|experiments   kết quả sinh từ thực nghiệm
docs/                       ARCHITECTURE, DATASET_DOCUMENTATION, DATA_PREPROCESSING, MODEL_DOCUMENTATION, DEPLOYMENT
```

## Cài đặt

Yêu cầu Python 3.12.

```bash
python -m venv ~/.venvs/nids
# Windows: & "$HOME\.venvs\nids\Scripts\Activate.ps1"   |   Linux: source ~/.venvs/nids/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Bắt gói trực tiếp cần libpcap (Linux) hoặc Npcap (Windows) và quyền root/`CAP_NET_RAW` – xem [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Dataset

| Dataset | Vai trò | Lấy dữ liệu |
|---|---|---|
| CSE-CIC-IDS2018 (CSV CICFlowMeter, ~6.9 GB) | phân loại, anomaly detection | `python scripts/download_dataset.py cicids2018` (S3 công khai, tải tiếp khi bị ngắt, kiểm tra kích thước, SHA-256 trong `MANIFEST.json`) |
| CTU-13 kịch bản 1 (PCAP 6.2 GB) | botnet, đồ thị, beaconing | PCAP đặt tại đường dẫn `ctu13.pcap` trong config; nhãn `.binetflow`: `python scripts/download_dataset.py ctu13 --scenarios 42` (khi máy chủ CTU truy cập được) |
| PCAP nhỏ (NETRESEC / bài tập) | kiểm thử parser | bất kỳ file `.pcap/.pcapng` |

Mô tả chi tiết và số liệu thống kê: [docs/DATASET_DOCUMENTATION.md](docs/DATASET_DOCUMENTATION.md).

## Quy trình tái lập thực nghiệm

```bash
python scripts/preprocess_dataset.py --stage all     # kiểm tra, lấy mẫu, chia 70/15/15, làm sạch, chọn đặc trưng, so sánh imbalance
python scripts/train.py --stage scalers              # so sánh scaler cho MLP
python scripts/train.py --task multiclass            # LR, DT, RF, XGB, LGBM, CatBoost, MLP (tuning trên validation)
python scripts/train.py --task binary
python scripts/train_anomaly.py                      # Gaussian đơn/đa biến, Isolation Forest, OCSVM, unknown-attack
python scripts/evaluate.py --stage all               # test set độc lập, importance/SHAP, lặp seed, temporal split, ablation
python scripts/process_pcap.py <ctu.pcap> --part-dir data/processed/ctu13_s42_flows
python scripts/botnet_analysis.py                    # CTU-13: host/graph/beaconing/topology
python scripts/build_bundle.py                       # gói mô hình triển khai
python notebooks/build_notebooks.py                  # sinh và chạy notebook EDA
python -m pytest                                     # unit + integration tests
```

Seed cố định (`project.seed = 42`), cấu hình chia dữ liệu lưu ở `reports/metrics/split_config.json`, siêu tham số và log ở `models/*/*/params.json`, `models/*/*/tuning.csv`, `logs/*.log`.

## Sử dụng

```bash
# Phân tích một file PCAP (offline) và sinh cảnh báo
python scripts/process_pcap.py capture.pcap --detect --alerts logs/pcap_alerts

# Giám sát realtime (Kali Linux)
sudo python3 scripts/run_realtime.py --interface eth0

# Phát lại PCAP qua đúng đường xử lý realtime (không cần quyền root)
python scripts/run_realtime.py --pcap capture.pcap --speed 0

# Kịch bản lab: port scan, DoS mô phỏng, beaconing (chỉ đích loopback/private)
python scripts/lab/run_lab.py --interface lo --target 127.0.0.1
```

Ví dụ một cảnh báo (`logs/alerts/alerts.jsonl`):

```json
{"timestamp": "...", "src_ip": "...", "dst_ip": "...", "src_port": 0, "dst_port": 0, "protocol": "TCP",
 "detection_module": "AttackClassifier", "attack_type": "...", "confidence": 0.0, "severity": "HIGH",
 "reason": "...", "anomaly_score": 0.0, "relevant_features": {}}
```

`confidence` là điểm của mô hình (xác suất ước lượng của classifier, hoặc điểm xếp hạng beaconing), **không** phải xác suất chắc chắn có tấn công; `anomaly_score` = −log p(x) dưới mô hình traffic bình thường.
