# PROJECT STATUS

Cập nhật: 2026-11-21 (phiên 1, đang chạy). Môi trường: `C:\Users\KIM AN\.venvs\nids` (Python 3.12, CPU).
Chạy test: `& "C:\Users\KIM AN\.venvs\nids\Scripts\python.exe" -m pytest` trong `network-ids/`.

| Module | Status | Files | Tests / kiểm chứng | Kết quả |
|---|---|---|---|---|
| Dataset CICIDS2018 | XONG | scripts/download_dataset.py, data/raw/cicids2018 (10 CSV, MANIFEST sha256) | kích thước khớp S3 | 16,232,943 dòng |
| Dataset CTU-13 | XONG một phần | PCAP kịch bản 1 (sẵn có); nhãn mức host trong config | nhãn kiểm chứng bằng hành vi (ctu13_label_verification.json) | binetflow không tải được (mcfp timeout) |
| PCAP parser / flow / features | XONG | pcap_parser.py, flow_builder.py, feature_extractor.py | test_packets_flows.py (9 pass); CTU: 71,829,042/71,829,068 gói = capinfos | 4,998,700 flow CTU |
| Preprocessing | XONG | dataset_loader.py, preprocessing.py, scripts/preprocess_dataset.py | test_preprocessing_loader.py (7 pass); 0 overlap giữa split | 1,775,365 dòng, 30 đặc trưng |
| EDA | XONG | notebooks/01_data_exploration.ipynb, build_notebooks.py, commentary/ | notebook chạy hết | reports/figures/eda_* |
| LightGBM divergence | ĐÃ SỬA | lightgbm_model.py (min_sum_hessian_in_leaf=1) | reports/experiments/lightgbm_divergence.json | val mF1 0.62 → 0.90 |
| ML/DL training | ĐANG CHẠY | scripts/train.py (multiclass → binary → scalers) | test_models.py (7 pass) | logs/stage_train_*.out |
| Gaussian anomaly | ĐANG CHẠY lại (thêm GMM, matched-FPR LOFO) | gaussian_detector.py, scripts/train_anomaly.py | test_anomaly_graph.py (10 pass) | lần 1: flow-level recall ≤3% @1% FPR |
| Botnet CTU-13 | XONG | host_aggregator.py, graph/*, scripts/botnet_analysis.py | | RF host: 11/11 bot windows test; topology sim 34% → 83% sau khi thêm population baseline |
| Detection pipeline / realtime | Code xong, integration test pass | src/detection/*, src/realtime/*, scripts/run_realtime.py, build_bundle.py | test_pipeline_integration.py (6 pass) | cần bundle thật |
| Lab scenarios | Code xong, CHƯA CHẠY | scripts/lab/* | | cần bundle + Npcap (đã cài) |
| Docs | Một phần | README, PROJECT_AUDIT, docs/ARCHITECTURE, DATASET_DOCUMENTATION, DATA_PREPROCESSING, MODEL_DOCUMENTATION, DEPLOYMENT | | còn: EXPERIMENT_RESULTS, MODEL_COMPARISON, REPORT |

## Remaining issues
- CTU-13 binetflow chính thức chưa có → nhãn mức host (1 bot, 6 normal); kết quả botnet có cỡ mẫu nhỏ.
- Infiltration chồng lấn Benign (nhiễu nhãn của dataset) → recall thấp.
- Flow-level anomaly detection yếu ở FPR thấp – ghi rõ trong báo cáo.

## Next steps
1. Chờ train xong → `python scripts/evaluate.py --stage all`.
2. `python scripts/build_bundle.py` → `python scripts/process_pcap.py <exercise pcap> --detect`, replay, lab (`scripts/lab/run_lab.py --interface "Loopback Pseudo-Interface 1"`).
3. Viết EXPERIMENT_RESULTS.md, MODEL_COMPARISON.md, REPORT.md; cập nhật file này.

## Tạm dừng theo yêu cầu – 2026-11-21 07:45

* Đã xong: huấn luyện đa lớp 7/7 mô hình; huấn luyện nhị phân 6/7 (lr, dt, lgbm, xgb, rf, cat).
* Bị dừng giữa chừng: MLP nhị phân (epoch 17/40) – chưa lưu mô hình, phải chạy lại từ đầu.
* Chưa chạy: `train.py --stage scalers`, `evaluate.py --stage all`, `train_deployment.py`, lab, tài liệu kết quả.
* Tiếp tục (mỗi mô hình đã lưu `params.json` riêng; `training_summary_binary.json` được dựng lại từ các file này và `train.py` nay ghi tổng hợp sau từng mô hình):

```powershell
& "$env:USERPROFILE\.venvs\nids\Scripts\python.exe" -u scripts/train.py --task binary --models mlp *> logs/stage_train_binary_mlp.out
& "$env:USERPROFILE\.venvs\nids\Scripts\python.exe" -u scripts/train.py --stage scalers *> logs/stage_scalers.out
```

## Tiếp tục – 2026-11-21

* Đang chạy lại MLP nhị phân, sau đó so sánh scaler.
* Kiểm tra: accuracy validation gần như bằng nhau giữa đa lớp và nhị phân ở mọi mô hình (LGBM 0.9456 / 0.9451, RF 0.9437 / 0.9437) → gần như toàn bộ lỗi là nhầm BENIGN ↔ tấn công (FNR ≈ 9%, chủ yếu Infiltration), không phải nhầm giữa các họ tấn công. Vì vậy val_acc của MLP trùng nhau (0.9365) là hệ quả, không phải lỗi.

## Báo cáo Word – 2026-11-21

* `BAO_CAO_ATTT_hoan_thien.docx` (+ `.pdf`, 50 trang): giữ trang bìa, thành viên, phân công, mục lục và định dạng của `BAO_CAO_ATTT.docx` (file gốc không bị sửa); viết lại Tóm tắt, Phần 1–6 và Phụ lục theo kết quả của network-ids.
* Số liệu mô hình có giám sát lấy từ **validation** (theo yêu cầu); detector bất thường và mô hình triển khai đã có số liệu test.
* Dựng lại: `python scripts/make_report_figures.py` → `python scripts/report/build_report.py` → mở bằng Word, Update Field (F9) cho mục lục/danh mục.
* Cần dựng lại khi: MLP nhị phân xong (hiện ghi “đang huấn luyện”), và khi có kết quả `evaluate.py` (test, nhiều seed).
* Sửa lỗi tài liệu: số file CSV bị cắt ở giới hạn Excel là 7/10, không phải 8/10 (DATASET_DOCUMENTATION, notebook).

## Chuyển vị trí – 2026-11-21

* OneDrive hết dung lượng → dự án chuyển sang `C:\Users\KIM AN\network-ids`, PCAP CTU-13 sang `C:\Users\KIM AN\data\capture20110810.truncated.pcap\` (sao chép bằng robocopy, đối chiếu 322 file khớp từng byte, 38 test pass, rồi mới xoá bản OneDrive).
* `config/config.yaml`: đường dẫn PCAP đổi thành `../data/capture20110810.truncated.pcap/...`.
* MLP nhị phân bị dừng ở epoch 23 khi chuyển; đang chạy lại từ đầu, sau đó so sánh scaler.

## Hoàn thiện + web demo – 2026-11-21 (tối)

* **MLP nhị phân: XONG** (12:48, best epoch 39/40) – val Macro-F1 0,937, FPR 1,1%, FNR 11,6%; test 0,937. So sánh scaler: XONG (`reports/experiments/scaler_comparison.csv`, minmax+log tốt nhất 0,826).
* **evaluate.py**
  * `test`: XONG (20:36) – `reports/metrics/model_comparison_{multiclass,binary}.csv`, `cm_*`, `per_class_*`, `roc_*/pr_*`.
    7 lớp: RF 0,911 · LightGBM 0,909 · XGBoost 0,906 · CatBoost 0,900 · MLP 0,876 · DT 0,875 · LR 0,735.
    Nhị phân: LightGBM = CatBoost 0,945 · XGBoost 0,944 · RF 0,944 · DT 0,938 · MLP 0,937 · LR 0,884.
  * `importance`: `shap.TreeExplainer` treo ~1 giờ trên LightGBM 300 cây × 255 lá × 7 lớp → đổi sang TreeSHAP gốc
    của LightGBM (`booster_.predict(pred_contrib=True)`, cùng giá trị). Đang chạy lại cùng `temporal`, `ablation`, `repeats`
    (log: `logs/stage_evaluate_rest.out`).
* **Sửa lỗi realtime replay** (`src/realtime/realtime_processor.py`): ở tốc độ tối đa một lô 5.000 gói có thể nhảy vài
  phút thời gian gói nên flow quét cổng rời cửa sổ host trước khi được phân tích → nay dừng lô tại mỗi khoảng
  `detection_interval_s` theo thời gian gói. 41/41 test pass.
* **Web demo: XONG** – `web/` (FastAPI + HTML/JS), hướng dẫn `docs/WEB_DEMO.md`, test `tests/test_web_demo.py` (3 pass).
  Chạy: `python -m uvicorn web.app:app --port 8000`. Mẫu: `python web/make_samples.py`.
* Còn lại: dựng lại báo cáo Word + slide với số liệu test/MLP nhị phân/seed khi evaluate xong; kịch bản lab realtime (E12) vẫn chưa chạy.
