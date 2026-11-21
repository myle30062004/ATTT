# Tài liệu mô hình

Tài liệu này trình bày cơ sở lý thuyết, công thức, kiến trúc, siêu tham số và cách huấn luyện/suy luận của từng mô hình. Siêu tham số **được chọn** và kết quả thực nghiệm nằm trong `models/<model>/<task>/params.json`, `tuning.csv` và [EXPERIMENT_RESULTS.md](../EXPERIMENT_RESULTS.md).

## 0. Ký hiệu và quy trình chung

* Một mẫu là một flow hai chiều \(x \in \mathbb{R}^d\) gồm \(d\) đặc trưng CICFlowMeter đã chọn; nhãn \(y\) là `BENIGN/ATTACK` (nhị phân) hoặc một nhóm tấn công (đa lớp).
* Tập train/validation/test = 70/15/15, chia phân tầng theo nhãn chi tiết, đã khử trùng lặp trước khi chia.
* **Tuning chỉ dùng validation**: với mỗi mô hình, lấy ngẫu nhiên \(N\) cấu hình từ không gian tham số (`ParameterSampler`, seed 42), huấn luyện trên một mẫu con phân tầng 250k dòng của train, đánh giá macro-F1 trên validation; cấu hình tốt nhất được huấn luyện lại trên toàn bộ train. XGBoost/LightGBM/CatBoost dùng validation cho early stopping (30 vòng). Test chỉ được đọc trong `scripts/evaluate.py`.
* Tiêu chí chọn: macro-F1 (mọi lớp có trọng số như nhau, nên lớp hiếm như WebAttack không bị lấn át), hoà thì chọn FPR thấp hơn.

## 1. Random Forest

Một tập \(T\) cây quyết định, mỗi cây học trên một mẫu bootstrap và tại mỗi nút chỉ xét ngẫu nhiên `max_features` đặc trưng. Nút được tách theo đặc trưng/ngưỡng làm giảm nhiều nhất độ vẩn đục Gini

\[ G(S) = 1 - \sum_k p_k^2, \qquad \Delta = G(S) - \tfrac{|S_L|}{|S|}G(S_L) - \tfrac{|S_R|}{|S|}G(S_R) \]

Dự đoán: \( \hat p(k\mid x) = \frac1T \sum_t \hat p_t(k\mid x) \). Phương sai giảm nhờ lấy trung bình các cây ít tương quan; cây sâu cho phép biểu diễn ngưỡng phi tuyến (ví dụ "SYN nhiều *và* thời lượng ngắn"). Không cần scaling vì cây chỉ so sánh ngưỡng.

* Không gian tham số: `n_estimators` ∈ {100, 200, 300}, `max_depth` ∈ {None, 20, 30, 40}, `min_samples_split` ∈ {2, 5, 10}, `min_samples_leaf` ∈ {1, 2, 4}, `max_features` ∈ {sqrt, log2, 0.3}, `class_weight` ∈ {None, balanced, balanced_subsample}; 6 cấu hình.
* `class_weight="balanced"` gán trọng số \(w_k = n / (K n_k)\) cho lớp \(k\).
* Feature importance = tổng mức giảm Gini có trọng số mà đặc trưng tạo ra, trung bình trên các cây (thiên về đặc trưng nhiều giá trị – vì vậy đối chiếu thêm với gain của boosting và SHAP).

## 2. Gradient boosting: XGBoost, LightGBM, CatBoost

Mô hình cộng \(F_M(x) = \sum_{m=1}^{M} \eta f_m(x)\), mỗi cây \(f_m\) khớp gradient của hàm mất mát log-loss (softmax cho đa lớp). XGBoost dùng khai triển bậc hai:

\[ \mathcal{L}^{(m)} \approx \sum_i \big[g_i f_m(x_i) + \tfrac12 h_i f_m(x_i)^2\big] + \gamma T_m + \tfrac12\lambda \sum_j w_j^2 + \alpha \sum_j |w_j| \]

với \(g_i, h_i\) là đạo hàm bậc 1, 2 của loss; trọng số lá tối ưu \(w_j^* = -\frac{\sum_{i\in j} g_i}{\sum_{i\in j} h_i + \lambda}\); gain của một phép tách
\( \tfrac12\big[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\big] - \gamma \).

* **XGBoost** (`tree_method=hist`): tham số `n_estimators, max_depth, learning_rate (η), subsample, colsample_bytree, min_child_weight, gamma (γ), reg_alpha (α), reg_lambda (λ)`; 8 cấu hình.
* **LightGBM**: cây phát triển theo lá (leaf-wise, giới hạn bởi `num_leaves`), histogram và GOSS → nhanh nhất khi huấn luyện; tham số `n_estimators, num_leaves, max_depth, learning_rate, feature_fraction, bagging_fraction, lambda_l1, lambda_l2, min_child_samples, class_weight`; 10 cấu hình.
* **CatBoost** (đối chứng): cây đối xứng (oblivious), ordered boosting giảm prediction shift; tham số `iterations, depth, learning_rate, l2_leaf_reg, auto_class_weights`; 4 cấu hình.
* Early stopping: dừng khi log-loss validation không giảm sau 30 vòng; số vòng tốt nhất lưu trong `params.json`.
* Feature importance: *gain* (tổng mức giảm loss do đặc trưng).

## 3. Baseline

* **Logistic Regression** (softmax, `class_weight=balanced`, đầu vào đã scale): \( p(k\mid x) = \frac{\exp(w_k^\top x + b_k)}{\sum_j \exp(w_j^\top x + b_j)} \). Đo xem bài toán có tách được tuyến tính hay không.
* **Decision Tree** (`max_depth=20`, `class_weight=balanced`): một cây đơn, mức tham chiếu cho ensemble.

## 4. MLP (Deep Learning)

Kiến trúc (`src/models/mlp.py`):

| Lớp | Kích thước | Kích hoạt / chuẩn hoá |
|---|---|---|
| Input | \(d\) đặc trưng đã chọn, đã `sign·log1p` + StandardScaler | |
| Hidden 1 | 256 | ReLU → BatchNorm → Dropout(p) |
| Hidden 2 | 128 | ReLU → BatchNorm → Dropout(p) |
| Hidden 3 | 64 | ReLU → Dropout(p) |
| Output | \(K\) logits | (softmax trong loss) |

* Loss: cross-entropy có trọng số lớp \( \mathcal{L} = -\frac1n \sum_i w_{y_i} \log \mathrm{softmax}(z_i)_{y_i} \), với \(w_k \propto \sqrt{n/(K n_k)}\) (căn bậc hai để không phóng đại quá mức lớp cực hiếm).
* Tối ưu: Adam (lr khởi đầu 1e-3), batch 256/512, `ReduceLROnPlateau` (giảm lr × 0.5 khi val loss không giảm sau 2 epoch), early stopping (patience 6) và khôi phục trọng số của epoch có val loss nhỏ nhất.
* Lựa chọn cấu hình: 4 cấu hình (dropout 0.2/0.3, lr 1e-3/5e-4, batch 256/512, có/không trọng số lớp) huấn luyện ngắn (≤12 epoch) trên mẫu con, chọn theo macro-F1 validation; cấu hình thắng được huấn luyện đầy đủ (≤40 epoch) trên toàn bộ train.
* Lý do chọn MLP: dữ liệu flow là dạng bảng (không có cấu trúc chuỗi/ảnh), MLP là mạng sâu phù hợp nhất, dễ kiểm soát, suy luận nhanh trên CPU. Biểu đồ loss/accuracy: `reports/figures/mlp_*_training_curves.png`.
* Scaling: so sánh Standard/Robust/MinMax, có/không `log1p`, trong `reports/experiments/scaler_comparison.csv`.

## 5. Gaussian Anomaly Detection

Mục tiêu khác hẳn phân loại: **chỉ học phân phối của traffic bình thường** \(p(x)\), rồi coi các flow có mật độ thấp là bất thường. Không dùng nhãn tấn công khi học tham số; nhãn chỉ dùng để chọn ngưỡng trên validation. Đây **không phải Gaussian Naive Bayes** (GNB học \(p(x\mid y)\) cho *mọi* lớp, kể cả lớp tấn công, rồi áp dụng Bayes – tức là một bộ phân loại có giám sát, không phát hiện được tấn công chưa biết).

Tiền xử lý: \( \tilde x = \mathrm{sign}(x)\log(1+|x|) \), rồi chuẩn hoá theo trung bình/độ lệch chuẩn của traffic BENIGN trong train. Biến đổi log làm phân phối byte/gói/thời gian (lệch phải, trải nhiều bậc độ lớn) gần Gaussian hơn.

**Đơn biến** (giả định các đặc trưng độc lập):
\[ \mu_j = \frac1m\sum_i x_j^{(i)}, \quad \sigma_j^2 = \frac1m\sum_i (x_j^{(i)}-\mu_j)^2 \]
\[ \log p(x) = \sum_{j=1}^{d}\Big[-\tfrac12\log(2\pi\sigma_j^2) - \frac{(x_j-\mu_j)^2}{2\sigma_j^2}\Big] \]

**Đa biến** (ma trận hiệp phương sai đầy đủ, mô hình hoá tương quan):
\[ \Sigma = \frac1{m-1}\sum_i (x^{(i)}-\mu)(x^{(i)}-\mu)^\top + \lambda I \]
\[ \log p(x) = -\tfrac12\big[d\log(2\pi) + \log|\Sigma| + (x-\mu)^\top\Sigma^{-1}(x-\mu)\big] \]

* Hạng \((x-\mu)^\top\Sigma^{-1}(x-\mu)\) là khoảng cách Mahalanobis: một flow có từng đặc trưng "bình thường" nhưng *tổ hợp* bất thường (ví dụ nhiều gói nhưng gần như không byte) vẫn bị phát hiện – mô hình đơn biến bỏ sót trường hợp này (được kiểm chứng trong `tests/test_anomaly_graph.py`).
* **Ổn định số**: (1) dùng log-density thay vì tích các mật độ (với \(d\) ≈ 30, \(p(x)\) dễ nhỏ hơn \(10^{-300}\) và underflow về 0); (2) ridge \(\lambda I\) (\(\lambda = 10^{-3}\)) giữ \(\Sigma\) xác định dương khi các đặc trưng gần cộng tuyến (ma trận suy biến); nếu phân tích Cholesky vẫn thất bại, \(\lambda\) được tăng 10 lần; (3) \(\log|\Sigma| = 2\sum_i \log L_{ii}\) và \(\Sigma^{-1}(x-\mu)\) tính qua nhân tử Cholesky \(\Sigma = LL^\top\), không nghịch đảo ma trận trực tiếp. Số điều kiện của \(\Sigma\) được ghi lại.
* **Chọn ε**: ứng viên là 1000 phân vị của log-density trên validation; flow bất thường nếu \(\log p(x) < \log\varepsilon\); chọn ε tối đa F1 (attack = lớp dương). Một điểm hoạt động thứ hai giới hạn FPR validation ≤ 1%.
* **Điểm bất thường** \(s(x) = -\log p(x)\) chỉ là thước đo "mức khác thường so với traffic bình thường đã học", **không phải xác suất bị tấn công**.
* **Giải thích**: với mỗi flow bất thường, pipeline báo 5 đặc trưng có \(|z_j| = |x_j-\mu_j|/\sigma_j\) lớn nhất.
* Đối chứng: **Isolation Forest** (điểm = độ sâu cô lập trung bình; điểm bất thường cao khi mẫu bị cô lập nhanh) và **One-Class SVM** (RBF, ν = 0.01, học trên 15k flow BENIGN do độ phức tạp \(O(n^2)\)).

## 6. Phát hiện botnet (CTU-13)

### 6.1 Đặc trưng mức host (cửa sổ 300 s)
Với host nguồn \(h\) và tập flow \(F_h\) do \(h\) khởi tạo:

| Đặc trưng | Công thức / ý nghĩa |
|---|---|
| `unique_dst_ips`, `unique_dst_ports` | \(|\{dst\}|\), \(|\{port\}|\) |
| `total_connections`, `connections_per_sec` | \(|F_h|\), \(|F_h| / \text{khoảng hoạt động}\) |
| `failed_connections`, `failed_ratio` | flow không có gói chiều về; tỉ lệ trên \(|F_h|\) |
| `syn_without_ack` | số gói SYN không kèm ACK |
| `dst_port_entropy`, `dst_ip_entropy` | \(H = -\sum_v p_v \log_2 p_v\) trên phân phối đích; cao ↔ quét/lan truyền |
| `conn_interval_mean/std` | trung bình/độ lệch khoảng cách giữa các lần bắt đầu kết nối |
| `periodicity_score` | \(1 - \min(1, CV)\), \(CV = \sigma/\mu\) của khoảng cách kết nối |
| `repeated_connections`, `repeated_ratio` | kết nối tới cặp (ip, port) đã gặp |
| `distinct_protocols`, `bytes_out_mean`, `packets_out_mean` | |

### 6.2 Đồ thị giao tiếp
Đồ thị có hướng \(G=(V,E)\): đỉnh là IP, cạnh \(u\to v\) nếu \(u\) khởi tạo flow tới \(v\), trọng số = số flow. Độ đo: in/out-degree, weighted degree, degree centrality \(C_D(v) = \deg(v)/(|V|-1)\), betweenness \(C_B(v) = \sum_{s\ne v\ne t} \sigma_{st}(v)/\sigma_{st}\) (xấp xỉ bằng \(k\) đỉnh mẫu), closeness (hiệu chỉnh Wasserman–Faust), PageRank \(PR(v) = \frac{1-\alpha}{|V|} + \alpha\sum_{u\to v} PR(u)\frac{w_{uv}}{\sum_x w_{ux}}\), hệ số cụm.

### 6.3 Centralized vs P2P
Không kết luận từ một độ đo. Bộ phân tích bỏ phiếu trên 4 nhóm bằng chứng: (1) đích chung được ≥ 50% host nghi ngờ liên lạc (ứng viên C&C); (2) mức tập trung lưu lượng: tỉ lệ flow tới đích phổ biến nhất ≥ 0.5 hoặc hệ số Gini ≥ 0.8; (3) liên kết ngang hàng: tỉ lệ cạnh giữa các host nghi ngờ ≥ 5% và số peer trung bình ≥ 50 khi không tập trung; (4) độ phân tán in-degree trong đồ thị lân cận (CV ≥ 2 → dạng sao; < 1 → lưới phẳng). Kết luận cần ≥ 2 phiếu và nhiều hơn phía còn lại; một host đơn lẻ luôn bị trả về "không đủ bằng chứng".

### 6.4 Beaconing
Với mỗi cặp (src, dst, port): sắp xếp thời điểm, gộp các lần bắt đầu cách nhau < 1 s (burst, retransmission, sai số timestamp), tính \(\Delta t_i\), \(\mu\), \(\sigma\), \(CV=\sigma/\mu\) và CV bền vững \(1.4826\cdot MAD/\text{median}\). Beacon khi \(\min(CV, CV_{robust}) \le 0.25\), ≥ 8 kết nối, khoảng quan sát ≥ 60 s, chu kỳ trung vị ≥ 2 s, port không thuộc whitelist (DNS 53, NTP 123). `beacon_score` chỉ dùng để xếp hạng.

## 7. Suy luận và mức độ cảnh báo

| Điều kiện | Kết quả | Severity |
|---|---|---|
| classifier = tấn công, điểm ≥ τ (0.6) | ATTACK (loại tấn công) | theo loại: Botnet/Infiltration CRITICAL; DoS/DDoS/BruteForce/WebAttack HIGH; hạ 1 bậc nếu điểm < 0.8 |
| classifier = tấn công, điểm < τ | `Uncertain` (không khẳng định loại) | LOW, chuyển host sang phân tích botnet |
| Gaussian bất thường, classifier = BENIGN | `Unknown` | MEDIUM (LOW nếu classifier rất chắc BENIGN), chuyển host sang phân tích botnet |
| Beaconing | `Beaconing` | MEDIUM |
| Host model (CTU-13) | `Botnet` | HIGH nếu điểm ≥ 0.8, ngược lại MEDIUM |

"Điểm" của classifier là xác suất ước lượng của mô hình (chưa hiệu chỉnh trên mạng triển khai), dùng như độ tin cậy tương đối, **không** phải xác suất thật sự có tấn công.
