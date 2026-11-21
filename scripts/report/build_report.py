"""Rebuild the written report from the original BAO_CAO_ATTT.docx and the project's saved results.

    python scripts/report/build_report.py [--src BAO_CAO_ATTT.docx] [--out BAO_CAO_ATTT_hoan_thien.docx]

Keeps the cover page, member and task tables, the front-matter section and all styles of the source
file; replaces the abstract, extends the abbreviation table, inserts a table of contents field and
rewrites PHẦN 1 – PHỤ LỤC. All numbers are read from reports/ and models/; supervised models are
reported on the validation split (the test-set evaluation stage has not been run yet).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import docx
import pandas as pd
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from docx_helpers import Writer, clone_row, set_paragraph_text, toc_field  # noqa: E402

M = ROOT / "reports" / "metrics"
EXP = ROOT / "reports" / "experiments"
FIG = ROOT / "reports" / "figures"
RFIG = FIG / "report"


def vn(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def pct(x: float, d: int = 1) -> str:
    return vn(100 * x, d) + "%"


def n(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def load(name: str, base: Path = M):
    return json.loads((base / name).read_text(encoding="utf-8"))


NAMES = {"lr": "Logistic Regression", "dt": "Decision Tree", "rf": "Random Forest", "xgb": "XGBoost",
         "lgbm": "LightGBM", "cat": "CatBoost", "mlp": "MLP (PyTorch)"}


# =====================================================================================================
def front_matter(d, w: Writer) -> None:
    paras = d.paragraphs
    idx = {p.text.strip(): i for i, p in enumerate(paras)}
    tt = idx["TÓM TẮT"]
    abstract = [
        "Đề tài xây dựng một hệ thống phát hiện xâm nhập mạng (NIDS) dựa trên học máy, đi trọn từ dữ liệu "
        "thô đến chương trình giám sát thời gian thực: đọc và kiểm tra chất lượng dữ liệu, tái cấu trúc flow "
        "từ gói tin, chọn đặc trưng, huấn luyện – so sánh mô hình, phát hiện bất thường, phân tích hành vi "
        "botnet ở mức host và đồ thị, và cuối cùng là pipeline phát hiện nhiều tầng chạy trên card mạng.",
        "Bộ dữ liệu chính là CSE-CIC-IDS2018 (16.232.943 flow, 10 file CSV). Kiểm tra dữ liệu cho thấy 26% số "
        "dòng là bản sao, 7/10 file bị cắt đúng ở giới hạn 1.048.575 dòng của Excel, có 22.359 vector đặc "
        "trưng mang hai nhãn mâu thuẫn và cột thời gian dùng đồng hồ 12 giờ không có AM/PM. Sau khi khử trùng "
        "lặp, loại nhãn mâu thuẫn và chia 70/15/15 không chồng lấn, 30 đặc trưng được chọn bằng xếp hạng tổng "
        "hợp và quét k trên tập validation.",
        "Bảy mô hình có giám sát được so sánh trên tập validation (266.305 flow, 7 lớp): Random Forest đạt "
        "Macro-F1 0,917, LightGBM 0,909, XGBoost 0,904, CatBoost 0,899 (FPR thấp nhất 1,3%), MLP 0,865. Ở bài "
        "toán nhị phân, bốn mô hình cây đạt Macro-F1 0,944–0,945. Lớp khó nhất là Infiltration (F1 ≈ 0,56): "
        "khoảng một nửa số flow của lớp này không phân biệt được với lưu lượng bình thường ở mức flow.",
        "Phát hiện bất thường được xây dựng đúng nghĩa Gaussian anomaly detection (học p(x) chỉ từ lưu lượng "
        "bình thường, chọn ngưỡng ε trên validation), không phải Gaussian Naive Bayes. Gaussian đơn biến và đa "
        "biến thất bại (ROC-AUC 0,43 và 0,76) do dữ liệu flow đa cụm; mô hình hỗn hợp Gaussian 16 thành phần "
        "đạt ROC-AUC 0,884 và phát hiện được tấn công chưa từng thấy mà bộ phân loại bỏ sót (DDoS: 0,748 so "
        "với 0,001 ở cùng FPR 5%). Trên PCAP botnet Neris của CTU-13 (71,8 triệu gói, 4,84 triệu flow), đặc "
        "trưng host và đồ thị tách được máy nhiễm; phân tích beaconing cho thấy tính chu kỳ đơn thuần là tín "
        "hiệu yếu, và bộ phân tích topology chỉ kết luận centralized/P2P khi nhiều chỉ số cùng ủng hộ.",
        "Kết quả là một hệ thống có thể chạy trên Linux/Kali: bắt gói bằng Scapy, dựng flow cùng mã với "
        "khi huấn luyện, chấm điểm bằng mô hình triển khai 24 đặc trưng đã kiểm chứng khớp với CICFlowMeter "
        "trên PCAP gốc, và sinh cảnh báo có mức độ nghiêm trọng ra console, JSON, CSV và SQLite.",
    ]
    for i, text in enumerate(abstract):
        set_paragraph_text(paras[tt + 2 + i]._p, text, w)

    # the source already holds a TOC field (inside a content control); Word refreshes it

    abbr = d.tables[3]
    have = {r.cells[0].text.strip() for r in abbr.rows}
    extra = [
        ("PCAP", "Packet Capture", "Định dạng tệp lưu gói tin bắt được"),
        ("CTU-13", "CTU-13 Botnet Dataset", "Bộ dữ liệu botnet của Đại học Kỹ thuật Séc (CTU)"),
        ("GMM", "Gaussian Mixture Model", "Mô hình hỗn hợp Gaussian"),
        ("LOFO", "Leave-One-Family-Out", "Giấu một họ tấn công khi huấn luyện để mô phỏng tấn công mới"),
        ("FPR / FNR", "False Positive / Negative Rate", "Tỷ lệ báo động giả / tỷ lệ bỏ sót"),
        ("ROC-AUC / PR-AUC", "Area Under ROC / Precision-Recall Curve", "Diện tích dưới đường ROC / PR"),
        ("MLP", "Multilayer Perceptron", "Mạng nơ-ron truyền thẳng nhiều lớp"),
        ("CV / MAD", "Coefficient of Variation / Median Absolute Deviation",
         "Hệ số biến thiên / độ lệch tuyệt đối trung vị"),
        ("SMOTE", "Synthetic Minority Over-sampling Technique", "Sinh mẫu tổng hợp cho lớp thiểu số"),
        ("EDA", "Exploratory Data Analysis", "Phân tích khám phá dữ liệu"),
        ("IRC", "Internet Relay Chat", "Giao thức chat, thường bị botnet dùng làm kênh C&C"),
    ]
    for a, b, c in extra:
        if a not in have:
            clone_row(abbr, [a, b, c])


# =====================================================================================================
def part1(w: Writer) -> None:
    w.h1("PHẦN 1: TỔNG QUAN ĐỀ TÀI")
    w.h2("1.1. Bối cảnh và đặt vấn đề")
    w.para("Lưu lượng mạng của các tổ chức ngày càng lớn và đa dạng, trong khi các cuộc tấn công như từ chối "
           "dịch vụ (DoS/DDoS), dò mật khẩu, botnet, tấn công ứng dụng web hay xâm nhập nội bộ ngày càng khó "
           "phát hiện bằng luật tĩnh. Hệ thống phát hiện xâm nhập dựa trên học máy học các đặc trưng thống kê "
           "của từng luồng (flow) để phân biệt lưu lượng bình thường với các nhóm tấn công.")
    w.para("Phần lớn các công trình trên bộ dữ liệu CIC dừng ở việc báo cáo độ chính xác rất cao trên một tập "
           "test ngẫu nhiên. Khi bắt tay vào làm, đề tài gặp các vấn đề mà con số đó che khuất: dữ liệu trùng "
           "lặp hàng triệu dòng (làm điểm số bị thổi phồng nếu bản sao rơi vào cả train và test), nhãn mâu "
           "thuẫn, tệp bị cắt, mất cân bằng cực lớn, và quan trọng nhất là khoảng cách giữa dữ liệu CSV đã "
           "trích sẵn với gói tin thật mà hệ thống phải xử lý khi triển khai.")
    w.para("Vì vậy đề tài đặt mục tiêu xây dựng một hệ thống **có thể kiểm chứng ở từng bước**: mọi quyết định "
           "tiền xử lý, chọn đặc trưng, chọn siêu tham số và ngưỡng đều dựa trên tập train/validation; mọi "
           "giới hạn được ghi nhận bằng thực nghiệm thay vì giả định; và cùng một đoạn mã tạo đặc trưng được "
           "dùng cho cả huấn luyện lẫn giám sát thời gian thực.")

    w.h2("1.2. Bài toán")
    w.para("Bài toán được phát biểu ở bốn mức, tương ứng các tầng của hệ thống:")
    w.bullets([
        "**Mức flow – phân loại có giám sát:** với mỗi flow x ∈ ℝᵈ, dự đoán nhãn y thuộc 7 họ {BENIGN, Botnet, "
        "BruteForce, DDoS, DoS, Infiltration, WebAttack} (bài toán đa lớp) hoặc {BENIGN, ATTACK} (nhị phân).",
        "**Mức flow – phát hiện bất thường:** học hàm mật độ p(x) chỉ từ flow bình thường; flow có "
        "log p(x) < ε bị coi là bất thường. Mục tiêu là bắt được tấn công không có trong dữ liệu huấn luyện.",
        "**Mức host – phân tích botnet:** gom flow theo máy nguồn trong cửa sổ thời gian, tính đặc trưng "
        "hành vi (số đích, entropy cổng, kết nối thất bại, beaconing, chỉ số đồ thị) để phát hiện máy nhiễm "
        "và nhận diện cấu trúc điều khiển (tập trung hay ngang hàng).",
        "**Mức hệ thống – thời gian thực:** bắt gói trên card mạng, dựng flow, chấm điểm và sinh cảnh báo với "
        "độ trễ, tài nguyên đo được, và đặc trưng phải khớp với dữ liệu huấn luyện.",
    ])

    w.h2("1.3. Mô hình đe dọa")
    w.h3("1.3.1. Vị trí quan sát và giả định")
    w.bullets([
        "Hệ thống đặt tại một điểm quan sát được lưu lượng hai chiều (cổng mạng, port mirror/SPAN, hoặc chính "
        "máy Kali trong phòng lab). Hệ thống chỉ dùng tiêu đề gói và thống kê thời gian, không đọc payload, "
        "nên vẫn áp dụng được với lưu lượng mã hoá.",
        "Nhãn huấn luyện lấy từ cột Label chính thức của CSE-CIC-IDS2018. Với CTU-13 kịch bản 1, nhãn ở mức "
        "host theo tài liệu của bộ dữ liệu và được kiểm chứng lại bằng hành vi quan sát được (mục 3.10).",
        "Kẻ tấn công không biết và không chủ động né mô hình (không xét tấn công đối kháng vào mô hình học máy).",
    ])
    w.h3("1.3.2. Các nhóm hành vi tấn công cần phát hiện")
    w.table("Các họ tấn công và nhãn gốc trong CSE-CIC-IDS2018",
            ["Họ (family)", "Nhãn gốc", "Đặc điểm hành vi ở mức flow"],
            [["BENIGN", "Benign", "Lưu lượng người dùng và dịch vụ trong testbed"],
             ["Botnet", "Bot", "Máy nhiễm (Ares/Zeus) liên lạc định kỳ với máy C&C qua HTTP"],
             ["BruteForce", "BruteForce-FTP, BruteForce-SSH", "Nhiều kết nối ngắn, kích thước gói gần như cố định"],
             ["DoS", "DoS-Hulk, -SlowHTTPTest, -GoldenEye, -Slowloris",
              "Một nguồn làm cạn tài nguyên web: flood HTTP hoặc giữ kết nối mở chậm"],
             ["DDoS", "DDoS-HOIC, -LOIC-HTTP, -LOIC-UDP", "Nhiều nguồn cùng flood HTTP/UDP"],
             ["Infiltration", "Infiltration", "Máy nội bộ bị chiếm, quét và di chuyển ngang – gần giống bình thường"],
             ["WebAttack", "Web-BruteForce, Web-XSS, Web-SQLInjection",
              "Tấn công nằm trong nội dung HTTP; dấu hiệu ở mức flow rất yếu"]],
            [2.2, 3.6, 5.2])
    w.h3("1.3.3. Mối đe dọa đối với chính hệ thống IDS")
    w.bullets([
        "**Tấn công chưa từng thấy:** bộ phân loại chỉ nhận ra họ tấn công có trong dữ liệu huấn luyện – cần "
        "tầng phát hiện bất thường độc lập với nhãn.",
        "**Dịch chuyển phân phối:** mô hình học trên testbed năm 2018 có thể sai lệch trên mạng khác; ngưỡng "
        "bất thường phải hiệu chỉnh lại bằng lưu lượng bình thường của chính mạng triển khai.",
        "**Lệch đặc trưng offline/online:** nếu đặc trưng tính lúc chạy thật khác đặc trưng của CICFlowMeter, "
        "mô hình nhận đầu vào nằm ngoài phân phối và báo động sai hàng loạt.",
        "**Quá tải:** DoS cũng tấn công chính IDS – hàng đợi gói phải có giới hạn, gói bị bỏ phải được đếm và "
        "cảnh báo lặp phải được gộp.",
    ])

    w.h2("1.4. Câu hỏi nghiên cứu")
    w.bullets([
        "**RQ1.** Dữ liệu CSE-CIC-IDS2018 có những vấn đề chất lượng nào, và cần xử lý thế nào để đánh giá "
        "không bị rò rỉ giữa train/validation/test?",
        "**RQ2.** Trong các mô hình cây (RF, XGBoost, LightGBM, CatBoost), mạng nơ-ron MLP và hai baseline "
        "(LR, DT), mô hình nào tốt nhất theo Macro-F1 và FPR? Lớp nào khó và vì sao?",
        "**RQ3.** Gaussian anomaly detection học chỉ từ lưu lượng bình thường có phát hiện được tấn công, đặc "
        "biệt là tấn công chưa có trong dữ liệu huấn luyện, hay không?",
        "**RQ4.** Đặc trưng mức host, beaconing và đồ thị liên lạc có phát hiện được máy nhiễm botnet trong "
        "CTU-13, và có phân biệt được botnet tập trung với P2P hay không?",
        "**RQ5.** Đặc trưng dựng từ gói tin có khớp với CICFlowMeter để mô hình học trên CSV dùng được cho "
        "giám sát thời gian thực hay không?",
    ])

    w.h2("1.5. Mục tiêu và phạm vi")
    w.h3("1.5.1. Mục tiêu")
    w.bullets([
        "Kiểm tra và làm sạch CSE-CIC-IDS2018 bằng xử lý dòng (streaming) trên toàn bộ 16,2 triệu flow.",
        "Chọn đặc trưng, so sánh cách chuẩn hoá và cách xử lý mất cân bằng; mọi lựa chọn dựa trên validation.",
        "Huấn luyện và so sánh 7 mô hình có giám sát; Gaussian đơn biến, đa biến, hỗn hợp; Isolation Forest và "
        "One-Class SVM làm baseline bất thường.",
        "Phân tích botnet trên PCAP CTU-13: dựng flow, đặc trưng host, đồ thị, beaconing, topology.",
        "Xây dựng pipeline phát hiện nhiều tầng và chương trình realtime chạy trên Kali Linux.",
    ])
    w.h3("1.5.2. Phạm vi")
    w.table("Phạm vi của đề tài",
            ["Hạng mục", "Phạm vi"],
            [["Dữ liệu chính", "CSE-CIC-IDS2018: 10 file CSV, 16.232.943 flow, 78 đặc trưng CICFlowMeter"],
             ["Dữ liệu botnet", "CTU-13 kịch bản 1 (Neris): capture20110810.truncated.pcap, 71.829.068 gói"],
             ["Kiểm chứng parity", "Một PCAP gốc của CICIDS2018 (ngày 02/03/2018, máy nạn nhân Bot)"],
             ["Mô hình", "LR, DT, RF, XGBoost, LightGBM, CatBoost, MLP; Gaussian UV/MV/GMM, Isolation Forest, OCSVM"],
             ["Triển khai", "Python 3.12, Scapy; chạy offline (PCAP) và realtime (card mạng)"],
             ["Ngoài phạm vi", "Phân tích payload, tấn công đối kháng vào mô hình, mạng tốc độ > 1 Gbps"]],
            [2.6, 8.4])

    w.h2("1.6. Nghiên cứu liên quan")
    w.para("CSE-CIC-IDS2018 do Canadian Institute for Cybersecurity và Communications Security Establishment xây "
           "dựng trên hạ tầng AWS, đặc trưng flow được trích bằng CICFlowMeter [1][2][3]. Nhiều công trình báo "
           "cáo độ chính xác trên 99% với cây quyết định, Random Forest hay mạng nơ-ron; tuy nhiên các nghiên "
           "cứu phân tích lỗi chỉ ra vấn đề gán nhãn, flow bị cắt và artifact của CICFlowMeter, như hiện tượng "
           "“TCP appendix” sau gói FIN [11][12]. Đề tài gặp lại đúng các vấn đề này và đo ảnh hưởng của chúng.")
    w.para("Gradient boosting (XGBoost, LightGBM, CatBoost) là nhóm mô hình mạnh nhất cho dữ liệu dạng bảng "
           "[13][14][15]. Về phát hiện bất thường, mô hình mật độ Gaussian, Isolation Forest và One-Class SVM là "
           "các hướng kinh điển [7][8][16]. Với botnet, CTU-13 [17] là bộ dữ liệu chuẩn có PCAP gốc; phát hiện "
           "botnet thường dựa trên hành vi theo host, tính chu kỳ của kết nối C&C và cấu trúc đồ thị liên lạc.")
    w.table("Các hướng nghiên cứu liên quan và vai trò trong đề tài",
            ["Hướng", "Ý tưởng chính", "Vai trò trong đề tài"],
            [["Supervised IDS", "Học ranh giới giữa bình thường và từng loại tấn công", "Tầng phân loại (LightGBM)"],
             ["Anomaly IDS", "Mô hình hoá lưu lượng bình thường, cảnh báo khi lệch", "Tầng sàng lọc (Gaussian mixture)"],
             ["Host-based behaviour", "Đặc trưng theo máy trong cửa sổ thời gian", "Phát hiện quét, máy nhiễm"],
             ["Beaconing", "Tìm kết nối lặp đều đặn tới C&C", "Tín hiệu bổ trợ, không dùng đơn lẻ"],
             ["Graph analysis", "Bậc, độ trung gian, đích hiếm dùng chung", "Phân loại topology botnet"]],
            [2.4, 4.6, 4.0])

    w.h2("1.7. Bố cục báo cáo")
    w.bullets([
        "Phần 1 – Tổng quan: bối cảnh, bài toán, mô hình đe dọa, câu hỏi nghiên cứu, mục tiêu, phạm vi.",
        "Phần 2 – Cơ sở lý thuyết: flow, IDS, các loại tấn công, mô hình học máy, Gaussian anomaly detection, "
        "beaconing và đồ thị botnet, mất cân bằng, chỉ số đánh giá, rò rỉ dữ liệu.",
        "Phần 3 – Triển khai: kiến trúc, dữ liệu và chất lượng dữ liệu, tiền xử lý, chọn đặc trưng, huấn luyện, "
        "phát hiện bất thường, dựng flow từ PCAP, phân tích botnet, pipeline nhiều tầng và realtime.",
        "Phần 4 – Thực nghiệm và đánh giá: kết quả từng tầng, phân tích lỗi, hiệu năng, trả lời câu hỏi nghiên cứu.",
        "Phần 5 – Kết luận, hạn chế và hướng phát triển. Phần 6 – Tài liệu tham khảo; Phụ lục.",
    ])


# =====================================================================================================
def part2(w: Writer) -> None:
    w.h1("PHẦN 2: CƠ SỞ LÝ THUYẾT")
    w.h2("2.1. Gói tin, flow và CICFlowMeter")
    w.para("Gói tin (packet) là đơn vị dữ liệu truyền trên mạng, gồm tiêu đề các tầng (Ethernet, IP, TCP/UDP) "
           "và phần tải (payload). Flow là tập các gói có cùng bộ 5 thành phần (IP nguồn, IP đích, cổng nguồn, "
           "cổng đích, giao thức) trong một khoảng thời gian. CICFlowMeter dựng flow **hai chiều**: gói đầu tiên "
           "xác định chiều đi (forward), gói theo chiều ngược lại thuộc chiều về (backward).")
    w.para("Một flow TCP kết thúc khi gặp cờ FIN hoặc RST, hoặc khi đạt active timeout 120 giây tính từ gói "
           "đầu. Các gói đến sau FIN của cùng kết nối tạo thành một flow mới – hiện tượng “TCP appendix” đã "
           "được ghi nhận trong các nghiên cứu về lỗi của CICFlowMeter. Mỗi flow được mô tả bởi 78 đặc trưng "
           "thống kê:")
    w.table("Các nhóm đặc trưng flow của CICFlowMeter",
            ["Nhóm đặc trưng", "Ví dụ", "Ý nghĩa"],
            [["Thời lượng, số lượng", "Flow Duration, Tot Fwd/Bwd Pkts", "Độ dài và độ đối xứng của phiên"],
             ["Kích thước gói", "Fwd/Bwd Pkt Len Max/Mean/Std", "Loại ứng dụng; flood thường có kích thước cố định"],
             ["Tốc độ", "Flow Byts/s, Flow Pkts/s", "Cường độ lưu lượng; DoS/DDoS có tốc độ cao"],
             ["Khoảng thời gian giữa gói", "Flow/Fwd/Bwd IAT Mean/Std/Max/Min", "Nhịp gửi; Slowloris có IAT lớn, đều"],
             ["Cờ TCP", "SYN/FIN/RST/PSH/ACK Flag Cnt", "Trạng thái kết nối, quét cổng"],
             ["Tiêu đề, cửa sổ", "Fwd/Bwd Header Len, Init Fwd/Bwd Win Byts", "Đặc điểm hệ điều hành và công cụ"],
             ["Active/Idle", "Active Mean/Max, Idle Mean/Max", "Chu kỳ hoạt động/nghỉ trong flow (ngưỡng 5 s)"],
             ["Bulk, subflow", "Fwd Byts/b Avg, Subflow Fwd Pkts", "Truyền khối; trong dữ liệu 6 cột bulk luôn bằng 0"]],
            [2.6, 4.2, 4.2])

    w.h2("2.2. Hệ thống phát hiện xâm nhập")
    w.para("IDS được phân loại theo vị trí đặt (host-based, network-based) và phương pháp phát hiện:")
    w.bullets([
        "**Dựa trên chữ ký:** so khớp mẫu tấn công đã biết; chính xác nhưng không phát hiện biến thể mới.",
        "**Dựa trên bất thường:** mô hình hoá hành vi bình thường, cảnh báo khi lệch; bắt được tấn công mới "
        "nhưng dễ báo động giả và phụ thuộc mạng cụ thể.",
        "**Học máy có giám sát:** học từ dữ liệu có nhãn; mạnh với tấn công đã biết, phụ thuộc chất lượng nhãn.",
    ])
    w.para("Đề tài kết hợp hai hướng sau trong một kiến trúc nhiều tầng và bổ sung tầng phân tích mức host, "
           "nơi các hành vi như quét cổng hay beaconing chỉ thể hiện khi nhìn nhiều flow cùng lúc.")

    w.h2("2.3. Các loại tấn công")
    for title, text in [
        ("2.3.1. Botnet", "Máy nhiễm mã độc liên lạc với máy chủ điều khiển (C&C) để nhận lệnh. Botnet tập trung "
         "(IRC, HTTP) có một hoặc vài máy C&C mà mọi bot cùng kết nối; botnet ngang hàng (P2P) không có điểm "
         "trung tâm, các bot liên lạc với nhau và với nhiều peer. Dấu hiệu: kết nối định kỳ (beaconing), nhiều "
         "máy nội bộ cùng liên lạc với một đích hiếm, gửi thư rác hoặc quét hàng loạt."),
        ("2.3.2. Brute Force", "Thử liên tục nhiều cặp tên đăng nhập/mật khẩu vào SSH hoặc FTP. Ở mức flow là "
         "nhiều kết nối ngắn tới cùng cổng, kích thước gói gần như giống hệt nhau."),
        ("2.3.3. DoS và DDoS", "DoS dùng một nguồn làm cạn tài nguyên: flood HTTP (Hulk), giữ kết nối keep-alive "
         "(GoldenEye), hoặc gửi request rất chậm để chiếm luồng xử lý (Slowloris, SlowHTTPTest). DDoS dùng "
         "nhiều nguồn cùng lúc (LOIC-HTTP, HOIC, LOIC-UDP)."),
        ("2.3.4. Infiltration", "Kẻ tấn công đưa mã độc vào máy nội bộ (qua tài liệu độc hại), sau đó quét và di "
         "chuyển ngang. Phần lớn flow của máy bị chiếm vẫn là lưu lượng bình thường nên đây là lớp khó nhất."),
        ("2.3.5. Web Attack", "Brute force đăng nhập web, XSS và SQL Injection. Tấn công nằm trong nội dung HTTP "
         "mà CICFlowMeter không lưu; số mẫu rất ít (928 dòng trên 16 triệu)."),
        ("2.3.6. Quét cổng và quét host", "Một nguồn gửi SYN tới nhiều cổng (port scan) hoặc nhiều địa chỉ (host "
         "sweep). Mỗi gói SYN không được trả lời là một flow một gói – CICFlowMeter không xuất loại flow này, "
         "nên quét chỉ phát hiện được ở mức host."),
    ]:
        w.h3(title)
        w.para(text)

    w.h2("2.4. Các mô hình học có giám sát")
    w.para("**Decision Tree** chia không gian đặc trưng bằng chuỗi phép tách theo ngưỡng, chọn phép tách giảm "
           "độ hỗn tạp Gini nhiều nhất; dễ giải thích nhưng phương sai cao. **Random Forest** [4] lấy biểu quyết "
           "của nhiều cây học trên mẫu bootstrap và tập con đặc trưng ngẫu nhiên, giảm phương sai so với một cây.")
    w.para("**Gradient boosting** xây cây tuần tự, mỗi cây mới khớp với gradient của hàm mất mát còn lại. "
           "XGBoost [13] thêm chính quy hoá L1/L2 và xấp xỉ bậc hai; LightGBM [14] phát triển cây theo lá "
           "(leaf-wise) và gom giá trị thành histogram nên rất nhanh; CatBoost [15] dùng cây đối xứng và "
           "ordered boosting để giảm lệch. **Logistic Regression** là mô hình tuyến tính, dùng làm baseline.")
    w.para("**MLP** (multilayer perceptron) là mạng nơ-ron truyền thẳng. Đề tài dùng kiến trúc 256–128–64 với "
           "BatchNorm, ReLU và Dropout sau mỗi lớp ẩn, tối ưu bằng Adam (lr = 10⁻³), giảm learning rate khi "
           "loss validation không giảm (ReduceLROnPlateau) và dừng sớm (early stopping).")
    w.table("So sánh các nhóm mô hình có giám sát",
            ["Mô hình", "Ưu điểm", "Hạn chế"],
            [["Logistic Regression", "Đơn giản, nhanh, có xác suất", "Ranh giới tuyến tính, nhạy thang đo"],
             ["Decision Tree", "Nhanh, dễ giải thích, bắt quan hệ phi tuyến", "Dễ quá khớp, phương sai cao"],
             ["Random Forest", "Ổn định, ít cần tinh chỉnh", "Mô hình rất lớn (GB), suy luận chậm"],
             ["XGBoost / LightGBM / CatBoost", "Chính xác nhất trên dữ liệu bảng, mô hình gọn",
              "Nhiều siêu tham số; LightGBM có thể phân kỳ (mục 3.6.2)"],
             ["MLP", "Học biểu diễn phi tuyến, mở rộng tốt", "Cần chuẩn hoá, huấn luyện lâu, khó giải thích"]],
            [3.2, 4.0, 3.8])

    w.h2("2.5. Phát hiện bất thường bằng phân phối Gaussian")
    w.para("Gaussian anomaly detection ước lượng mật độ p(x) của lưu lượng **bình thường** rồi đánh dấu x là "
           "bất thường khi p(x) < ε. Đây **không phải** Gaussian Naive Bayes: GNB là bộ phân loại có giám sát "
           "học một phân phối cho mỗi lớp, còn ở đây chỉ có một phân phối, học không cần nhãn tấn công.")
    w.bullets([
        "**Đơn biến:** p(x) = ∏ⱼ N(xⱼ; μⱼ, σⱼ²), giả định các đặc trưng độc lập.",
        "**Đa biến:** p(x) = N(x; μ, Σ) với ma trận hiệp phương sai đầy đủ; Σ được cộng λI (λ = 10⁻³) và phân "
        "rã Cholesky để ổn định số học.",
        "**Hỗn hợp Gaussian (GMM):** p(x) = Σₖ πₖ N(x; μₖ, Σₖ), học bằng EM; mô tả được dữ liệu có nhiều cụm "
        "hành vi (web, DNS, SSH…) mà một Gaussian đơn không mô tả nổi.",
    ])
    w.para("Mọi tính toán dùng log p(x) để tránh tràn số. Ngưỡng ε được chọn trên tập validation theo hai cách: "
           "cực đại F1, hoặc đặt tại phân vị của log p(x) trên flow bình thường để đạt FPR mục tiêu (1% hoặc 5%). "
           "Điểm bất thường −log p(x) **không phải xác suất bị tấn công**: nó chỉ cho biết flow hiếm đến mức nào "
           "so với lưu lượng bình thường đã học. Trước khi ước lượng, đặc trưng được biến đổi "
           "sign(x)·log(1+|x|) rồi chuẩn hoá, vì dữ liệu flow lệch phải rất mạnh.")
    w.para("Hai baseline được dùng để so sánh: **Isolation Forest** [8] cô lập điểm bất thường bằng các phép tách "
           "ngẫu nhiên (điểm càng dễ cô lập càng bất thường) và **One-Class SVM** [16] tìm siêu phẳng bao lấy "
           "dữ liệu bình thường trong không gian kernel.")

    w.h2("2.6. Beaconing và phân tích đồ thị botnet")
    w.para("Beaconing là hành vi kết nối lặp lại theo chu kỳ gần cố định giữa máy nhiễm và C&C. Với dãy thời "
           "điểm kết nối t₁ < t₂ < … của một cặp (nguồn, đích, cổng), tính khoảng cách Δᵢ = tᵢ₊₁ − tᵢ và hệ số "
           "biến thiên CV = σ(Δ)/μ(Δ); CV càng nhỏ, chu kỳ càng đều. Vì một vài khoảng nghỉ dài làm CV tăng vọt, "
           "đề tài dùng thêm CV bền vững = MAD(Δ)/median(Δ). Nhiều ứng dụng hợp lệ (NTP, cập nhật, keep-alive) "
           "cũng có chu kỳ, nên beaconing **không được dùng làm bằng chứng duy nhất**.")
    w.para("Phân tích đồ thị biểu diễn mạng thành đồ thị có hướng: đỉnh là địa chỉ IP, cạnh là liên lạc có "
           "trọng số theo số flow. Các chỉ số dùng gồm bậc vào/ra, độ trung gian (betweenness), hệ số phân cụm, "
           "và số máy nghi vấn cùng liên lạc với một đích hiếm. Botnet tập trung thể hiện qua một vài đích hiếm "
           "được nhiều bot cùng kết nối (fan-in cao); botnet P2P thể hiện qua các bot liên lạc trực tiếp với "
           "nhau và dùng chung nhiều peer hiếm. Kết luận topology chỉ được đưa ra khi nhiều chỉ số cùng ủng hộ.")

    w.h2("2.7. Mất cân bằng dữ liệu")
    w.para("Trong dữ liệu gốc, BENIGN chiếm 83% còn Web-SQLInjection chỉ có 87 dòng (tỷ lệ khoảng 155.000 lần). "
           "Có hai hướng xử lý: mức dữ liệu (undersampling lớp đa số, SMOTE [18] sinh mẫu tổng hợp cho lớp "
           "thiểu số) và mức thuật toán (trọng số lớp tỷ lệ nghịch với tần suất). Nguyên tắc bắt buộc: chỉ "
           "resample **tập train**; validation và test giữ nguyên phân phối thật.")

    w.h2("2.8. Chỉ số đánh giá")
    w.table("Các chỉ số đánh giá sử dụng trong đề tài",
            ["Chỉ số", "Định nghĩa / ý nghĩa", "Vai trò"],
            [["Precision", "TP / (TP + FP): tỷ lệ cảnh báo đúng", "Đo mức báo động giả"],
             ["Recall", "TP / (TP + FN): tỷ lệ tấn công bắt được", "Đo mức bỏ sót"],
             ["Macro-F1", "Trung bình F1 của các lớp, mỗi lớp trọng số như nhau", "Chỉ số chính để chọn mô hình"],
             ["Weighted-F1", "Trung bình F1 theo số mẫu mỗi lớp", "Tham khảo; bị lớp lớn chi phối"],
             ["FPR", "Tỷ lệ flow BENIGN bị dự đoán là tấn công", "Chi phí vận hành (số cảnh báo giả)"],
             ["FNR", "Tỷ lệ flow tấn công bị dự đoán là BENIGN", "Rủi ro bỏ lọt"],
             ["ROC-AUC (OvR)", "Diện tích dưới đường ROC, one-vs-rest, macro", "Khả năng xếp hạng"],
             ["PR-AUC (OvR)", "Diện tích dưới đường Precision–Recall", "Phù hợp khi mất cân bằng [10]"],
             ["Thời gian, kích thước", "Thời gian train, µs/flow khi suy luận, MB", "Khả năng triển khai realtime"]],
            [2.4, 5.0, 3.6])
    w.para("Với nhiều lớp, FPR và FNR được tính theo góc nhìn nhị phân: mọi lớp tấn công gộp thành ATTACK. "
           "Accuracy không được dùng để kết luận vì bị lớp BENIGN chi phối.")

    w.h2("2.9. Rò rỉ dữ liệu, quá khớp và học lối tắt")
    w.bullets([
        "**Rò rỉ dữ liệu:** thông tin của test lọt vào huấn luyện – fit scaler trên toàn bộ dữ liệu, chọn "
        "ngưỡng trên test, hoặc **cùng một dòng trùng lặp nằm ở cả train và test**. Với 26% dòng trùng lặp, "
        "rủi ro thứ ba là đáng kể nhất trong CSE-CIC-IDS2018.",
        "**Quá khớp:** mô hình học thuộc train; nhận biết qua khoảng cách giữa điểm train và validation.",
        "**Học lối tắt:** mô hình dựa vào artifact của testbed (ví dụ cổng đích, kích thước cửa sổ TCP của "
        "công cụ tấn công) thay vì hành vi; vì vậy cần kiểm tra trên dữ liệu khác (CTU-13, PCAP thật).",
    ])


# =====================================================================================================
def part3(w: Writer) -> None:
    w.h1("PHẦN 3: TRIỂN KHAI HỆ THỐNG")
    w.h2("3.1. Môi trường và công cụ")
    w.table("Môi trường và công cụ triển khai",
            ["Thành phần", "Chi tiết"],
            [["Ngôn ngữ, môi trường", "Python 3.12.3, virtualenv; Windows 10 (phát triển), Kali Linux (triển khai)"],
             ["Xử lý dữ liệu", "pandas 3.1.1, NumPy 2.5.5, PyArrow 26.0.1 (đọc CSV dạng luồng)"],
             ["Học máy", "scikit-learn 1.9.1, XGBoost 3.4.1, LightGBM 4.7.0, CatBoost 1.2.10, imbalanced-learn 0.14.2"],
             ["Học sâu", "PyTorch 2.15.0 (CPU)"],
             ["Gói tin, đồ thị", "dpkt 1.9.8 (đọc PCAP offline), Scapy 2.8.0 (bắt gói realtime), NetworkX 3.7"],
             ["Trực quan, kiểm thử", "matplotlib 3.11.2, Jupyter; pytest (38 kiểm thử đơn vị và tích hợp)"],
             ["Phần cứng", "Laptop 16 GB RAM, CPU; không dùng GPU"]],
            [2.8, 8.2])

    w.h2("3.2. Kiến trúc tổng thể")
    w.para("Hệ thống gồm bốn khối: nguồn dữ liệu, tái cấu trúc flow, tạo đặc trưng và ba tầng phát hiện, cuối "
           "cùng là bộ quyết định sinh cảnh báo. Điểm thiết kế quan trọng nhất là **chỉ có một đường đi** từ "
           "gói tin đến đặc trưng: cùng hàm `decode_frame()` và lớp `FlowBuilder` được dùng khi đọc PCAP offline "
           "và khi bắt gói trực tiếp, nên đặc trưng lúc chạy thật không thể lệch khỏi đặc trưng lúc đánh giá.")
    w.figure(RFIG / "architecture.png", "Kiến trúc tổng thể của hệ thống phát hiện xâm nhập", 15.5)
    w.table("Các module mã nguồn chính",
            ["Module", "Chức năng"],
            [["src/data_processing", "Giải mã gói, dựng flow hai chiều, 78 đặc trưng, đọc CICIDS2018, làm sạch, chia dữ liệu"],
             ["src/models", "RF, XGBoost, LightGBM, CatBoost, baseline, MLP (PyTorch), Gaussian UV/MV/GMM"],
             ["src/graph", "Dựng đồ thị liên lạc, chỉ số đỉnh, phát hiện beaconing, phân tích topology"],
             ["src/detection", "Bộ quyết định nhiều tầng, mức độ nghiêm trọng, quản lý và lưu cảnh báo"],
             ["src/realtime", "Bắt gói bằng Scapy, replay PCAP, bộ xử lý luồng, đo độ trễ và tài nguyên"],
             ["scripts/", "Các bước pipeline: tiền xử lý, huấn luyện, đánh giá, botnet, parity, realtime, lab"],
             ["tests/", "38 kiểm thử: parser, flow, tiền xử lý, mô hình, Gaussian, đồ thị, pipeline end-to-end"]],
            [2.8, 8.2])

    # ---------------------------------------------------------------- data
    ins = load("cicids_inspection.json")
    w.h2("3.3. Bộ dữ liệu")
    w.h3("3.3.1. CSE-CIC-IDS2018")
    w.para(f"Bộ dữ liệu gồm 10 file CSV (khoảng 6,9 GB), mỗi file là một ngày thu thập từ 14/02 đến 02/03/2018, "
           f"tổng {n(ins['total_rows'])} flow với 78 đặc trưng số và cột Label. Mỗi ngày chỉ chứa một đến ba loại "
           "tấn công, nên ngày thu thập gắn chặt với nhãn:")
    rows = []
    for day, labels in ins["label_counts_by_day"].items():
        attacks = {k: v for k, v in labels.items() if k != "Benign"}
        dd = day[8:10] + "/" + day[5:7]
        rows.append([dd, n(labels.get("Benign", 0)), ", ".join(f"{k} ({n(v)})" for k, v in attacks.items())])
    w.table("Phân bố nhãn theo ngày thu thập", ["Ngày", "Benign", "Tấn công (số flow)"], rows, [1.3, 2.0, 7.7],
            align="crl", size=21)
    w.h3("3.3.2. CTU-13 kịch bản 1 (botnet Neris)")
    w.para("CTU-13 [17] gồm 13 kịch bản botnet thực thu tại mạng Đại học Kỹ thuật Séc năm 2011. Đề tài dùng "
           "kịch bản 1: tệp capture20110810.truncated.pcap (71.829.068 gói, 6 giờ 7 phút), trong đó máy "
           "147.32.84.165 nhiễm Neris – botnet điều khiển qua IRC, gửi thư rác và gian lận click. Tệp đã bị cắt "
           "phần payload (truncated) nhưng giữ nguyên tiêu đề. Nhãn chính thức mức flow (.binetflow) không tải "
           "được vì máy chủ của CTU không truy cập được trong thời gian thực hiện, nên nhãn được gán theo host: "
           "1 máy nhiễm, 6 máy bình thường theo tài liệu của bộ dữ liệu, các máy còn lại là Background (không "
           "biết nhãn).")
    w.h3("3.3.3. PCAP kiểm chứng")
    w.para("Để kiểm tra đặc trưng tự dựng có khớp CICFlowMeter hay không, đề tài tải một tệp PCAP gốc của "
           "CICIDS2018 (ngày 02/03/2018, máy nạn nhân Bot 172.31.69.26). Vì archive của ngày này nặng 44,8 GB, "
           "script đọc thư mục ZIP qua HTTP Range và chỉ tải đúng tệp cần thiết. Ngoài ra một PCAP bài tập "
           "(không nhãn) được dùng để thử parser và pipeline trên mạng hoàn toàn khác.")

    w.h2("3.4. Kiểm tra chất lượng dữ liệu")
    w.para("Toàn bộ 16,2 triệu dòng được đọc theo khối 250.000 dòng bằng PyArrow (mọi cột đọc dạng chuỗi rồi "
           "mới chuyển kiểu, để phát hiện giá trị lỗi). Mỗi dòng được băm (hash) trên toàn bộ đặc trưng để đếm "
           "trùng lặp mà không cần nạp hết dữ liệu vào RAM. Kết quả:")
    uniq = ins["unique_rows_per_label"]
    raw = ins["label_counts"]
    w.table("Các vấn đề chất lượng dữ liệu phát hiện được",
            ["Vấn đề", "Mức độ", "Cách xử lý"],
            [["Dòng trùng lặp (cả đặc trưng và nhãn)",
              f"{n(ins['duplicate_rows_features_and_label'])} dòng (26,1%)", "Khử trùng lặp trước khi chia"],
             ["Lớp gần như chỉ gồm bản sao",
              f"BruteForce-FTP: {n(uniq['BruteForce-FTP'])}/{n(raw['BruteForce-FTP'])} dòng duy nhất; "
              f"SlowHTTPTest: {n(uniq['DoS-SlowHTTPTest'])}/{n(raw['DoS-SlowHTTPTest'])}",
              "Ghi nhận: điểm của hai lớp này gần như vô nghĩa"],
             ["Vector đặc trưng mang nhiều nhãn",
              f"{n(ins['conflicting_label_feature_vectors'])} vector ({n(ins['rows_in_conflicting_vectors'])} dòng)",
              "Loại khỏi tập làm việc"],
             ["File bị cắt ở giới hạn Excel", "7/10 file có đúng 1.048.575 dòng dữ liệu", "Không khắc phục được; ghi nhận"],
             ["Dòng tiêu đề lặp giữa file", "59 dòng ở 3 file", "Loại khi đọc"],
             ["Giá trị +Inf / NaN", f"{n(ins['total_pos_inf_cells'])} / {n(ins['total_nan_cells'])} ô (cột tốc độ)",
              "Thay bằng max hữu hạn / 0, fit trên train"],
             ["Cột hằng số", "8 cột (6 cột bulk, Bwd PSH/URG Flags)", "Loại bỏ"],
             ["Cột thời gian 12 giờ không AM/PM", "Giờ 01–07 thực chất là buổi chiều",
              "Cộng 12 giờ cho giờ < 8; mỗi ngày nằm trong 08:28–17:30"],
             ["Cột IP chỉ có ở một file", "Chỉ file 20/02 có Flow ID, Src IP, Src Port, Dst IP",
              "Không dùng; phân tích mức host thực hiện trên CTU-13"]],
            [3.4, 4.2, 3.4], size=21)
    w.para("Vấn đề trùng lặp có hệ quả lớn: nếu chia ngẫu nhiên trên dữ liệu gốc, phần lớn dòng test của "
           "FTP-BruteForce và SlowHTTPTest có bản sao y hệt trong train, và mô hình chỉ cần “tra cứu” là đạt "
           "điểm tuyệt đối. Vector mang nhiều nhãn chủ yếu là cặp Benign | Infiltration – cùng một flow vừa được "
           "gán bình thường vừa được gán tấn công, giải thích một phần vì sao Infiltration khó (mục 4.4).")

    # ---------------------------------------------------------------- preprocessing
    conf = load("conflicting_labels_removed.json")
    split = load("split_config.json")
    w.h2("3.5. Tiền xử lý dữ liệu")
    w.h3("3.5.1. Lấy mẫu và ánh xạ nhãn")
    w.para("Toàn bộ dữ liệu duy nhất (12 triệu dòng) vượt quá khả năng huấn luyện nhiều mô hình trên một "
           "laptop, nên đề tài lấy một tập làm việc: tối đa 150.000 dòng duy nhất cho mỗi nhãn tấn công và "
           "900.000 dòng Benign. Việc chọn dòng dựa trên giá trị băm của dòng (không phụ thuộc thứ tự đọc), nên "
           "tái lập được và lấy đều trên cả ngày. 15 nhãn gốc được gộp thành 7 họ như ở bảng các họ tấn công "
           "(mục 1.3.2) cho bài toán đa lớp, và thành BENIGN/ATTACK cho bài toán nhị phân.")
    w.para(f"Tiếp theo, {n(conf['conflicting_vectors'])} vector đặc trưng xuất hiện với nhiều nhãn "
           f"({n(conf['rows_removed'])} dòng, trong đó {n(conf['label_combinations']['Benign | Infiltration'])} "
           f"cặp Benign | Infiltration) bị loại, còn lại **{n(conf['rows_kept'])} dòng**.")
    w.h3("3.5.2. Chia dữ liệu")
    dist = pd.read_csv(M / "split_distribution_family.csv")
    rows = [[r.Family, n(r.train), n(r.val), n(r.test), n(r.total)] for r in dist.itertuples()]
    s = split["sizes"]
    rows.append(["Tổng", n(s["train"]), n(s["val"]), n(s["test"]), n(sum(s.values()))])
    w.table("Kích thước các tập sau khi chia 70/15/15 phân tầng theo nhãn gốc (seed 42)",
            ["Họ", "Train", "Validation", "Test", "Tổng"], rows, [2.4, 2.2, 2.2, 2.2, 2.0], align="lrrrr",
            highlight_rows={len(rows)},
            note="Kiểm tra băm dòng: 0 dòng trùng giữa train–validation, train–test và validation–test.")
    w.para("Ngoài cách chia ngẫu nhiên phân tầng, đề tài tạo thêm một cách chia **theo thời gian trong từng "
           "nhãn**: với mỗi nhãn, 70% flow sớm nhất vào train, 15% kế tiếp vào validation, 15% muộn nhất vào "
           "test. Cách chia này kiểm tra mô hình có dự đoán được lưu lượng xảy ra *sau* dữ liệu huấn luyện hay "
           "không; chia toàn cục theo ngày là không khả thi vì mỗi ngày chỉ có một vài loại tấn công.")
    w.h3("3.5.3. Làm sạch")
    clean = load("cleaning_report.json")
    w.para("Lớp `FlowCleaner` được **fit chỉ trên train** rồi áp dụng cho validation, test và dữ liệu "
           "realtime: +Inf ở cột tốc độ thay bằng giá trị hữu hạn lớn nhất của train, NaN ở cột tốc độ (flow có "
           "thời lượng 0) thay bằng 0, thời gian âm đặt về 0, giá trị −1 của Init Win Byts (không có gói theo "
           f"chiều đó) được giữ nguyên vì mang thông tin, và 8 cột hằng số bị loại – còn {clean['n_kept']} đặc "
           "trưng ứng viên. Tham số của bộ làm sạch được lưu cùng mô hình để dùng lại khi triển khai.")

    w.h2("3.6. Chọn đặc trưng và cấu hình huấn luyện")
    w.h3("3.6.1. Chọn đặc trưng")
    w.para("Quy trình chọn đặc trưng chạy hoàn toàn trên train, chỉ dùng validation để chọn số đặc trưng k:")
    w.bullets([
        "Loại đặc trưng phương sai gần 0 (không có đặc trưng nào bị loại ở bước này).",
        "Xếp hạng theo bốn tiêu chí: mutual information, độ quan trọng Random Forest, LightGBM và XGBoost; "
        "lấy trung bình thứ hạng.",
        "Loại 20 đặc trưng tương quan |r| > 0,98 với một đặc trưng xếp hạng cao hơn (ví dụ Flow Duration ≈ "
        "Fwd IAT Tot, r = 0,996).",
        "Huấn luyện LightGBM với k = 5…50 đặc trưng đứng đầu; chọn k nhỏ nhất có Macro-F1 validation không "
        "kém quá 0,002 so với tốt nhất → **k = 30**.",
    ])
    w.figure(RFIG / "feature_k_sweep.png", "Macro-F1 validation theo số đặc trưng k", 11)
    w.figure(FIG / "fs_mutual_information.png", "Mutual information giữa đặc trưng và nhãn (train)", 9.5)
    w.h3("3.6.2. Mất cân bằng và chuẩn hoá")
    w.para("Năm chiến lược xử lý mất cân bằng được so sánh với cùng một LightGBM, chỉ resample tập train:")
    w.figure(RFIG / "imbalance_comparison.png", "So sánh các chiến lược xử lý mất cân bằng trên validation", 16)
    w.para("Không xử lý mất cân bằng cho Macro-F1 cao nhất (0,892) và FPR thấp nhất (2,1%). Class weight và "
           "undersampling làm tăng recall của Infiltration (0,40 → 0,78) nhưng phải trả giá bằng FPR tăng gấp "
           "4–9 lần: vì Infiltration chồng lấn với BENIGN, đẩy mô hình về phía Infiltration đồng nghĩa với báo "
           "động nhầm hàng loạt flow bình thường. SMOTE gần như không thay đổi gì vì tập làm việc đã được lấy mẫu "
           "cân bằng hơn dữ liệu gốc. Đề tài chọn **không resample**; với mô hình cây, chuẩn hoá không ảnh hưởng, "
           "còn MLP và Gaussian dùng biến đổi sign(x)·log(1+|x|) rồi StandardScaler fit trên train.")
    w.h3("3.6.3. Tinh chỉnh siêu tham số")
    w.para("Mỗi mô hình cây được tinh chỉnh bằng random search (6–10 cấu hình) trên một mẫu phân tầng 250.000 "
           "dòng của train, chấm điểm bằng Macro-F1 trên **validation**; cấu hình tốt nhất được huấn luyện lại "
           "trên toàn bộ train. Tập test không được dùng ở bất kỳ bước nào.")
    lg = load("lightgbm_divergence.json", EXP)["runs"]
    a = lg["default (min_sum_hessian_in_leaf=1e-3)"]
    b = lg["min_sum_hessian_in_leaf=1"]
    w.para(f"Trong quá trình này phát hiện LightGBM đa lớp **phân kỳ số học** với cấu hình mặc định: log-loss "
           f"trên train giảm tới {vn(a['train_logloss_min'])} ở vòng {a['argmin_iteration']} rồi tăng lên "
           f"{vn(a['train_logloss_last'], 2)}, Macro-F1 validation chỉ {vn(a['val_macro_f1'])}. Loss trên train "
           "tăng nên đây không phải quá khớp, mà là lá có tổng hessian rất nhỏ (xác suất gần 0 hoặc 1) cho giá "
           f"trị lá khổng lồ. Đặt `min_sum_hessian_in_leaf = 1` khắc phục hoàn toàn (Macro-F1 {vn(b['val_macro_f1'])}).")
    params = {k: json.loads((ROOT / "models" / p / "multiclass" / "params.json").read_text())["params"]
              for k, p in (("rf", "random_forest"), ("lgbm", "lightgbm"), ("xgb", "xgboost"), ("cat", "catboost"),
                           ("mlp", "mlp"))}
    fmt = lambda d, keys: ", ".join(f"{k}={d[k]}" for k in keys if k in d)  # noqa: E731
    w.table("Siêu tham số được chọn (bài toán 7 lớp)",
            ["Mô hình", "Siêu tham số"],
            [["Random Forest", fmt(params["rf"], ["n_estimators", "max_features", "min_samples_leaf", "min_samples_split", "max_depth"])],
             ["LightGBM", fmt(params["lgbm"], ["n_estimators", "num_leaves", "learning_rate", "min_sum_hessian_in_leaf", "bagging_fraction", "lambda_l1"])],
             ["XGBoost", fmt(params["xgb"], ["n_estimators", "max_depth", "learning_rate", "colsample_bytree", "min_child_weight", "reg_lambda"])],
             ["CatBoost", fmt(params["cat"], ["iterations", "depth", "learning_rate", "l2_leaf_reg"])],
             ["MLP", "256–128–64, BatchNorm, ReLU, dropout=0.2, Adam lr=0.001, batch=512, tối đa 40 epoch, "
                     "early stopping patience=6, ReduceLROnPlateau (×0,5 sau 2 epoch)"]],
            [2.4, 8.6], size=20)

    # ---------------------------------------------------------------- anomaly
    an = load("anomaly_results.json")
    ks = an["_gmm_component_selection"]["val_pr_auc_by_k"]
    w.h2("3.7. Phát hiện bất thường")
    w.para("Các mô hình bất thường được fit trên 300.000 flow BENIGN của train (One-Class SVM: 15.000 do độ "
           "phức tạp O(n²)), trên 30 đặc trưng đã chọn sau biến đổi log và chuẩn hoá. Số thành phần K của GMM "
           "được chọn theo PR-AUC trên validation:")
    w.table("Chọn số thành phần K của Gaussian mixture",
            ["K", "2", "4", "8", "16"],
            [["PR-AUC validation"] + [vn(ks[k]) for k in ("2", "4", "8", "16")]],
            [3.0, 2, 2, 2, 2], align="lcccc")
    w.para("Mỗi detector có ba điểm vận hành: ngưỡng cực đại F1 và ngưỡng đạt FPR 1% hoặc 5% trên flow BENIGN "
           "của validation. Mô hình triển khai dùng GMM ở FPR 1%, vì trong vận hành thật cảnh báo giả mới là chi "
           "phí chính. Thí nghiệm leave-one-family-out (LOFO) kiểm tra tấn công chưa từng thấy: lần lượt giấu "
           "từng họ khỏi train của bộ phân loại, rồi so recall trên họ đó giữa GMM và bộ phân loại ở **cùng** "
           "FPR (ngưỡng của bộ phân loại đặt trên xác suất BENIGN).")

    # ---------------------------------------------------------------- pcap / parity
    par = load("feature_parity_check.json")
    w.h2("3.8. Dựng flow từ gói tin và kiểm chứng feature parity")
    w.para("Bộ dựng flow tái hiện logic của CICFlowMeter: flow hai chiều theo 5-tuple, gói đầu xác định chiều "
           "đi, kết thúc khi FIN/RST hoặc 120 giây, độ dài gói tính theo payload L4, thời gian theo µs, độ lệch "
           "chuẩn mẫu, Active/Idle với ngưỡng 5 giây. Payload được tính từ trường *IP total length* thay vì độ "
           "dài bắt được, để xử lý đúng PCAP bị cắt như CTU-13. Danh sách flow đang mở được giữ theo thứ tự hoạt "
           "động gần nhất nên việc hết hạn flow có độ phức tạp O(1); khi vượt 200.000 flow, flow cũ nhất bị đóng "
           "để giới hạn RAM.")
    w.para(f"Dùng chung mã chưa đủ: đặc trưng phải **bằng** giá trị CICFlowMeter đã tính. Trên PCAP nạn nhân Bot, "
           f"dự án dựng được {n(par['our_flows'])} flow, trong đó {n(par['victim_cnc_flows'])} flow C&C. Vì CSV "
           "của ngày này không có cột IP và timestamp không đủ chính xác, các flow được so khớp theo “chữ ký "
           "kích thước” (số gói, tổng byte, độ dài header, cửa sổ TCP đầu) rồi so từng đặc trưng:")
    w.bullets([
        "Cùng cách tách kết nối như CICFlowMeter: mỗi kết nối C&C thành một flow chính (3/4 gói, 326/129 byte) "
        "và một flow “đuôi” sau FIN (2/0 gói) – đúng hiện tượng TCP appendix.",
        "Không khớp: mọi bộ đếm cờ TCP, Init Bwd Win Byts (CICFlowMeter ghi 219 trong khi gói SYN-ACK có cửa sổ "
        "26.883), Flow/Fwd/Bwd IAT Min, Pkt Len Mean/Var. Trong CSV, SYN Flag Cnt trùng hoàn toàn với Fwd PSH "
        "Flags (r = 1), nên các cột cờ của bản CSV không thể tái tạo từ gói tin.",
        f"Kết quả: **{len(par['parity_safe_features'])}/78 đặc trưng khớp** (parity-safe). CICFlowMeter không bao "
        "giờ xuất flow 1 gói và không có idle timeout, nên dự án đặt idle timeout = 120 s và không chấm điểm "
        "flow < 2 gói ở mức flow (vẫn dùng ở mức host để phát hiện quét).",
    ])
    dep = load("deployment_model_metrics.json")
    w.para(f"Từ đó, **mô hình triển khai** dùng {len(dep['features'])} đặc trưng = 30 đặc trưng đã chọn ∩ "
           f"{len(par['parity_safe_features'])} đặc trưng khớp, loại "
           f"{', '.join(dep['dropped_non_parity_features'])}.")

    # ---------------------------------------------------------------- botnet
    bs = load("botnet_summary.json")
    lv = load("ctu13_label_verification.json")
    inf, nor = lv["147.32.84.165"], lv["147.32.84.170"]
    w.h2("3.9. Phân tích botnet trên CTU-13")
    w.h3("3.9.1. Dựng flow và kiểm chứng nhãn")
    w.para(f"Toàn bộ PCAP được xử lý một lượt: 71.829.042/71.829.068 gói giải mã được (26 khung hỏng; số gói "
           f"khớp capinfos), tạo {n(bs['flows_total'])} flow hai chiều. Vì không có nhãn mức flow chính thức, "
           "nhãn host được kiểm chứng bằng hành vi:")
    w.table("Kiểm chứng nhãn host bằng hành vi quan sát được",
            ["Chỉ số", "147.32.84.165 (nhiễm)", "147.32.84.170 (bình thường)"],
            [["Flow khởi tạo", n(inf["flows_initiated"]), n(nor["flows_initiated"])],
             ["Số IP đích khác nhau", n(inf["unique_dst_ips"]), n(nor["unique_dst_ips"])],
             ["Tỷ lệ kết nối thất bại", pct(inf["failed_share"], 0), pct(nor["failed_share"], 0)],
             ["Cổng đích nhiều nhất", "25 (SMTP): 8.367; 53: 7.126; 80: 2.140; 6667 (IRC): 1.349",
              "53: 12.239; 80: 11.189; 443: 2.761"]],
            [3.0, 4.6, 3.4], align="lcc", size=21)
    w.para("Máy 147.32.84.165 là máy gửi SMTP và IRC nhiều nhất toàn capture, liên lạc với 4.190 đích – đúng "
           "hành vi spam và C&C qua IRC của Neris. Nhãn host vì vậy đáng tin; giới hạn là chỉ có **một** máy nhiễm.")
    w.h3("3.9.2. Đặc trưng host, đồ thị và beaconing")
    w.para("Flow được gom theo IP nguồn trong cửa sổ 300 giây, tạo 18 đặc trưng: tổng kết nối, số IP/cổng đích "
           "khác nhau, entropy IP và cổng đích, số kết nối thất bại, SYN không có ACK, byte/gói, độ lệch khoảng "
           "thời gian giữa kết nối, tỷ lệ kết nối lặp… Đồ thị liên lạc toàn capture có "
           f"{n(bs['graph']['nodes'])} đỉnh và {n(bs['graph']['edges'])} cạnh; chỉ số đồ thị (bậc, bậc có trọng "
           "số, độ trung gian xấp xỉ bằng lấy mẫu, hệ số phân cụm) được ghép vào đặc trưng host.")
    w.para(f"Dữ liệu được chia theo **cửa sổ thời gian**, không trộn ngẫu nhiên, để các cửa sổ liền kề của cùng "
           f"một máy không rơi vào cả train và test: train {bs['split']['train']['windows']} cửa sổ, validation "
           f"{bs['split']['val']['windows']}, test {bs['split']['test']['windows']} "
           f"({bs['split']['test']['rows']} dòng host-cửa sổ, {bs['split']['test']['botnet_rows']} dòng của bot). "
           "So sánh năm bộ phát hiện: Gaussian đa biến và Isolation Forest (không giám sát), Random Forest với "
           "đặc trưng host và với host + đồ thị, và bộ luật đồ thị + beaconing.")
    w.para("Beaconing được tính cho mọi cặp (nguồn, đích, cổng) có ít nhất 8 kết nối: các kết nối cách nhau "
           "không quá 1 giây được gộp thành một đợt, rồi tính CV và CV bền vững của khoảng cách giữa các đợt; "
           "DNS (53) và NTP (123) được loại vì có chu kỳ tự nhiên. Một cặp là beacon khi min(CV, CV bền vững) "
           "≤ 0,25, chu kỳ trung vị ≥ 2 giây và chuỗi kết nối kéo dài ≥ 60 giây.")
    w.h3("3.9.3. Phân tích topology: tập trung hay ngang hàng")
    w.para("Bộ phân tích topology nhận danh sách máy nghi vấn và đồ thị, rồi bỏ phiếu bằng nhiều chỉ số: số máy "
           "điều khiển chung (đích hiếm được ≥ 50% máy nghi vấn kết nối), số peer hiếm dùng chung, liên kết trực "
           "tiếp giữa các máy nghi vấn, độ tương đồng Jaccard giữa tập đích hiếm và độ tập trung lưu lượng. "
           "Phiên bản đầu chỉ đúng 34% vì coi máy chủ DNS và web phổ biến là C&C; phiên bản hai so với **mức phổ "
           "biến trong toàn mạng** (loại đích được nhiều máy bình thường cùng dùng) trước khi bỏ phiếu. Kết luận "
           "có thể là centralized, p2p, không có cấu trúc phối hợp, hoặc không đủ bằng chứng – hệ thống không bao "
           "giờ đoán khi chỉ có một máy nghi vấn.")

    # ---------------------------------------------------------------- pipeline
    w.h2("3.10. Pipeline phát hiện nhiều tầng và thời gian thực")
    w.para("Mỗi flow đủ điều kiện (≥ 2 gói) được làm sạch bằng FlowCleaner đã lưu, rồi đi qua hai tầng mức flow; "
           "các flow nghi vấn và toàn bộ flow trong cửa sổ được chuyển cho tầng mức host:")
    w.table("Logic quyết định của pipeline",
            ["Điều kiện", "Kết quả", "Mức độ"],
            [["Classifier dự đoán tấn công, xác suất ≥ 0,6", "Cảnh báo đúng loại tấn công",
              "Botnet, Infiltration: CRITICAL; DoS, DDoS, BruteForce, WebAttack: HIGH; hạ 1 mức nếu xác suất < 0,8"],
             ["Classifier dự đoán tấn công, xác suất < 0,6", "Uncertain, chuyển host xuống tầng host", "LOW"],
             ["Classifier dự đoán BENIGN nhưng GMM báo bất thường", "Unknown (có thể là tấn công chưa biết)",
              "MEDIUM (LOW nếu xác suất BENIGN ≥ 0,9)"],
             ["BENIGN và không bất thường", "Không cảnh báo", "–"],
             ["Host quét ≥ 50 cổng hoặc ≥ 50 IP, ≥ 50% thất bại", "PortScan / HostSweep (HostScanRule)", "HIGH"],
             ["Cặp kết nối có beaconing", "Beaconing – tín hiệu bổ trợ", "MEDIUM"],
             ["Mô hình host đánh giá máy nhiễm", "Botnet host", "HIGH nếu điểm ≥ 0,8, ngược lại MEDIUM"],
             ["Nhiều máy nghi vấn có cấu trúc phối hợp", "Kết luận topology centralized / P2P", "HIGH"]],
            [4.2, 3.8, 3.0], size=21)
    w.para("Chương trình realtime dùng Scapy AsyncSniffer bắt gói trên card mạng (hoặc replay PCAP theo đúng "
           "nhịp thời gian), chuyển byte thô qua hàng đợi có giới hạn 100.000 gói tới bộ xử lý; khi hàng đợi đầy, "
           "gói bị bỏ được **đếm và báo cáo**, không bỏ qua âm thầm. Mỗi 5 giây các flow đã đóng được chấm điểm; "
           "phân tích host chạy trên cửa sổ trượt 300 giây. Cảnh báo cùng (nguồn, đích, loại, module) trong 30 "
           "giây được gộp. Đầu ra gồm console, alerts.jsonl, alerts.csv, SQLite và run_summary.json (số gói, gói "
           "lỗi, gói bị bỏ, độ trễ p50/p95, CPU, RAM). Hướng dẫn triển khai trên Kali nằm ở docs/DEPLOYMENT.md.")


# =====================================================================================================
def part4(w: Writer) -> None:
    w.h1("PHẦN 4: THỰC NGHIỆM VÀ ĐÁNH GIÁ")
    w.h2("4.1. Ma trận thí nghiệm")
    w.table("Ma trận thí nghiệm của đề tài",
            ["Mã", "Thí nghiệm", "Dữ liệu / cấu hình", "Trạng thái"],
            [["E1", "Kiểm tra chất lượng dữ liệu", "16,2 triệu flow, đọc dạng luồng", "Hoàn thành"],
             ["E2", "Chọn đặc trưng, quét k", "Train; chọn k trên validation", "Hoàn thành"],
             ["E3", "So sánh xử lý mất cân bằng", "LightGBM, 5 chiến lược", "Hoàn thành"],
             ["E4", "7 mô hình đa lớp và nhị phân", "Tinh chỉnh trên validation", "Hoàn thành (validation)"],
             ["E5", "Gaussian UV/MV/GMM, IF, OCSVM", "Fit trên BENIGN; ε trên validation", "Hoàn thành (test)"],
             ["E6", "Tấn công chưa từng thấy (LOFO)", "Giấu từng họ, cùng FPR", "Hoàn thành"],
             ["E7", "Feature parity trên PCAP gốc", "PCAP CICIDS2018 02/03/2018", "Hoàn thành"],
             ["E8", "Mô hình triển khai 24 đặc trưng", "LightGBM + GMM (FPR 1%)", "Hoàn thành (test)"],
             ["E9", "Botnet CTU-13: host, đồ thị, beaconing", "Chia theo cửa sổ thời gian", "Hoàn thành"],
             ["E10", "Topology centralized/P2P", "Mô phỏng trên đồ thị CTU-13 thật", "Hoàn thành"],
             ["E11", "Test set, nhiều seed, chia theo thời gian", "evaluate.py", "Đang thực hiện"],
             ["E12", "Kịch bản lab realtime (quét, DoS, beacon)", "Mạng lab có phép", "Chưa thực hiện"]],
            [0.9, 4.0, 3.9, 2.2], size=21)
    w.h2("4.2. Giao thức đánh giá")
    w.para("Tập test được giữ kín trong toàn bộ quá trình chọn đặc trưng, chọn chiến lược mất cân bằng, chọn "
           "siêu tham số và chọn ngưỡng. Kết quả so sánh mô hình có giám sát trong phần này là trên **tập "
           "validation (266.305 flow)** – đây cũng là tập dùng để tinh chỉnh, nên các con số có thể lạc quan nhẹ. "
           "Bước đánh giá cuối trên tập test (kèm lặp nhiều seed và cách chia theo thời gian) đang chạy và sẽ "
           "được cập nhật. Riêng các detector bất thường và mô hình triển khai đã được đánh giá một lần trên tập "
           "test sau khi mọi lựa chọn đã cố định; các bảng đó ghi rõ “test”.")

    w.h2("4.3. Kết quả phân tích khám phá dữ liệu")
    w.figure(FIG / "eda_class_distribution_family.png", "Số flow theo họ trong toàn bộ CSE-CIC-IDS2018 (thang log)", 14)
    w.bullets([
        "Mất cân bằng cực lớn: BENIGN 13,5 triệu flow, WebAttack chỉ 928 – chênh hơn bốn bậc độ lớn; Accuracy "
        "vì vậy vô nghĩa và Macro-F1 là chỉ số chính.",
        "Các đặc trưng tốc độ và thời gian lệch phải rất mạnh (đuôi dài tới 10⁸ µs), cần biến đổi log trước "
        "khi dùng cho mô hình tuyến tính, MLP và Gaussian.",
        "Nhiều nhóm đặc trưng gần như trùng nhau (Subflow ≡ Tot; Pkt Size Avg ≈ Pkt Len Mean), là lý do cần lọc "
        "tương quan.",
    ])
    w.figure(FIG / "eda_correlation_heatmap.png", "Ma trận tương quan giữa các đặc trưng (train)", 11)

    # ---------------------------------------------------------------- supervised
    w.h2("4.4. Kết quả các mô hình có giám sát")
    rows = []
    s = load("training_summary_multiclass.json", EXP)
    order = ["rf", "lgbm", "xgb", "cat", "dt", "mlp", "lr"]
    for k in order:
        v = s[k]["val_metrics"]
        rows.append([NAMES[k], vn(v["accuracy"]), vn(v["f1_macro"]), vn(v["f1_weighted"]), vn(v["recall_macro"]),
                     pct(v["fpr"]), pct(v["fnr"]), vn(v["pr_auc_ovr_macro"])])
    w.table("Kết quả bài toán 7 lớp trên tập validation (266.305 flow)",
            ["Mô hình", "Accuracy", "Macro-F1", "Weighted-F1", "Macro recall", "FPR", "FNR", "PR-AUC"],
            rows, [2.6, 1.3, 1.3, 1.4, 1.4, 1.1, 1.1, 1.2], align="lrrrrrrr", size=20, highlight_rows={1},
            note="FPR/FNR theo góc nhìn nhị phân (mọi lớp tấn công gộp thành ATTACK); PR-AUC one-vs-rest, macro.")
    sb = load("training_summary_binary.json", EXP)
    rows = []
    for k in order:
        if k not in sb:
            rows.append([NAMES[k], "đang huấn luyện", "", "", "", ""])
            continue
        v = sb[k]["val_metrics"]
        rows.append([NAMES[k], vn(v["accuracy"]), vn(v["f1_macro"]), vn(v["precision_macro"]), pct(v["fpr"]),
                     pct(v["fnr"])])
    w.table("Kết quả bài toán nhị phân BENIGN/ATTACK trên tập validation",
            ["Mô hình", "Accuracy", "Macro-F1", "Macro precision", "FPR", "FNR"],
            rows, [2.8, 1.6, 1.6, 1.8, 1.5, 1.5], align="lrrrrr", size=20)
    w.figure(RFIG / "val_model_comparison.png", "Macro-F1 và FPR của các mô hình trên tập validation", 16)
    w.h3("4.4.1. Nhận xét")
    w.bullets([
        "**Bốn mô hình cây tốt nhất và gần như ngang nhau.** Random Forest dẫn Macro-F1 (0,917) nhờ lớp "
        "WebAttack; LightGBM và XGBoost kém 0,008–0,013; CatBoost có FPR thấp nhất (1,3%). Ở bài toán nhị phân, "
        "chênh lệch giữa bốn mô hình dưới 0,002 – nhỏ hơn mức dao động thường gặp giữa các seed, nên không thể "
        "kết luận mô hình nào “thắng” chỉ bằng Macro-F1.",
        "**Accuracy gần như bằng nhau giữa bài toán 7 lớp và nhị phân** (LightGBM 0,9456 và 0,9451): gần như "
        "toàn bộ lỗi là nhầm giữa BENIGN và tấn công, hầu như không có nhầm giữa các họ tấn công với nhau.",
        "**Decision Tree và Logistic Regression** có recall cao nhưng FPR rất lớn (20,5% và 38,0%) – báo động "
        "nhầm 1/5 đến 2/5 lưu lượng bình thường, không dùng được trong vận hành.",
        "**MLP** (0,865) kém các mô hình cây – kết quả thường gặp trên dữ liệu dạng bảng với các đặc trưng có "
        "ngưỡng rõ ràng. MLP dừng ở giới hạn 40 epoch (epoch tốt nhất 38) khi loss validation đã gần như đi "
        "ngang (0,165), nên huấn luyện thêm khó cải thiện đáng kể. Bù lại, MLP có FPR thấp (1,1%) và mô hình "
        "nhỏ nhất (0,2 MB).",
        "**Quá khớp:** Macro-F1 trên train và validation của LightGBM là 0,921 và 0,909, của Random Forest là "
        "0,954 và 0,917 – Random Forest học thuộc train nhiều hơn nhưng vẫn tổng quát tốt.",
    ])
    w.figure(RFIG / "val_per_class_f1.png", "F1 theo từng lớp của các mô hình trên tập validation", 15.5)

    w.h2("4.5. Phân tích lỗi: Infiltration và WebAttack")
    w.figure(RFIG / "cm_val_lgbm.png", "Ma trận nhầm lẫn của LightGBM trên tập validation (chuẩn hoá theo hàng)", 13)
    cm = __import__("numpy").load(RFIG / "cm_val_lgbm.npy")
    w.para(f"Botnet, BruteForce, DDoS và DoS gần như hoàn hảo (F1 ≥ 0,999). Toàn bộ sai số tập trung ở hai lớp. "
           f"**Infiltration:** {n(cm[5, 0])}/{n(cm[5].sum())} flow ({pct(cm[5, 0] / cm[5].sum(), 0)}) bị dự đoán "
           f"là BENIGN, và {n(cm[0, 5])} flow BENIGN bị dự đoán là Infiltration – chiếm gần hết FPR 2%. Nguyên "
           "nhân không nằm ở mô hình mà ở bản chất nhãn: CIC gán nhãn Infiltration cho *mọi* flow của máy bị "
           "chiếm trong khoảng thời gian tấn công, kể cả lướt web và DNS bình thường. Bằng chứng: 1.894 vector "
           "đặc trưng xuất hiện giống hệt với cả hai nhãn Benign và Infiltration; mọi mô hình và mọi chiến lược "
           "cân bằng đều dừng ở F1 ≈ 0,35–0,56; và đẩy recall lên bằng class weight làm FPR tăng gấp 9 lần.")
    w.para(f"**WebAttack:** {n(cm[6, 0])}/{n(cm[6].sum())} flow bị coi là BENIGN. Chỉ có 128 mẫu validation nên "
           "mỗi flow thay đổi F1 gần 1 điểm phần trăm; tấn công XSS/SQLi nằm trong nội dung HTTP mà đặc trưng "
           "flow không chứa. Hai lớp này là giới hạn của **dữ liệu mức flow**, không phải giới hạn có thể khắc "
           "phục bằng mô hình phức tạp hơn.")

    w.h2("4.6. Hiệu năng tính toán")
    rows = []
    for k in order:
        r = s[k]
        rows.append([NAMES[k], vn(r["train_seconds"] / 60, 1), vn(r["val_inference_us_per_flow"], 1),
                     vn(r["model_size_mb"], 1)])
    w.table("Chi phí huấn luyện và suy luận (bài toán 7 lớp, 1,24 triệu flow train, CPU)",
            ["Mô hình", "Thời gian train (phút)", "Suy luận (µs/flow)", "Kích thước (MB)"],
            rows, [3.0, 2.6, 2.6, 2.4], align="lrrr", size=21,
            note="Suy luận đo theo lô trên toàn bộ tập validation; Random Forest không giới hạn độ sâu nên mô hình "
                 "lên tới 2,1 GB.")
    w.para("Random Forest chính xác nhất nhưng nặng 2,1 GB và huấn luyện lâu nhất trong nhóm cây, không phù hợp "
           "chạy trên máy giám sát. LightGBM huấn luyện nhanh nhất (5 phút), mô hình 17,5 MB, độ chính xác chỉ "
           "kém RF 0,008 Macro-F1 và có FPR thấp hơn, nên được chọn làm bộ phân loại triển khai.")

    dep = load("deployment_model_metrics.json")
    w.h2("4.7. Mô hình triển khai (24 đặc trưng parity-safe)")
    dv, dt_ = dep["classifier_val"], dep["classifier_test"]
    g = dep["gaussian_test_at_1pct_val_fpr"]
    w.table("Mô hình triển khai: LightGBM 24 đặc trưng và GMM tại FPR 1%",
            ["Chỉ số", "Validation", "Test"],
            [["LightGBM – Macro-F1", vn(dv["f1_macro"]), vn(dt_["f1_macro"])],
             ["LightGBM – FPR / FNR", f"{pct(dv['fpr'], 2)} / {pct(dv['fnr'], 2)}", f"{pct(dt_['fpr'], 2)} / {pct(dt_['fnr'], 2)}"],
             ["LightGBM – PR-AUC (OvR macro)", vn(dv["pr_auc_ovr_macro"]), vn(dt_["pr_auc_ovr_macro"])],
             ["GMM – recall / FPR (ngưỡng chọn trên validation)", "–", f"{vn(g['recall'])} / {pct(g['fpr'], 2)}"],
             ["GMM – ROC-AUC / PR-AUC", "–", f"{vn(g['roc_auc'])} / {vn(g['pr_auc'])}"]],
            [5.0, 3.0, 3.0], align="lcc", size=21)
    w.para(f"Bỏ 6 đặc trưng không khớp CICFlowMeter chỉ làm Macro-F1 giảm từ 0,909 xuống {vn(dv['f1_macro'])} trên "
           "validation. Validation và test gần như trùng nhau (chênh dưới 0,002), cho thấy việc tinh chỉnh trên "
           "validation không làm kết quả lạc quan đáng kể.")

    # ---------------------------------------------------------------- anomaly
    an = load("anomaly_results.json")
    w.h2("4.8. Kết quả phát hiện bất thường")
    rows = []
    for name in ("Gaussian (univariate)", "Gaussian (multivariate)", "Gaussian mixture", "Isolation Forest",
                 "One-Class SVM"):
        v = an[name]
        o1, o5 = v["fpr_1pct"], v["fpr_5pct"]
        label = {"Gaussian (univariate)": "Gaussian đơn biến", "Gaussian (multivariate)": "Gaussian đa biến",
                 "Gaussian mixture": "Gaussian mixture (K=16)"}.get(name, name)
        rows.append([label, vn(o1["roc_auc"]), vn(o1["pr_auc"]), vn(o1["recall"]), vn(o1["precision"]),
                     vn(o5["recall"]), vn(o5["precision"])])
    w.table("Phát hiện bất thường trên tập test (ngưỡng chọn trên validation)",
            ["Detector", "ROC-AUC", "PR-AUC", "Recall\n@FPR 1%", "Precision\n@FPR 1%", "Recall\n@FPR 5%",
             "Precision\n@FPR 5%"],
            rows, [3.2, 1.3, 1.3, 1.4, 1.5, 1.4, 1.5], align="lrrrrrr", size=20, highlight_rows={3})
    w.figure(FIG / "anomaly_roc.png", "Đường ROC của các detector bất thường (test)", 11.5)
    w.bullets([
        "**Gaussian đơn biến thất bại hoàn toàn** (ROC-AUC 0,427 < 0,5): flood DoS/DDoS có hành vi rất đồng "
        "nhất nên nằm ở vùng mật độ cao theo từng đặc trưng riêng lẻ, trong khi lưu lượng bình thường đa dạng "
        "lại bị coi là hiếm. Giả định độc lập giữa các đặc trưng sai hoàn toàn với dữ liệu flow.",
        "**Gaussian đa biến** khá hơn (0,760) nhờ mô hình hoá tương quan, nhưng một ellipsoid duy nhất không "
        "bao được nhiều cụm hành vi bình thường.",
        "**Gaussian mixture** mô tả được các cụm và tốt nhất ở mọi chỉ số (ROC-AUC 0,884, PR-AUC 0,897); tại "
        "FPR 5% bắt được 77% tấn công với precision 94%. Tại FPR 1%, recall còn 17% – detector bất thường hữu "
        "ích như một tầng sàng lọc bổ sung, không thay thế được bộ phân loại.",
        "Isolation Forest (0,782) đứng giữa; One-Class SVM gần như ngẫu nhiên (0,500) vì chỉ fit được trên "
        "15.000 mẫu.",
    ])
    w.figure(FIG / "anomaly_score_distribution.png", "Phân phối điểm bất thường của GMM theo họ lưu lượng (test)", 15)
    w.para("Phân phối điểm giải thích kết quả theo họ: DoS (Hulk, Slowloris) nằm hẳn bên phải ngưỡng – rất "
           "khác bình thường; Botnet và BruteForce nằm ngay sát ngưỡng nên chỉ bị bắt khi nới FPR lên 5%; "
           "Infiltration trùng khít với BENIGN – xác nhận lại rằng ở mức flow, lớp này *là* lưu lượng bình thường.")
    w.h3("4.8.1. Tấn công chưa từng thấy (leave-one-family-out)")
    w.figure(RFIG / "lofo_unknown_attack.png", "Recall trên họ tấn công bị giấu khỏi dữ liệu huấn luyện", 14)
    w.para("Đây là thí nghiệm trả lời câu hỏi “tầng bất thường có đáng giữ không”. Khi một họ bị giấu khỏi "
           "train, bộ phân loại gần như không nhận ra nó: DDoS chỉ 0,001, DoS 0,355 – mô hình gán chúng vào "
           "BENIGN vì chưa từng thấy. GMM không phụ thuộc nhãn nên giữ nguyên khả năng phát hiện: DDoS 0,748, "
           "DoS 1,000, WebAttack 0,719. Botnet và BruteForce vẫn bị bộ phân loại bắt do giống các họ đã biết. "
           "Infiltration thì cả hai đều thất bại. Kết hợp hai tầng (cảnh báo khi một trong hai báo) đạt recall "
           "≥ 0,74 trên năm/sáu họ chưa biết, đổi lại FPR tăng từ 5% lên khoảng 9,5%.")

    # ---------------------------------------------------------------- botnet
    bs = load("botnet_summary.json")
    w.h2("4.9. Kết quả phân tích botnet CTU-13")
    w.figure(FIG / "botnet_graph_normal_vs_infected.png",
             "Lân cận trong đồ thị liên lạc: máy bình thường (trái) và máy nhiễm Neris (phải)", 15.5)
    w.para("Máy nhiễm có bậc ra 4.190 so với 302 của máy bình thường bận rộn nhất – gấp gần 14 lần – và phần lớn "
           "kết nối tới đích chỉ gặp một lần (thư rác, quét). Đây là tín hiệu rõ nhất và cũng là đặc trưng quan "
           "trọng nhất của mô hình host (out_degree, entropy IP đích, SYN không ACK, số IP đích).")
    det = pd.read_csv(M / "botnet_detector_comparison.csv", index_col=0)
    names = {"Gaussian (multivariate, host features)": "Gaussian đa biến (đặc trưng host)",
             "Isolation Forest (host features)": "Isolation Forest (đặc trưng host)",
             "Random Forest (host features)": "Random Forest (đặc trưng host)",
             "Random Forest (host + graph features)": "Random Forest (host + đồ thị)",
             "Graph + beaconing heuristic rules": "Luật đồ thị + beaconing"}
    rows = [[names[i], vn(r.precision), vn(r.recall), vn(r.f1), pct(r.fpr), "–" if pd.isna(r.roc_auc) else vn(r.roc_auc)]
            for i, r in det.iterrows()]
    w.table(f"Phát hiện máy nhiễm trên tập test CTU-13 ({bs['split']['test']['rows']} cửa sổ host, "
            f"{bs['split']['test']['botnet_rows']} của bot)",
            ["Bộ phát hiện", "Precision", "Recall", "F1", "FPR", "ROC-AUC"], rows,
            [4.2, 1.4, 1.3, 1.2, 1.3, 1.6], align="lrrrrr", size=21)
    ba = bs["background_alert_rate"]
    w.para("Random Forest đạt điểm tuyệt đối, nhưng con số này **không chứng minh khả năng tổng quát**: tập test "
           "chỉ có 11 cửa sổ của đúng một máy nhiễm, và máy này khác biệt cực lớn so với phần còn lại. Thông tin "
           "thực tế hơn là tỷ lệ cảnh báo trên lưu lượng Background (chưa biết nhãn): Random Forest cảnh báo "
           f"{pct(ba['Random Forest (host features)'])} trong {n(ba['n_background_host_windows'])} cửa sổ, luật "
           f"đồ thị + beaconing cảnh báo {pct(ba['Graph + beaconing heuristic rules'])}. Thêm đặc trưng đồ thị "
           "không cải thiện được thêm vì đặc trưng host đã tách hoàn toàn.")
    w.figure(FIG / "botnet_beaconing_timeline.png",
             "Beaconing: kết nối C&C của bot (trên) và keep-alive của ứng dụng hợp lệ (dưới)", 15.5)
    bp = bs["beacon_pairs"]
    w.para(f"Bot kết nối tới 78.129.163.119:6251 đúng mỗi 120 giây (CV = 0,04) – một beacon điển hình. Nhưng "
           f"ứng dụng bình thường cũng vậy: {bp['sum']['Normal']}/{bp['count']['Normal']} cặp của các máy bình "
           f"thường bị đánh dấu beaconing, so với chỉ {bp['sum']['Botnet']}/{bp['count']['Botnet']} cặp của bot. "
           "Ví dụ ở hình trên là keep-alive AOL/AIM tới cổng 443 với trung vị đúng 120 giây. Kết luận: tính chu "
           "kỳ là tín hiệu yếu, chỉ có giá trị khi kết hợp với bằng chứng khác.")
    ts = bs["topology_simulation"]
    w.table("Mô phỏng phân loại topology trên đồ thị CTU-13 thật (60 lần thử mỗi loại)",
            ["Thực tế", "Kết luận đúng", "Độ chính xác"],
            [["Centralized (một C&C chung)", f"{round(ts['accuracy_by_truth']['centralized'] * 60)}/60", vn(ts["accuracy_by_truth"]["centralized"], 2)],
             ["P2P (peer chung, liên kết giữa bot)", f"{round(ts['accuracy_by_truth']['p2p'] * 60)}/60", vn(ts["accuracy_by_truth"]["p2p"], 2)],
             ["Không có botnet (chỉ lưu lượng thật)", f"{round(ts['accuracy_by_truth']['none'] * 60)}/60", vn(ts["accuracy_by_truth"]["none"], 2)],
             ["Tổng", f"{int(round(ts['accuracy'] * 180))}/180", vn(ts["accuracy"], 2)]],
            [5.0, 3.0, 3.0], align="lcc", highlight_rows={4},
            note="Phiên bản không so sánh với mức phổ biến toàn mạng chỉ đạt 0,34. Khi các chỉ số không đồng thuận, hệ "
                 "thống trả “không đủ bằng chứng” thay vì đoán (28/180 lần; với trường hợp không có botnet, kết luận "
                 "này được tính là đúng).")
    w.para("Trên capture thật chỉ có một máy nhiễm, nên hệ thống trả kết luận “một host, không đủ bằng chứng” – "
           "đúng như thiết kế: không thể nói về cấu trúc phối hợp của botnet khi chỉ quan sát một bot.")

    # ---------------------------------------------------------------- parity
    par = load("feature_parity_check.json")
    w.h2("4.10. Kiểm chứng trên PCAP thật và thời gian thực")
    w.table("Kết quả end-to-end trên PCAP nạn nhân Bot của CICIDS2018",
            ["Chỉ số", "Trước khi sửa parity", "Sau khi sửa"],
            [["Flow C&C (TCP → 8080) phân loại Botnet", "99,99%", f"{pct(par['cnc_classified_botnet_share'], 2)} (29.734/29.737)"],
             ["Flow C&C bị GMM coi là bất thường", "100%", pct(par["cnc_flagged_anomalous_share"])],
             ["Flow bình thường bị GMM coi là bất thường", "99,3%", pct(par["other_flows_flagged_anomalous_share"])],
             ["Flow bình thường phân loại BENIGN", "96,8%", "99,0%"]],
            [5.0, 3.0, 3.0], align="lcc", size=21)
    w.para("Bộ phân loại cây chuyển giao tốt từ CSV sang flow tự dựng ngay cả trước khi sửa. Mô hình mật độ thì "
           "nhạy hơn nhiều: khi còn dùng đặc trưng cờ TCP và flow 1 gói, GMM báo động 99,3% lưu lượng bình "
           "thường. Sau khi chỉ dùng đặc trưng parity-safe, tỷ lệ này còn 16,6% – vẫn xa mục tiêu 1%. Vì vậy "
           "dự án cung cấp công cụ `calibrate_anomaly.py` dịch ngưỡng ε tới phân vị yêu cầu trên lưu lượng bình "
           "thường của chính mạng triển khai (mô hình giữ nguyên, lịch sử hiệu chỉnh được lưu).")
    w.para("Kiểm thử tích hợp xác nhận chế độ realtime (replay PCAP qua đúng đường đi của Scapy) cho số flow "
           "bằng chế độ offline, phát hiện được quét cổng 300 cổng bằng HostScanRule và beacon chu kỳ 30 giây. "
           "Các kịch bản lab trên mạng thật (quét cổng, mô phỏng DoS có giới hạn tốc độ, beacon) và phép đo độ "
           "trễ/thông lượng đã có script (scripts/lab) nhưng **chưa được chạy**, nên báo cáo không nêu số liệu "
           "cho phần này.")

    w.h2("4.11. Tổng hợp trade-off")
    w.table("Tổng hợp trade-off giữa các hướng tiếp cận",
            ["Tiêu chí", "LightGBM", "Random Forest", "Gaussian mixture", "Mức host + đồ thị"],
            [["Hiệu năng", "Macro-F1 0,909", "Macro-F1 0,917", "ROC-AUC 0,884", "F1 1,0 (n nhỏ)"],
             ["Cần nhãn tấn công", "Có", "Có", "Không", "Có (RF) / không (luật)"],
             ["Tấn công mới", "Kém (DDoS 0,001)", "Kém", "Tốt (DDoS 0,748)", "Theo hành vi"],
             ["Báo động giả", "2,0%", "2,6%", "Chỉnh được (1%/5%)", "6,3% cửa sổ nền"],
             ["Chi phí", "17,5 MB, train 5 phút", "2,1 GB, 17 phút", "K=16, 11 µs/flow", "Cần IP, cửa sổ 300 s"],
             ["Phụ thuộc mạng", "Thấp", "Thấp", "Cao – phải hiệu chỉnh ε", "Trung bình"]],
            [2.2, 2.2, 2.2, 2.4, 2.0], size=20, bold_first_col=True)

    w.h2("4.12. Trả lời các câu hỏi nghiên cứu")
    w.table("Trả lời các câu hỏi nghiên cứu",
            ["RQ", "Trả lời"],
            [["RQ1", "26% dòng trùng lặp, 22.359 vector nhãn mâu thuẫn, 7/10 file bị cắt, Inf/NaN, cột hằng, đồng "
                     "hồ 12 giờ. Khử trùng lặp, loại nhãn mâu thuẫn, fit mọi bước trên train → 0 dòng chồng lấn "
                     "giữa các tập."],
             ["RQ2", "Mô hình cây tốt nhất (RF 0,917, LightGBM 0,909 Macro-F1 validation), chênh lệch giữa chúng "
                     "nhỏ; LightGBM được chọn vì gọn và FPR thấp. Infiltration (F1 ≈ 0,56) và WebAttack là lớp "
                     "khó do nhãn chồng lấn với BENIGN và thông tin nằm trong payload."],
             ["RQ3", "Gaussian đơn/đa biến không đủ; Gaussian mixture đạt ROC-AUC 0,884 và phát hiện được tấn "
                     "công chưa từng thấy mà bộ phân loại bỏ sót (DDoS 0,748 so với 0,001)."],
             ["RQ4", "Đặc trưng host tách được máy nhiễm Neris (bậc ra gấp 14 lần); beaconing đơn lẻ là tín hiệu "
                     "yếu; phân loại topology đúng 82% trong mô phỏng, chỉ kết luận khi đủ bằng chứng."],
             ["RQ5", "58/78 đặc trưng khớp CICFlowMeter; cờ TCP thì không. Mô hình 24 đặc trưng giữ Macro-F1 "
                     "0,905 và phân loại đúng 99,99% flow C&C thật; GMM cần hiệu chỉnh ε theo mạng triển khai."]],
            [1.0, 10.0], size=21, bold_first_col=True)


# =====================================================================================================
def part5_6(w: Writer) -> None:
    w.h1("PHẦN 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN")
    w.h2("5.1. Kết luận")
    w.para("Đề tài đã xây dựng hoàn chỉnh một hệ thống phát hiện xâm nhập dựa trên học máy, từ kiểm tra dữ "
           "liệu thô đến chương trình giám sát thời gian thực, với các kết quả chính:")
    w.bullets([
        "Phát hiện và xử lý các vấn đề chất lượng nghiêm trọng của CSE-CIC-IDS2018 (trùng lặp, nhãn mâu "
        "thuẫn, file bị cắt, đồng hồ 12 giờ) – nếu bỏ qua, điểm số sẽ bị thổi phồng.",
        "Mô hình cây đạt Macro-F1 0,90–0,92 (7 lớp) và 0,944–0,945 (nhị phân) trên validation với FPR 1,3–2,6%; "
        "giới hạn còn lại nằm ở bản chất nhãn Infiltration và WebAttack chứ không ở mô hình.",
        "Gaussian anomaly detection được triển khai đúng nghĩa; mô hình hỗn hợp vượt trội Gaussian đơn/đa biến "
        "và bổ sung khả năng phát hiện tấn công chưa biết.",
        "Phân tích botnet trên 71,8 triệu gói CTU-13 xác định được máy nhiễm Neris bằng hành vi, và cho thấy "
        "beaconing đơn lẻ không đủ tin cậy.",
        "Đặc trưng tự dựng được kiểm chứng với CICFlowMeter trên PCAP gốc; mô hình triển khai chỉ dùng đặc "
        "trưng đã kiểm chứng và chạy được trên Kali Linux với cảnh báo nhiều mức độ.",
    ])
    w.h2("5.2. Hạn chế")
    w.bullets([
        "So sánh mô hình có giám sát hiện dựa trên tập validation – cũng là tập dùng để tinh chỉnh; kết quả trên "
        "test, độ lệch chuẩn qua nhiều seed và cách chia theo thời gian đang được chạy.",
        "Tập làm việc là mẫu 1,78 triệu dòng của dữ liệu gốc, giới hạn 150.000 dòng mỗi nhãn tấn công; phân phối "
        "lớp vì vậy cân bằng hơn thực tế.",
        "Các lớp FTP-BruteForce và SlowHTTPTest chỉ có 54–55 dòng duy nhất – điểm số của chúng không phản ánh "
        "khả năng tổng quát.",
        "CTU-13 chỉ dùng kịch bản 1 với một máy nhiễm; nhãn mức host do không lấy được nhãn flow chính thức. Kết "
        "quả F1 = 1,0 không chứng minh khả năng phát hiện botnet khác.",
        "Ngưỡng GMM nhạy với mạng triển khai (16,6% báo động trên lưu lượng bình thường của chính testbed khi "
        "dựng flow từ PCAP); bắt buộc hiệu chỉnh trước khi dùng.",
        "Chưa chạy kịch bản lab realtime và đo thông lượng; Python + Scapy giới hạn ở khoảng vài chục nghìn gói/giây.",
    ])
    w.h2("5.3. Khuyến nghị triển khai")
    w.bullets([
        "Dùng Macro-F1, recall từng lớp và FPR làm tiêu chí chấp nhận; không dùng Accuracy.",
        "Triển khai LightGBM 24 đặc trưng làm tầng phân loại, GMM làm tầng sàng lọc sau khi hiệu chỉnh ε bằng ít "
        "nhất vài giờ lưu lượng bình thường của mạng đích.",
        "Coi cảnh báo Infiltration, Unknown và Beaconing là tín hiệu cần chuyên viên xem xét, không phải kết "
        "luận tự động.",
        "Giám sát số gói bị bỏ trong run_summary.json; khi tăng, dùng BPF để lọc hoặc chuyển sang AF_PACKET.",
    ])
    w.h2("5.4. Hướng phát triển")
    w.bullets([
        "Hoàn tất đánh giá trên test, lặp nhiều seed, chia theo thời gian, và giải thích mô hình bằng SHAP.",
        "Huấn luyện lại trên dữ liệu đã sửa lỗi của CICIDS2018 [12] hoặc trên flow tự dựng từ PCAP gốc để loại "
        "hoàn toàn sai khác với CICFlowMeter, sau đó dùng lại toàn bộ 78 đặc trưng.",
        "Mở rộng phân tích botnet ra nhiều kịch bản CTU-13 (có nhiều máy nhiễm, có botnet P2P) để đánh giá thật "
        "bộ phân tích topology.",
        "Kiểm thử chéo bộ dữ liệu (huấn luyện CICIDS2018, kiểm thử CICIDS2017 hoặc lưu lượng lab).",
        "Tăng thông lượng bằng xử lý song song hoặc tích hợp với Zeek/Suricata làm nguồn flow.",
    ])

    w.h1("PHẦN 6: DANH MỤC TÀI LIỆU THAM KHẢO")
    refs = [
        "Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection "
        "dataset and intrusion traffic characterization. Proceedings of ICISSP 2018, 108–116.",
        "Lashkari, A. H., Draper-Gil, G., Mamun, M. S. I., & Ghorbani, A. A. (2017). Characterization of Tor "
        "traffic using time based features. Proceedings of ICISSP 2017 (nguồn gốc công cụ CICFlowMeter).",
        "Canadian Institute for Cybersecurity. CSE-CIC-IDS2018 on AWS. University of New Brunswick. "
        "https://www.unb.ca/cic/datasets/ids-2018.html",
        "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.",
        "Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). Classification and Regression Trees. Wadsworth.",
        "Pedregosa, F. và cs. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning "
        "Research, 12, 2825–2830.",
        "Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. ACM Computing Surveys, 41(3), 1–58.",
        "Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest. IEEE International Conference on Data "
        "Mining (ICDM), 413–422.",
        "He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE Transactions on Knowledge and Data "
        "Engineering, 21(9), 1263–1284.",
        "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when "
        "evaluating binary classifiers on imbalanced datasets. PLoS ONE, 10(3), e0118432.",
        "Engelen, G., Rimmer, V., & Joosen, W. (2021). Troubleshooting an intrusion detection dataset: the "
        "CICIDS2017 case study. IEEE Security and Privacy Workshops (SPW), 7–12.",
        "Liu, L., Engelen, G., Lynar, T., Essam, D., & Joosen, W. (2022). Error prevalence in NIDS datasets: A case "
        "study on CIC-IDS-2017 and CSE-CIC-IDS-2018. IEEE Conference on Communications and Network Security (CNS).",
        "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of ACM KDD, 785–794.",
        "Ke, G. và cs. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural "
        "Information Processing Systems 30 (NIPS 2017).",
        "Prokhorenkova, L. và cs. (2018). CatBoost: unbiased boosting with categorical features. Advances in "
        "Neural Information Processing Systems 31 (NeurIPS 2018).",
        "Schölkopf, B. và cs. (2001). Estimating the support of a high-dimensional distribution. Neural "
        "Computation, 13(7), 1443–1471.",
        "García, S., Grill, M., Stiborek, J., & Zunino, A. (2014). An empirical comparison of botnet detection "
        "methods. Computers & Security, 45, 100–123.",
        "Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority "
        "over-sampling technique. Journal of Artificial Intelligence Research, 16, 321–357.",
        "Paszke, A. và cs. (2019). PyTorch: An imperative style, high-performance deep learning library. "
        "Advances in Neural Information Processing Systems 32.",
        "Hagberg, A., Schult, D., & Swart, P. (2008). Exploring network structure, dynamics, and function using "
        "NetworkX. Proceedings of the 7th Python in Science Conference (SciPy), 11–15.",
    ]
    for i, r in enumerate(refs, 1):
        w.para(f"[{i}] {r}", indent=False)


def appendix(w: Writer) -> None:
    w.h1("PHỤ LỤC")
    w.h2("A. Cấu trúc mã nguồn và hướng dẫn tái lập")
    w.subhead("A.1. Cấu trúc thư mục")
    w.code("""network-ids/
├── config/config.yaml          # mọi tham số: seed, đường dẫn, timeout, ngưỡng
├── src/
│   ├── data_processing/        # pcap_parser, flow_builder, feature_extractor,
│   │                           # dataset_loader, preprocessing, host_aggregator
│   ├── models/                 # random_forest, xgboost_model, lightgbm_model,
│   │                           # catboost_model, baselines, mlp, gaussian_detector
│   ├── graph/                  # graph_builder, beaconing_detector, topology_analyzer
│   ├── detection/              # decision_engine, alert_manager, detection_pipeline
│   ├── realtime/               # packet_capture, realtime_processor, monitor
│   └── utils/                  # config, logger, metrics, visualization
├── scripts/                    # các bước pipeline (bảng A.2)
├── notebooks/01_data_exploration.ipynb
├── tests/                      # 38 kiểm thử pytest
├── models/                     # mô hình đã huấn luyện, bundle triển khai
├── reports/                    # metrics/*.json|csv, figures/*.png
└── docs/                       # tài liệu dữ liệu, tiền xử lý, mô hình, triển khai""")
    w.subhead("A.2. Lệnh chạy")
    w.code("""python scripts/download_dataset.py
python scripts/preprocess_dataset.py --stage all  # làm sạch, chia, chọn đặc trưng
python scripts/train.py --task multiclass --models lr dt lgbm xgb rf cat mlp
python scripts/train.py --task binary --models lr dt lgbm xgb rf cat mlp
python scripts/train_anomaly.py                   # Gaussian UV/MV/GMM, IF, OCSVM, LOFO
python scripts/botnet_analysis.py                 # CTU-13: host, đồ thị, beaconing, topology
python scripts/parity_check.py                    # so khớp với CICFlowMeter
python scripts/train_deployment.py                # mô hình 24 đặc trưng
python scripts/evaluate.py --stage all            # test, nhiều seed, chia theo thời gian
python scripts/run_realtime.py --interface eth0   # giám sát realtime (Kali, quyền root)
python -m pytest                                  # 38 kiểm thử""")
    w.subhead("A.3. Thông tin tái lập")
    w.table("Thông tin tái lập kết quả",
            ["Mục", "Giá trị"],
            [["Seed", "42 (lặp lại với 42, 7, 2024 ở bước đánh giá)"],
             ["Tập làm việc", "1.775.365 dòng; train/val/test = 1.242.755 / 266.305 / 266.305"],
             ["Đặc trưng", "30 đặc trưng chọn (models/preprocessing/selected_features.json); 24 cho triển khai"],
             ["Cấu hình", "config/config.yaml; siêu tham số trong models/<model>/<task>/params.json"],
             ["Kết quả", "reports/metrics/, reports/experiments/, reports/figures/"],
             ["Hình trong báo cáo", "scripts/make_report_figures.py; báo cáo dựng bằng scripts/report/build_report.py"]],
            [2.6, 8.4], size=21)

    w.h2("B. Mã nguồn các thành phần chính")
    w.subhead("B.1. Chọn ngưỡng ε cho Gaussian anomaly detection")
    w.code("""def threshold_at_fpr(normal_log_density, target_fpr):
    # flow bình thường có log p(x) thấp hơn phân vị target_fpr bị coi là bất thường
    return float(np.quantile(normal_log_density, target_fpr))

gmm = GaussianMixtureDetector(n_components=16, reg_covar=1e-3, seed=42).fit(x_train_benign)
log_p_val = gmm.log_density(x_val)
gmm.log_epsilon = threshold_at_fpr(log_p_val[y_val == 0], 0.01)   # FPR 1% trên validation
is_anomaly = gmm.log_density(x_new) < gmm.log_epsilon""")
    w.subhead("B.2. Phát hiện beaconing (CV và CV bền vững)")
    w.code("""def interval_stats(times):
    iv = np.diff(np.sort(times))
    mean, std, median = iv.mean(), iv.std(), np.median(iv)
    mad = np.median(np.abs(iv - median))
    return {"median": median,
            "cv": std / mean,                          # nhạy với khoảng nghỉ dài
            "robust_cv": 1.4826 * mad / median}        # bền vững với ngoại lai

times = merge_bursts(np.sort(start_times), tolerance=1.0)   # gộp kết nối cách nhau <= 1 s
st = interval_stats(times)
regular = min(st["cv"], st["robust_cv"])
is_beacon = (len(times) >= 8 and regular <= 0.25 and st["median"] >= 2.0
             and span >= 60.0 and dst_port not in (53, 123))""")

    w.subhead("B.3. Quyết định nhiều tầng cho một flow")
    w.code("""FAMILY_SEVERITY = {"DoS": "HIGH", "DDoS": "HIGH", "BruteForce": "HIGH",
                   "WebAttack": "HIGH", "Infiltration": "CRITICAL", "Botnet": "CRITICAL"}
is_attack = v.predicted_family != "BENIGN"
if is_attack and v.confidence >= tau:      # tau = 0.6
    severity = FAMILY_SEVERITY[v.predicted_family]
    if v.confidence < 0.8:
        severity = bump(severity, -1)      # hạ một mức
    return Alert("AttackClassifier", v.predicted_family, severity), False
if is_attack:                              # không đủ tin cậy -> tầng host
    return Alert("AttackClassifier", "Uncertain", "LOW"), True
if v.is_anomaly:                           # BENIGN nhưng hiếm
    sev = "MEDIUM" if v.confidence < 0.9 else "LOW"
    return Alert("GaussianAnomaly", "Unknown", sev), True
return None, False""")


    w.h2("C. Danh sách đặc trưng")
    sel = json.loads((ROOT / "models" / "preprocessing" / "selected_features.json").read_text())
    safe = set(load("feature_parity_check.json")["parity_safe_features"])
    rows = [[str(i), f, "Có" if f in safe else "Không – loại khỏi mô hình triển khai"] for i, f in enumerate(sel, 1)]
    w.table("30 đặc trưng được chọn và trạng thái khớp CICFlowMeter",
            ["#", "Đặc trưng", "Khớp CICFlowMeter (dùng khi triển khai)"], rows, [0.6, 4.0, 6.4], align="cll", size=20)

    w.h2("D. Rủi ro gặp phải và phương án xử lý")
    w.table("Rủi ro và phương án xử lý",
            ["Rủi ro", "Biểu hiện", "Phương án xử lý"],
            [["Dữ liệu trùng lặp", "Bản sao ở cả train và test, điểm bị thổi phồng", "Khử trùng lặp bằng băm dòng; kiểm tra 0 chồng lấn"],
             ["Nhãn mâu thuẫn", "Cùng vector, hai nhãn", "Loại 1.914 vector trước khi chia"],
             ["LightGBM phân kỳ", "Loss train tăng, Macro-F1 0,62", "min_sum_hessian_in_leaf = 1"],
             ["Sai múi giờ", "pandas 3 đổi đơn vị; đồng hồ 12 giờ", "Đơn vị tường minh; +12 giờ cho giờ < 8"],
             ["Lệch đặc trưng với CICFlowMeter", "GMM báo động 99,3% lưu lượng thật", "Chỉ dùng 24 đặc trưng parity-safe; hiệu chỉnh ε"],
             ["Topology nhận nhầm DNS/web là C&C", "Độ chính xác 34%", "So với mức phổ biến toàn mạng → 82%"],
             ["Bộ nhớ khi xử lý 16 triệu dòng", "Không nạp được vào RAM", "Đọc theo khối PyArrow, băm dòng"],
             ["Quá tải khi bị DoS", "Mất gói, hàng triệu cảnh báo", "Hàng đợi có giới hạn, đếm gói bỏ, gộp cảnh báo 30 s"],
             ["Máy chủ CTU không truy cập được", "Không có nhãn flow chính thức", "Nhãn host, kiểm chứng bằng hành vi"]],
            [3.0, 3.8, 4.2], size=20)


# =====================================================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "BAO_CAO_ATTT.docx"))
    ap.add_argument("--out", default=str(ROOT / "BAO_CAO_ATTT_hoan_thien.docx"))
    a = ap.parse_args()
    d = docx.Document(a.src)
    w = Writer(d)
    front_matter(d, w)

    body = d.element.body
    kids = list(body.iterchildren())
    start = next(i for i, el in enumerate(kids)
                 if el.tag == qn("w:p") and "".join(t.text or "" for t in el.iter(qn("w:t"))).strip()
                 == "PHẦN 1: TỔNG QUAN ĐỀ TÀI")
    for el in kids[start:]:
        if el.tag != qn("w:sectPr"):
            body.remove(el)

    part1(w)
    part2(w)
    part3(w)
    part4(w)
    part5_6(w)
    appendix(w)
    d.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
