"""Generate the analysis notebooks (code + commentary) with nbformat, then execute them.

    python notebooks/build_notebooks.py            # build + execute all
    python notebooks/build_notebooks.py --no-exec  # only build

Commentary cells are stored in notebooks/commentary/<notebook>.json (key -> markdown)
so they can be written after the results are known.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
COMMENTARY = HERE / "commentary"

SETUP = """import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src.utils import visualization as viz
from src.utils.config import project_path
pd.set_option('display.max_columns', 60); pd.set_option('display.width', 200)
FIG, METRICS, EXP = project_path('figures'), project_path('metrics'), project_path('experiments')
%matplotlib inline"""


def md(nb_name: str, key: str, default: str = "") -> nbformat.NotebookNode:
    path = COMMENTARY / f"{nb_name}.json"
    text = json.loads(path.read_text(encoding="utf-8")).get(key, default) if path.exists() else default
    return new_markdown_cell(text or f"*Nhận xét `{key}`: chưa có.*")


# --------------------------------------------------------------------------- 01 EDA
def eda() -> nbformat.NotebookNode:
    n = "01_data_exploration"
    c = [
        new_markdown_cell("# 01 – Khám phá dữ liệu CSE-CIC-IDS2018\n\n"
                          "Thống kê ở mục 1–5 được tính trên **toàn bộ 10 file CSV** (pass streaming trong "
                          "`dataset_loader.inspect_cicids`). Phân phối đặc trưng ở mục 6–12 dùng **tập làm việc** "
                          "đã khử trùng lặp (`data/interim/cicids2018_sample.parquet`), vì toàn bộ dữ liệu không "
                          "vừa RAM."),
        new_code_cell(SETUP),
        new_markdown_cell("## 1. Tổng quan dataset"),
        new_code_cell("""ins = json.loads((METRICS/'cicids_inspection.json').read_text())
files = pd.DataFrame(ins['files']).T
files['size_MB'] = (files['size_bytes']/2**20).round(1)
files[['rows','header_rows','size_MB','extra_columns']].sort_index()"""),
        new_code_cell("""print('Tổng số dòng:', f"{ins['total_rows']:,}", '| số cột đặc trưng:', ins['n_feature_columns'])
print('Cột hằng số trên toàn bộ dữ liệu:', ins['constant_columns'])"""),
        md(n, "overview"),
        new_markdown_cell("## 2. Giá trị thiếu, vô cực, giá trị âm"),
        new_code_cell("""stats = pd.read_csv(METRICS/'cicids_column_stats_raw.csv')
bad = stats[(stats.nan>0)|(stats.pos_inf>0)|(stats.neg_inf>0)][['feature','nan','pos_inf','neg_inf']]
display(bad)
neg = stats[stats.negative>0][['feature','negative','min']].sort_values('negative', ascending=False)
display(neg)
print('NaN cells:', ins['total_nan_cells'], '| +Inf:', ins['total_pos_inf_cells'], '| -Inf:', ins['total_neg_inf_cells'])"""),
        md(n, "missing"),
        new_markdown_cell("## 3. Bản ghi trùng lặp"),
        new_code_cell("""lab = pd.DataFrame({'raw_rows': pd.Series(ins['label_counts']),
                    'unique_rows': pd.Series(ins['unique_rows_per_label'])})
lab['duplicate_share'] = 1 - lab.unique_rows/lab.raw_rows
print('Dòng trùng (đặc trưng + nhãn):', f"{ins['duplicate_rows_features_and_label']:,}",
      f"({ins['duplicate_rows_features_and_label']/ins['total_rows']:.1%})")
print('Vector đặc trưng mang >1 nhãn:', f"{ins['conflicting_label_feature_vectors']:,}",
      '| số dòng liên quan:', f"{ins['rows_in_conflicting_vectors']:,}")
lab.sort_values('raw_rows', ascending=False).style.format({'raw_rows':'{:,}','unique_rows':'{:,}','duplicate_share':'{:.1%}'})"""),
        md(n, "duplicates"),
        new_markdown_cell("## 4. Phân phối lớp"),
        new_code_cell("""from src.data_processing.dataset_loader import FAMILY_OF
raw = pd.Series(ins['label_counts'])
viz.bar_counts(raw, 'CICIDS2018: số flow theo nhãn gốc (toàn bộ dữ liệu)', FIG/'eda_class_distribution_fine.png')
fam = raw.groupby(raw.index.map(FAMILY_OF)).sum()
viz.bar_counts(fam, 'CICIDS2018: số flow theo nhóm tấn công', FIG/'eda_class_distribution_family.png')
display(fam.sort_values(ascending=False).to_frame('rows').assign(share=lambda d: d.rows/d.rows.sum()))
by_day = pd.DataFrame(ins['label_counts_by_day']).fillna(0).astype(int)
by_day"""),
        new_code_cell("""from IPython.display import Image
Image(FIG/'eda_class_distribution_family.png')"""),
        md(n, "classes"),
        new_markdown_cell("## 5. Thống kê mô tả (toàn bộ dữ liệu, giá trị hữu hạn)"),
        new_code_cell("""stats.set_index('feature')[['min','max','mean','std','median_2pct_sample','zero']].style.format('{:.4g}')"""),
        md(n, "stats"),
        new_markdown_cell("## 6. Phân phối đặc trưng (tập làm việc)"),
        new_code_cell("""sample = pd.read_parquet(project_path('interim')/'cicids2018_sample.parquet')
print(sample.shape)
sample['Family'].value_counts()"""),
        new_code_cell("""key = ['Flow Duration','Tot Fwd Pkts','TotLen Fwd Pkts','Flow Byts/s','Flow Pkts/s','Flow IAT Mean',
       'Pkt Len Mean','Init Fwd Win Byts','SYN Flag Cnt']
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for ax, f in zip(axes.ravel(), key):
    for i, (name, mask) in enumerate([('BENIGN', sample.IsAttack==0), ('ATTACK', sample.IsAttack==1)]):
        v = sample.loc[mask, f].replace([np.inf,-np.inf], np.nan).dropna()
        ax.hist(np.sign(v)*np.log1p(np.abs(v)), bins=60, density=True, histtype='step', linewidth=2,
                color=viz.CATEGORICAL[i], label=name)
    ax.set_title(f, loc='left', fontsize=10); ax.set_xlabel('sign(x)·log(1+|x|)')
axes[0,0].legend()
fig.suptitle('Histogram đặc trưng chính: BENIGN vs ATTACK', x=0.01, ha='left', fontweight="bold")
fig.tight_layout(); viz.save(fig, FIG/'eda_histograms.png')
Image(FIG/'eda_histograms.png')"""),
        md(n, "histograms"),
        new_markdown_cell("## 7. Boxplot theo nhóm tấn công"),
        new_code_cell("""fams = ['BENIGN'] + sorted(f for f in sample.Family.unique() if f!='BENIGN')
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, f in zip(axes.ravel(), ['Flow Duration','Flow Pkts/s','Pkt Len Mean','Tot Fwd Pkts']):
    data = [np.log1p(np.clip(sample.loc[sample.Family==fm, f].replace([np.inf,-np.inf], np.nan).dropna(), 0, None)) for fm in fams]
    bp = ax.boxplot(data, tick_labels=fams, showfliers=False, patch_artist=True, widths=0.55)
    for patch in bp['boxes']: patch.set_facecolor('#cde2fb'); patch.set_edgecolor(viz.CATEGORICAL[0])
    for med in bp['medians']: med.set_color(viz.TEXT)
    ax.set_title(f + ' (log1p)', loc='left', fontsize=10); ax.tick_params(axis='x', rotation=30)
fig.tight_layout(); viz.save(fig, FIG/'eda_boxplots.png')
Image(FIG/'eda_boxplots.png')"""),
        md(n, "boxplots"),
        new_markdown_cell("## 8. Ma trận tương quan"),
        new_code_cell("""from src.data_processing.feature_extractor import CIC_FEATURE_COLUMNS
num = sample[CIC_FEATURE_COLUMNS].replace([np.inf,-np.inf], np.nan)
num = num.loc[:, num.nunique() > 1]
corr = num.sample(min(len(num), 200_000), random_state=0).corr()
viz.correlation_heatmap(corr, 'Pearson correlation của các đặc trưng không hằng số (tập làm việc)', FIG/'eda_correlation_heatmap.png')
pairs = corr.abs().where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().sort_values(ascending=False)
print('Số cặp |r| > 0.98:', int((pairs > 0.98).sum()))
display(pairs.head(15).to_frame('|r|'))
Image(FIG/'eda_correlation_heatmap.png')"""),
        md(n, "correlation"),
        new_markdown_cell("## 9. So sánh Normal vs Attack và 10. Phân tích theo loại tấn công"),
        new_code_cell("""cols = ['Flow Duration','Tot Fwd Pkts','Tot Bwd Pkts','TotLen Fwd Pkts','TotLen Bwd Pkts','Flow Pkts/s',
        'Pkt Len Mean','SYN Flag Cnt','ACK Flag Cnt','FIN Flag Cnt','RST Flag Cnt','Init Fwd Win Byts']
sample.groupby('Family')[cols].median().T.style.format('{:.4g}')"""),
        new_code_cell("""top_ports = (sample.groupby('FineLabel')['Dst Port'].agg(lambda s: ', '.join(f"{int(p)}({c/len(s):.0%})" for p, c in s.value_counts().head(3).items())))
proto = sample.groupby('FineLabel')['Protocol'].agg(lambda s: dict(s.value_counts(normalize=True).round(3)))
pd.DataFrame({'top_dst_ports': top_ports, 'protocol_share': proto, 'rows': sample.FineLabel.value_counts()})"""),
        md(n, "attack_types"),
        new_markdown_cell("## 11. Mất cân bằng lớp"),
        new_code_cell("""fc = sample.Family.value_counts()
print('Tỉ lệ lớn nhất / nhỏ nhất (tập làm việc):', round(fc.max()/fc.min(), 1))
print('Tỉ lệ lớn nhất / nhỏ nhất (toàn bộ dữ liệu):', round(fam.max()/fam.min(), 1))
split = pd.read_csv(METRICS/'split_distribution_family.csv', index_col=0)
split.assign(train_share=lambda d: d.train/d.total, val_share=lambda d: d.val/d.total, test_share=lambda d: d.test/d.total)"""),
        md(n, "imbalance"),
        new_markdown_cell("## 12. Phân tích ngoại lai (IQR)"),
        new_code_cell("""feat = ['Flow Duration','Tot Fwd Pkts','TotLen Fwd Pkts','Flow Byts/s','Flow Pkts/s','Pkt Len Max','Flow IAT Max']
rows = []
ben = sample[sample.IsAttack==0]
for f in feat:
    v = ben[f].replace([np.inf,-np.inf], np.nan).dropna()
    q1, q3 = v.quantile([.25,.75]); hi = q3 + 3*(q3-q1)
    out = sample[f].replace([np.inf,-np.inf], np.nan) > hi
    rows.append({'feature': f, 'upper_fence(benign, 3*IQR)': hi, 'outlier_share_benign': out[sample.IsAttack==0].mean(),
                 'outlier_share_attack': out[sample.IsAttack==1].mean(),
                 'attack_share_among_outliers': sample.loc[out, 'IsAttack'].mean()})
pd.DataFrame(rows).style.format({'upper_fence(benign, 3*IQR)':'{:.4g}','outlier_share_benign':'{:.1%}',
                                  'outlier_share_attack':'{:.1%}','attack_share_among_outliers':'{:.1%}'})"""),
        md(n, "outliers"),
        new_markdown_cell("## 13. Pairplot (mẫu nhỏ) và 14. Feature importance"),
        new_code_cell("""import seaborn as sns
pp = pd.concat([g.sample(min(len(g), 400), random_state=0) for _, g in sample.groupby('Family')])
pf = ['Flow Duration','Flow Pkts/s','Pkt Len Mean','Init Fwd Win Byts']
tmp = pp[pf].replace([np.inf,-np.inf], np.nan).fillna(0).apply(lambda s: np.sign(s)*np.log1p(np.abs(s)))
tmp['Family'] = pp['Family'].values
order = ['BENIGN'] + sorted(f for f in tmp.Family.unique() if f!='BENIGN')
g = sns.pairplot(tmp, hue='Family', hue_order=order, palette=viz.CATEGORICAL[:len(order)], plot_kws={'s':6,'alpha':0.5,'linewidth':0}, corner=True)
g.savefig(FIG/'eda_pairplot.png', dpi=110); plt.close('all')
Image(FIG/'eda_pairplot.png')"""),
        new_code_cell("""rank = pd.read_csv(METRICS/'feature_ranking.csv', index_col=0)
display(rank.head(20).style.format('{:.4g}'))
Image(FIG/'fs_rf_importance.png')"""),
        md(n, "importance"),
        md(n, "conclusion", "## Kết luận"),
    ]
    return new_notebook(cells=c, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})


NOTEBOOKS = {"01_data_exploration": eda}


def build(names, execute: bool) -> None:
    from nbconvert.preprocessors import ExecutePreprocessor
    for name in names:
        nb = NOTEBOOKS[name]()
        path = HERE / f"{name}.ipynb"
        if execute:
            ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
        nbformat.write(nb, path)
        print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(NOTEBOOKS))
    ap.add_argument("--no-exec", action="store_true")
    a = ap.parse_args()
    build(a.names, not a.no_exec)
