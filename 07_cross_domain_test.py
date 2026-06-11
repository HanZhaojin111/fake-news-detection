"""
跨域泛化测试：ISOT 训练 → LIAR 测试
======================================
验证模型在真实多源数据集上的泛化能力。

背景：
  ISOT 数据集真实新闻几乎全来自 Reuters，假新闻来自固定的不可靠站点，
  两类之间存在强烈的"来源风格泄漏"，导致模型在 ISOT 测试集上准确率虚高。
  本脚本将在 ISOT 上训练的模型直接在 LIAR 数据集上评估，
  考察其在来源多样、风格混杂的真实场景下的泛化能力。

LIAR 标签映射策略：
  六分类 → 二分类
  Real(1): true, mostly-true
  Fake(0): false, barely-true, pants-fire
  half-true: 两种方案均运行
    方案 A：丢弃（最严格，只保留明确真假的样本）
    方案 B：归入 Fake（最宽松，保留全量样本）

运行前提：
  需要 results/best_baseline.pkl（ISOT 训练好的 Linear SVM）
  需要 results/preprocessed.npz（ISOT 预处理数据，用于对比）
  将本脚本放入 src/ 目录后运行：python src/07_cross_domain_test.py

输出：
  results/cross_domain_results.json
  figures/12_cross_domain_compare.png
"""
import os, json, re, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix,
                             classification_report)

# ── 路径配置 ──────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES    = os.path.join(BASE, "results")
FIG    = os.path.join(BASE, "figures")
LIAR_TEST  = os.path.join(BASE, "liar_test.tsv")
LIAR_TRAIN = os.path.join(BASE, "liar_train.tsv")
LIAR_VALID = os.path.join(BASE, "liar_valid.tsv")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
PALETTE = {"Real": "#2E86AB", "Fake": "#E63946"}

# ── 停用词（与 ISOT 预处理保持完全一致）─────────────────────
STOPWORDS = set("""a an the and or but if then else of in on at to for with by from
about as is are was were be been being have has had do does did will would should could
this that these those it its he she they them their his her our your you we i me my
not no nor so than too very s t don should now also can just over more most other some
such only own same so than too very can will just don should re ve ll d m o""".split())


def clean_text(text: str) -> str:
    """与 02_preprocess.py 完全相同的清洗流程，保证特征空间一致"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if len(w) > 2 and w not in STOPWORDS]
    return " ".join(tokens)


def load_liar(path: str) -> pd.DataFrame:
    cols = ["id", "label", "statement", "subject", "speaker", "job",
            "state", "party", "barely_true", "false", "half_true",
            "mostly_true", "pants_fire", "context"]
    df = pd.read_csv(path, sep="\t", header=None, names=cols)
    df["text"] = df["statement"].fillna("")
    return df


def map_labels(df: pd.DataFrame, half_true_as_fake: bool = False):
    """
    六分类 → 二分类
    half_true_as_fake=False：丢弃 half-true（方案 A）
    half_true_as_fake=True ：half-true 归入 Fake（方案 B）
    """
    real_labels = {"true", "mostly-true"}
    fake_labels = {"false", "barely-true", "pants-fire"}

    def _map(lbl):
        if lbl in real_labels:
            return 1
        if lbl in fake_labels:
            return 0
        if half_true_as_fake:
            return 0   # half-true → Fake
        return -1      # 标记为丢弃

    df = df.copy()
    df["binary_label"] = df["label"].map(_map)
    if not half_true_as_fake:
        df = df[df["binary_label"] != -1].reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("跨域泛化测试：ISOT → LIAR")
    print("=" * 60)

    # ── 1. 加载 ISOT 训练好的模型 ──
    model_path = os.path.join(RES, "best_baseline.pkl")
    if not os.path.exists(model_path):
        print(f"\n[ERROR] 找不到 {model_path}")
        print("请先运行 python src/03_train_baseline.py 训练基线模型")
        return
    with open(model_path, "rb") as f:
        pkg = pickle.load(f)
    vec   = pkg["vectorizer"]
    clf   = pkg["model"]
    name  = pkg["name"]
    print(f"\n[1] 加载模型：{name}（在 ISOT 上训练）")

    # ── 2. 加载 ISOT 测试集结果（用于对比）──
    isot_results = None
    baseline_path = os.path.join(RES, "baseline_metrics.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            metrics_list = json.load(f)
        for m in metrics_list:
            if m["model"] == name:
                isot_results = m
                break
    if isot_results:
        print(f"   ISOT 测试集 F1  = {isot_results['f1']:.4f}（已知基准）")

    # ── 3. 处理 LIAR 数据集 ──
    print("\n[2] 加载并预处理 LIAR 测试集")
    liar_raw = load_liar(LIAR_TEST)
    print(f"   原始样本数：{len(liar_raw)}")
    print(f"   标签分布：\n{liar_raw['label'].value_counts().to_string()}")

    # 两种映射方案
    schemes = [
        ("方案A（丢弃half-true）",  False),
        ("方案B（half-true归Fake）", True),
    ]

    all_results = []
    scheme_data = {}

    for scheme_name, half_as_fake in schemes:
        df = map_labels(liar_raw, half_true_as_fake=half_as_fake)
        df["clean"] = df["text"].apply(clean_text)
        df = df[df["clean"].str.split().str.len() >= 3].reset_index(drop=True)

        real_n = (df["binary_label"] == 1).sum()
        fake_n = (df["binary_label"] == 0).sum()
        print(f"\n   {scheme_name}：{len(df)} 条"
              f"（Real={real_n}, Fake={fake_n}）")

        X = vec.transform(df["clean"])
        y_true = df["binary_label"].values
        y_pred = clf.predict(X)

        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(X)
            y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        elif hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X)[:, 1]
        else:
            y_prob = y_pred.astype(float)

        m = {
            "scheme":    scheme_name,
            "n_samples": len(df),
            "n_real":    int(real_n),
            "n_fake":    int(fake_n),
            "accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
            "auc":       round(roc_auc_score(y_true, y_prob), 4),
        }
        all_results.append(m)
        scheme_data[scheme_name] = (y_true, y_pred, y_prob)

        print(f"   Accuracy={m['accuracy']:.4f}  "
              f"Precision={m['precision']:.4f}  "
              f"Recall={m['recall']:.4f}  "
              f"F1={m['f1']:.4f}  "
              f"AUC={m['auc']:.4f}")
        print(f"\n   分类报告：")
        print(classification_report(y_true, y_pred,
                                    target_names=["Fake", "Real"],
                                    digits=4))

    # ── 4. 保存结果 ──
    output = {
        "model":        name,
        "isot_f1":      isot_results["f1"] if isot_results else None,
        "isot_acc":     isot_results["accuracy"] if isot_results else None,
        "liar_results": all_results,
    }
    out_path = os.path.join(RES, "cross_domain_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path}")

    # ── 5. 可视化 ──
    _plot_comparison(isot_results, all_results, FIG, name)
    _plot_confusion(scheme_data, FIG)


def _plot_comparison(isot, liar_list, fig_dir, model_name):
    """柱状图：ISOT vs LIAR 各指标对比"""
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    labels_map = {"accuracy": "Accuracy", "precision": "Precision",
                  "recall": "Recall", "f1": "F1", "auc": "AUC"}

    rows = []
    if isot:
        rows.append({"数据集": "ISOT（原测试集）", **{labels_map[m]: isot[m] for m in metrics}})
    for r in liar_list:
        rows.append({"数据集": f"LIAR {r['scheme']}", **{labels_map[m]: r[m] for m in metrics}})

    df_plot = pd.DataFrame(rows)
    x = np.arange(len(df_plot))
    width = 0.15
    colors = ["#2E86AB", "#E63946", "#F18F01"]

    fig, ax = plt.subplots(figsize=(13, 6))
    metric_labels = [labels_map[m] for m in metrics]
    for i, ml in enumerate(metric_labels):
        bars = ax.bar(x + i * width, df_plot[ml], width,
                      label=ml, color=plt.cm.Set2(i / len(metrics)),
                      edgecolor="black", linewidth=0.5)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(df_plot["数据集"], fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(f"Cross-Domain Generalization: {model_name}\nISOT (in-domain) vs LIAR (out-of-domain)",
                 fontweight="bold")
    ax.legend(loc="upper right", ncol=5, fontsize=9)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(len(df_plot) - 0.3, 0.51, "random baseline", fontsize=8, color="gray")

    plt.tight_layout()
    out = os.path.join(fig_dir, "12_cross_domain_compare.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


def _plot_confusion(scheme_data, fig_dir):
    """LIAR 两种方案的混淆矩阵"""
    n = len(scheme_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (scheme_name, (y_true, y_pred, _)) in zip(axes, scheme_data.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax,
                    xticklabels=["Fake", "Real"],
                    yticklabels=["Fake", "Real"],
                    cbar=False, annot_kws={"size": 13})
        ax.set_title(f"LIAR 测试集混淆矩阵\n{scheme_name}", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    out = os.path.join(fig_dir, "13_liar_confusion.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
