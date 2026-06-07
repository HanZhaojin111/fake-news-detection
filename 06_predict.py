"""
社交媒体虚假新闻检测 —— 模型综合对比
====================================
将 baseline (LR / SVM / RF) 与深度学习模型 (LSTM / Bi-LSTM)
统一在测试集上对比, 输出柱状图 + 综合表格.
"""
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"


def main():
    with open(os.path.join(RESULTS_DIR, "baseline_metrics.json")) as f:
        base = json.load(f)
    with open(os.path.join(RESULTS_DIR, "lstm_metrics.json")) as f:
        deep = json.load(f)

    rows = []
    for r in base:
        rows.append({
            "Model": r["model"], "Type": "Baseline (TF-IDF)",
            "Accuracy": r["accuracy"], "Precision": r["precision"],
            "Recall": r["recall"], "F1": r["f1"], "AUC": r["auc"],
        })
    for r in deep:
        rows.append({
            "Model": r["model"], "Type": "Deep Learning",
            "Accuracy": r["accuracy"], "Precision": r["precision"],
            "Recall": r["recall"], "F1": r["f1"], "AUC": r["auc"],
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS_DIR, "all_models_metrics.csv"), index=False)

    # ---- 柱状对比图 ----
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    x = np.arange(len(df))
    width = 0.16

    fig, ax = plt.subplots(figsize=(13, 6))
    palette = sns.color_palette("Set2", len(metrics))
    for i, m in enumerate(metrics):
        bars = ax.bar(x + i * width, df[m], width, label=m, color=palette[i],
                      edgecolor="black", linewidth=0.5)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels([f"{r['Model']}\n({r['Type']})" for _, r in df.iterrows()], fontsize=10)
    ax.set_ylim(0.85, 1.005)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison on the Test Set",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.95, ncol=5)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "10_all_models_compare.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")

    # ---- 雷达图 (额外亮点) ----
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    colors_radar = ["#2E86AB", "#A23B72", "#F18F01", "#6A994E", "#BC4749"]
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics] + [row[metrics[0]]]
        ax.plot(angles, vals, "o-", linewidth=2,
                label=row["Model"], color=colors_radar[i % len(colors_radar)])
        ax.fill(angles, vals, alpha=0.08, color=colors_radar[i % len(colors_radar)])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics, size=11)
    ax.set_ylim(0.85, 1.0)
    ax.set_yticks([0.88, 0.92, 0.96, 1.0])
    ax.set_yticklabels(["0.88", "0.92", "0.96", "1.00"], size=9)
    ax.set_title("Radar Chart of Model Performance",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "11_models_radar.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
