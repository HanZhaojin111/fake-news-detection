"""
社交媒体虚假新闻检测 —— 基线模型训练
====================================
使用 TF-IDF 特征 + 传统机器学习模型作为基线对照:
    - Logistic Regression
    - Linear SVM
    - Random Forest

输出每个模型在测试集上的 准确率/精确率/召回率/F1, 并绘制混淆矩阵与 ROC 曲线.
"""
import os
import json
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.svm import LinearSVC

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"


def load_data():
    data = np.load(os.path.join(RESULTS_DIR, "preprocessed.npz"), allow_pickle=True)
    return (data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
            data["X_test"], data["y_test"])


def evaluate(name, y_true, y_pred, y_prob=None):
    """计算并返回评估指标字典"""
    metrics = {
        "model": name,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
        "f1":        round(f1_score(y_true, y_pred), 4),
    }
    if y_prob is not None:
        metrics["auc"] = round(roc_auc_score(y_true, y_prob), 4)
    return metrics


def plot_confusion(y_true, y_pred, title, ax):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"],
                cbar=False, annot_kws={"size": 13})
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")


def main():
    print("=" * 60)
    print("Step 3: Baseline Models (TF-IDF + Classical ML)")
    print("=" * 60)

    X_tr, y_tr, X_va, y_va, X_te, y_te = load_data()
    print(f"train: {len(X_tr)} | val: {len(X_va)} | test: {len(X_te)}")

    # ---------- TF-IDF 特征 ----------
    print("\n[1] Fitting TF-IDF (1-2 gram, max_features=20000) ...")
    t0 = time.time()
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                          min_df=3, max_df=0.95, sublinear_tf=True)
    Xtr = vec.fit_transform(X_tr)
    Xva = vec.transform(X_va)
    Xte = vec.transform(X_te)
    print(f"      shape = {Xtr.shape}, took {time.time() - t0:.1f}s")

    # ---------- 训练三个基线 ----------
    models = {
        "LogReg":       LogisticRegression(max_iter=1000, C=1.0, solver="liblinear", random_state=42),
        "LinearSVM":    LinearSVC(C=1.0, random_state=42, max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=None,
                                               n_jobs=-1, random_state=42),
    }

    results = []
    preds_dict = {}     # name -> (y_pred, y_prob)
    train_times = {}

    for name, clf in models.items():
        print(f"\n[2] Training {name} ...")
        t0 = time.time()
        clf.fit(Xtr, y_tr)
        train_times[name] = round(time.time() - t0, 2)
        print(f"      train time: {train_times[name]}s")

        y_pred = clf.predict(Xte)
        # decision score → probability proxy for AUC
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(Xte)[:, 1]
        elif hasattr(clf, "decision_function"):
            scores = clf.decision_function(Xte)
            y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            y_prob = y_pred.astype(float)

        m = evaluate(name, y_te, y_pred, y_prob)
        m["train_time_s"] = train_times[name]
        results.append(m)
        preds_dict[name] = (y_pred, y_prob)

        print(f"      acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
              f"recall={m['recall']:.4f}  f1={m['f1']:.4f}  auc={m['auc']:.4f}")

    # ---------- 保存结果表 ----------
    df_res = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_DIR, "baseline_results.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")
    print(df_res.to_string(index=False))

    # ---------- 保存最佳基线模型 ----------
    best_name = df_res.sort_values("f1", ascending=False).iloc[0]["model"]
    best_clf = models[best_name]
    with open(os.path.join(RESULTS_DIR, "best_baseline.pkl"), "wb") as f:
        pickle.dump({"vectorizer": vec, "model": best_clf, "name": best_name}, f)
    print(f"[saved] best baseline = {best_name}")

    # ---------- 混淆矩阵图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, (y_pred, _)) in zip(axes, preds_dict.items()):
        plot_confusion(y_te, y_pred, name, ax)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "06_baseline_confusion.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")

    # ---------- ROC 曲线 ----------
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"LogReg": "#2E86AB", "LinearSVM": "#A23B72", "RandomForest": "#F18F01"}
    for name, (_, y_prob) in preds_dict.items():
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        auc = roc_auc_score(y_te, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})",
                color=colors[name], linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Baseline Models", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "07_baseline_roc.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")

    # 保存指标 json 供下游和报告使用
    with open(os.path.join(RESULTS_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
