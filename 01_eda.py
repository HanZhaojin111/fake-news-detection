"""
社交媒体虚假新闻检测 —— 数据探索性分析 (EDA)
========================================
对 ISOT Fake News 数据集进行探索性分析,
输出数据分布、文本长度、主题分布、关键词等可视化图表。
"""
import os
import re
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 路径配置
# -----------------------------------------------------------------------------
TRUE_PATH = "D:\大二下\py作业\社交媒体虚假新闻检测\True.csv"
FAKE_PATH = "D:\大二下\py作业\社交媒体虚假新闻检测\fake.csv"
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 全局绘图样式
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
PALETTE = {"Real": "#2E86AB", "Fake": "#E63946"}


def load_data():
    """加载数据并打上标签 (1=Real, 0=Fake)"""
    true_df = pd.read_csv(TRUE_PATH)
    fake_df = pd.read_csv(FAKE_PATH)
    true_df["label"] = 1
    fake_df["label"] = 0
    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


# -----------------------------------------------------------------------------
# 图 1: 类别分布
# -----------------------------------------------------------------------------
def plot_label_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    counts = df["label"].map({1: "Real", 0: "Fake"}).value_counts()
    bars = ax[0].bar(counts.index, counts.values,
                     color=[PALETTE[i] for i in counts.index], edgecolor="black")
    ax[0].set_title("News Sample Count by Class", fontsize=13, fontweight="bold")
    ax[0].set_ylabel("Number of Samples")
    for b in bars:
        ax[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 200,
                   f"{int(b.get_height())}", ha="center", fontsize=11)

    ax[1].pie(counts.values, labels=counts.index,
              colors=[PALETTE[i] for i in counts.index],
              autopct="%1.2f%%", startangle=90, wedgeprops={"edgecolor": "white"})
    ax[1].set_title("Class Proportion", fontsize=13, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "01_label_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


# -----------------------------------------------------------------------------
# 图 2: 主题分布
# -----------------------------------------------------------------------------
def plot_subject_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    real_sub = df[df["label"] == 1]["subject"].value_counts()
    fake_sub = df[df["label"] == 0]["subject"].value_counts()

    ax[0].barh(real_sub.index, real_sub.values, color=PALETTE["Real"], edgecolor="black")
    ax[0].set_title("Real News — Subject Distribution", fontsize=13, fontweight="bold")
    ax[0].set_xlabel("Count")
    for i, v in enumerate(real_sub.values):
        ax[0].text(v + 100, i, str(v), va="center")

    ax[1].barh(fake_sub.index, fake_sub.values, color=PALETTE["Fake"], edgecolor="black")
    ax[1].set_title("Fake News — Subject Distribution", fontsize=13, fontweight="bold")
    ax[1].set_xlabel("Count")
    for i, v in enumerate(fake_sub.values):
        ax[1].text(v + 50, i, str(v), va="center")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "02_subject_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


# -----------------------------------------------------------------------------
# 图 3: 文本长度分布
# -----------------------------------------------------------------------------
def plot_text_length(df: pd.DataFrame):
    df = df.copy()
    df["title_len"] = df["title"].fillna("").str.split().str.len()
    df["text_len"] = df["text"].fillna("").str.split().str.len()

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # Title length
    for lbl, name in [(1, "Real"), (0, "Fake")]:
        sub = df[df["label"] == lbl]["title_len"]
        ax[0, 0].hist(sub, bins=40, alpha=0.6, label=name,
                      color=PALETTE[name], edgecolor="black")
    ax[0, 0].set_title("Title Word Count Distribution", fontweight="bold")
    ax[0, 0].set_xlabel("Words"); ax[0, 0].set_ylabel("Frequency"); ax[0, 0].legend()

    # Text length (clipped)
    for lbl, name in [(1, "Real"), (0, "Fake")]:
        sub = df[df["label"] == lbl]["text_len"].clip(0, 1500)
        ax[0, 1].hist(sub, bins=50, alpha=0.6, label=name,
                      color=PALETTE[name], edgecolor="black")
    ax[0, 1].set_title("Body Word Count Distribution (clipped at 1500)", fontweight="bold")
    ax[0, 1].set_xlabel("Words"); ax[0, 1].set_ylabel("Frequency"); ax[0, 1].legend()

    # Boxplots
    sns.boxplot(data=df, x=df["label"].map({1: "Real", 0: "Fake"}),
                y="title_len", ax=ax[1, 0],
                palette=[PALETTE["Fake"], PALETTE["Real"]])
    ax[1, 0].set_title("Title Length — Boxplot", fontweight="bold")
    ax[1, 0].set_xlabel(""); ax[1, 0].set_ylabel("Words")

    sns.boxplot(data=df, x=df["label"].map({1: "Real", 0: "Fake"}),
                y=df["text_len"].clip(0, 2500), ax=ax[1, 1],
                palette=[PALETTE["Fake"], PALETTE["Real"]])
    ax[1, 1].set_title("Body Length — Boxplot (clipped at 2500)", fontweight="bold")
    ax[1, 1].set_xlabel(""); ax[1, 1].set_ylabel("Words")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "03_text_length.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")

    # 返回统计信息
    stats = df.groupby("label").agg(
        title_mean=("title_len", "mean"),
        title_median=("title_len", "median"),
        text_mean=("text_len", "mean"),
        text_median=("text_len", "median"),
    ).round(1)
    stats.index = ["Fake", "Real"]
    return stats


# -----------------------------------------------------------------------------
# 图 4: 高频词 Top20
# -----------------------------------------------------------------------------
STOPWORDS = set("""a an the and or but if then else of in on at to for with by from
about as is are was were be been being have has had do does did will would should could
this that these those it its he she they them their his her our your you we i me my
not no nor so than too very s t don should now also can just over more most other some
such only own same so than too very can will just don should re ve ll d m o""".split())


def tokenize(text: str):
    text = re.sub(r"http\S+|www\.\S+", " ", text)         # 去 URL
    text = re.sub(r"[^a-zA-Z\s]", " ", text).lower()      # 仅保留字母
    return [w for w in text.split() if len(w) > 2 and w not in STOPWORDS]


def plot_top_words(df: pd.DataFrame, n=20):
    real_counter, fake_counter = Counter(), Counter()
    # 为避免一次塞入 4.5 万长文本爆内存,各类抽 5000 篇
    real_sample = df[df["label"] == 1].sample(min(5000, (df["label"] == 1).sum()), random_state=42)
    fake_sample = df[df["label"] == 0].sample(min(5000, (df["label"] == 0).sum()), random_state=42)
    for txt in real_sample["text"].astype(str):
        real_counter.update(tokenize(txt))
    for txt in fake_sample["text"].astype(str):
        fake_counter.update(tokenize(txt))

    real_top = real_counter.most_common(n)[::-1]
    fake_top = fake_counter.most_common(n)[::-1]

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].barh([w for w, _ in real_top], [c for _, c in real_top],
               color=PALETTE["Real"], edgecolor="black")
    ax[0].set_title(f"Top {n} Words in Real News", fontweight="bold")
    ax[0].set_xlabel("Frequency")

    ax[1].barh([w for w, _ in fake_top], [c for _, c in fake_top],
               color=PALETTE["Fake"], edgecolor="black")
    ax[1].set_title(f"Top {n} Words in Fake News", fontweight="bold")
    ax[1].set_xlabel("Frequency")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "04_top_words.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")
    return real_top[::-1][:10], fake_top[::-1][:10]


# -----------------------------------------------------------------------------
# 图 5: 时间分布
# -----------------------------------------------------------------------------
def plot_time_distribution(df: pd.DataFrame):
    df = df.copy()
    df["pub_date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["pub_date"])
    df["year_month"] = df["pub_date"].dt.to_period("M").astype(str)

    pivot = df.groupby(["year_month", "label"]).size().unstack(fill_value=0)
    pivot.columns = ["Fake" if c == 0 else "Real" for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(14, 5))
    pivot.plot(kind="line", ax=ax, marker="o",
               color=[PALETTE["Fake"], PALETTE["Real"]], linewidth=2)
    ax.set_title("News Publication Volume Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year-Month"); ax.set_ylabel("Article Count")
    ax.legend(title="Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "05_time_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Step 1: Exploratory Data Analysis (EDA)")
    print("=" * 60)
    df = load_data()
    print(f"Total samples: {len(df)}")
    print(f"  Real news:   {(df.label == 1).sum()}")
    print(f"  Fake news:   {(df.label == 0).sum()}")
    print()

    plot_label_distribution(df)
    plot_subject_distribution(df)
    length_stats = plot_text_length(df)
    print("\nLength stats (words):")
    print(length_stats)

    real_top, fake_top = plot_top_words(df)
    plot_time_distribution(df)

    print("\nEDA finished. Figures saved to:", os.path.abspath(FIG_DIR))


if __name__ == "__main__":
    main()
