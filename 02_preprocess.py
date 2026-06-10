"""
社交媒体虚假新闻检测 —— 数据预处理
==================================
- 文本清洗 (去 URL/HTML/特殊字符,小写化)
- 去停用词
- 划分训练/验证/测试集 (70/15/15)
- 保存为 .npz / .pkl 供下游模型使用

输出:
    results/preprocessed.npz       清洗后的文本 + 标签 + 划分索引
    results/vocab.pkl              用于 LSTM 的词表
    results/sequences.npz          用于 LSTM 的整数序列
"""
import os
import pickle
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TRUE_PATH = "True.csv"
FAKE_PATH = "fake.csv"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 序列化超参
MAX_VOCAB = 20000     # 最多保留 2 万个高频词
MAX_LEN = 300         # 文本截断/填充到 300 词 (约覆盖中位数附近的样本)
PAD_IDX, UNK_IDX = 0, 1

STOPWORDS = set("""a an the and or but if then else of in on at to for with by from
about as is are was were be been being have has had do does did will would should could
this that these those it its he she they them their his her our your you we i me my
not no nor so than too very s t don should now also can just over more most other some
such only own same so than too very can will just don should re ve ll d m o""".split())


def clean_text(text: str) -> str:
    """统一文本清洗:小写、去 URL/HTML/Twitter 句柄/数字/特殊字符,折叠空白"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)             # URL
    text = re.sub(r"<.*?>", " ", text)                         # HTML 标签
    text = re.sub(r"@\w+", " ", text)                          # @用户
    text = re.sub(r"#(\w+)", r"\1", text)                      # 保留 hashtag 文本
    text = re.sub(r"[^a-z\s]", " ", text)                      # 仅保留小写字母
    text = re.sub(r"\s+", " ", text).strip()                   # 多空白合并
    tokens = [w for w in text.split() if len(w) > 2 and w not in STOPWORDS]
    return " ".join(tokens)


def load_and_clean():
    print("[1/4] Loading raw data ...")
    true_df = pd.read_csv(TRUE_PATH); true_df["label"] = 1
    fake_df = pd.read_csv(FAKE_PATH); fake_df["label"] = 0
    df = pd.concat([true_df, fake_df], ignore_index=True)

    # title 与 text 合并:标题往往含强信号,放在前面
    df["raw"] = df["title"].fillna("") + ". " + df["text"].fillna("")
    print(f"      total raw samples: {len(df)}")

    print("[2/4] Cleaning text ...")
    df["clean"] = df["raw"].apply(clean_text)
    df = df[df["clean"].str.split().str.len() >= 5].reset_index(drop=True)  # 去掉极短文本
    print(f"      after filtering empty/short: {len(df)}")

    # shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame):
    print("[3/4] Splitting train/val/test (70/15/15) ...")
    X = df["clean"].values
    y = df["label"].values

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1765, random_state=42, stratify=y_trainval)
    # 0.1765 ≈ 0.15 / 0.85, 使最终 val 占总体 ~15%

    print(f"      train: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_vocab(texts, max_vocab=MAX_VOCAB):
    """根据训练集构建词表; PAD=0, UNK=1"""
    cnt = Counter()
    for t in texts:
        cnt.update(t.split())
    most_common = cnt.most_common(max_vocab - 2)
    vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
    for w, _ in most_common:
        vocab[w] = len(vocab)
    return vocab


def texts_to_sequences(texts, vocab, max_len=MAX_LEN):
    seqs = np.full((len(texts), max_len), PAD_IDX, dtype=np.int32)
    lens = np.zeros(len(texts), dtype=np.int32)
    for i, t in enumerate(texts):
        toks = t.split()[:max_len]
        for j, w in enumerate(toks):
            seqs[i, j] = vocab.get(w, UNK_IDX)
        lens[i] = len(toks)
    return seqs, lens


def main():
    df = load_and_clean()
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_data(df)

    print("[4/4] Building vocab and sequences ...")
    vocab = build_vocab(X_tr)
    print(f"      vocab size = {len(vocab)}")

    seq_tr, len_tr = texts_to_sequences(X_tr, vocab)
    seq_va, len_va = texts_to_sequences(X_va, vocab)
    seq_te, len_te = texts_to_sequences(X_te, vocab)

    # 保存
    out_text = os.path.join(RESULTS_DIR, "preprocessed.npz")
    np.savez_compressed(
        out_text,
        X_train=X_tr, y_train=y_tr,
        X_val=X_va, y_val=y_va,
        X_test=X_te, y_test=y_te,
    )
    out_seq = os.path.join(RESULTS_DIR, "sequences.npz")
    np.savez_compressed(
        out_seq,
        seq_train=seq_tr, len_train=len_tr, y_train=y_tr,
        seq_val=seq_va, len_val=len_va, y_val=y_va,
        seq_test=seq_te, len_test=len_te, y_test=y_te,
    )
    out_vocab = os.path.join(RESULTS_DIR, "vocab.pkl")
    with open(out_vocab, "wb") as f:
        pickle.dump(vocab, f)

    print(f"\n[saved] {out_text}")
    print(f"[saved] {out_seq}")
    print(f"[saved] {out_vocab}")
    print("\nPreprocessing finished.")


if __name__ == "__main__":
    main()
