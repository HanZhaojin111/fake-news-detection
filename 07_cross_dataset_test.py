"""
社交媒体虚假新闻检测 —— 跨数据集联合训练 (ISOT + LIAR train → LIAR test)
=========================================================================
训练策略:
  - TF-IDF 基线: 在 ISOT 全量训练集 + LIAR train 联合语料上拟合向量化器及分类器
  - LSTM/BiLSTM : 在 ISOT 子集 + LIAR train 序列混合数据上训练,
                  使用 LIAR valid 作为验证集 (in-domain early stopping)
  - 测试集      : 统一使用 LIAR test (体现跨域+领域内双重能力)

输出:
  results/baseline_metrics.json   覆盖写入
  results/lstm_metrics.json       覆盖写入
  results/cross_dataset.flag      内容 "ISOT+LIAR->LIAR"
  figures/12_cross_dataset_confusion.png
"""
import os
import json
import pickle
import re
import time
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.svm import LinearSVC

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR     = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR,     exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"

# ---------- LIAR 标签映射 ----------
LABEL_MAP = {
    "true": 1, "mostly-true": 1,
    "false": 0, "barely-true": 0, "pants-fire": 0,
    # "half-true" 语义模糊，排除
}
# HuggingFace int: 0=false,1=half-true,2=mostly-true,3=true,4=barely-true,5=pants-fire
HF_INT_MAP = {0: 0, 2: 1, 3: 1, 4: 0, 5: 0}

LIAR_COLS = ["id", "label", "statement", "subject", "speaker", "job",
             "state", "party", "bt", "fc", "ht", "mt", "pf", "context"]

LIAR_TSV_BASE = "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/"

# ---------- LSTM 超参 ----------
SEED       = 42
EMBED_DIM  = 64
HIDDEN_DIM = 64
MAX_LEN    = 120
# LSTM 训练: 全量 LIAR train + 等量 ISOT 补充 (控制训练时长)
ISOT_SUPPLEMENT = 5000   # 从 ISOT 补充的样本数 (类别均衡)
BATCH      = 64
EPOCHS     = 6
LR         = 3e-3
CLIP       = 5.0

np.random.seed(SEED)

STOPWORDS = set("""a an the and or but if then else of in on at to for with by from
about as is are was were be been being have has had do does did will would should could
this that these those it its he she they them their his her our your you we i me my
not no nor so than too very s t don should now also can just over more most other some
such only own same so than too very can will just don should re ve ll d m o""".split())


# ===========================================================================
# 文本清洗
# ===========================================================================
def clean_text(text: str) -> str:
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


def _build_rich_text(stmt, speaker="", subject="", context=""):
    """拼接 LIAR 多字段，弥补单条 statement 文本极短的问题."""
    parts = [
        str(stmt).strip(),
        str(speaker).strip(),
        str(subject).replace(",", " ").strip(),
        str(context).strip(),
    ]
    return " ".join(p for p in parts if p and p.lower() != "nan")


# ===========================================================================
# 加载 LIAR 数据集 (train / valid / test 任意分割)
# ===========================================================================
def _load_liar_tsv(split: str) -> pd.DataFrame:
    """从 GitHub 下载指定 split 的 TSV 并解析为 DataFrame."""
    fname = f"liar_{split}.tsv"
    local = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(local):
        url = LIAR_TSV_BASE + f"{split}.tsv"
        print(f"  downloading {url} ...")
        last_err = None
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(url, local)
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < 2:
                    print(f"  retry {attempt+1}/3 ...")
                    time.sleep(2)
        if last_err:
            # 清理可能残留的空文件
            if os.path.exists(local):
                os.remove(local)
            raise RuntimeError(f"无法下载 {url}: {last_err}\n"
                               f"请手动将 {split}.tsv 放到 results/{fname}")
    df = pd.read_csv(local, sep="\t", header=None, names=LIAR_COLS)
    df["label_str"] = df["label"].astype(str).str.strip()
    df = df[df["label_str"].isin(LABEL_MAP)].copy()
    df["label"] = df["label_str"].map(LABEL_MAP)
    df["rich"]  = df.apply(
        lambda r: _build_rich_text(r["statement"], r["speaker"],
                                   r["subject"],   r["context"]), axis=1)
    df["clean"] = df["rich"].apply(clean_text)
    df = df[df["clean"].str.split().str.len() >= 3].reset_index(drop=True)
    return df[["clean", "label"]]


def _load_liar_hf(split: str) -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset("liar", split=split)
    records = []
    for row in ds:
        if row["label"] not in HF_INT_MAP:
            continue
        rich = _build_rich_text(
            row.get("statement", ""), row.get("speaker", ""),
            row.get("subject",   ""), row.get("context", ""),
        )
        records.append({"text": rich, "label": HF_INT_MAP[row["label"]]})
    df = pd.DataFrame(records)
    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"].str.split().str.len() >= 3].reset_index(drop=True)
    return df[["clean", "label"]]


def load_liar(split: str, fallback_df: pd.DataFrame = None) -> pd.DataFrame:
    """尝试 HuggingFace → TSV 下载 → fallback_df (用于 valid 降级)."""
    hf_split = {"train": "train", "valid": "validation", "test": "test"}[split]
    try:
        df = _load_liar_hf(hf_split)
        print(f"  [LIAR {split}] via HuggingFace: {len(df)} samples "
              f"(real={int((df['label']==1).sum())}, fake={int((df['label']==0).sum())})")
        return df
    except Exception as e:
        print(f"  [LIAR {split}] HuggingFace failed ({e}), using TSV ...")
    try:
        df = _load_liar_tsv(split)
        print(f"  [LIAR {split}] via TSV: {len(df)} samples "
              f"(real={int((df['label']==1).sum())}, fake={int((df['label']==0).sum())})")
        return df
    except RuntimeError as e:
        if fallback_df is not None:
            print(f"  [LIAR {split}] download failed, using fallback split from train data")
            return fallback_df
        raise


# ===========================================================================
# 序列工具
# ===========================================================================
PAD_IDX, UNK_IDX = 0, 1


def texts_to_sequences(texts, vocab, max_len=MAX_LEN):
    seqs = np.full((len(texts), max_len), PAD_IDX, dtype=np.int32)
    for i, t in enumerate(texts):
        toks = t.split()[:max_len]
        for j, w in enumerate(toks):
            seqs[i, j] = vocab.get(w, UNK_IDX)
    return seqs


def subset_balanced(seq, y, n, seed=0):
    rng = np.random.RandomState(seed)
    idx_pos = np.where(y == 1)[0]; rng.shuffle(idx_pos)
    idx_neg = np.where(y == 0)[0]; rng.shuffle(idx_neg)
    half = n // 2
    sel  = np.concatenate([idx_pos[:half], idx_neg[:half]])
    rng.shuffle(sel)
    return seq[sel], y[sel]


# ===========================================================================
# NumPy LSTM 实现
# ===========================================================================
def sigmoid(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def softmax_bce_logits(logits, y):
    p   = sigmoid(logits)
    eps = 1e-7
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    return loss, (p - y) / len(y), p


def init_param(shape, scale=None):
    if scale is None:
        scale = np.sqrt(1.0 / shape[-2]) if len(shape) >= 2 else 0.1
    return (np.random.randn(*shape) * scale).astype(np.float32)


class LSTMLayer:
    def __init__(self, input_dim, hidden_dim, name="lstm"):
        self.D, self.H, self.name = input_dim, hidden_dim, name
        self.Wx = init_param((input_dim,  4 * hidden_dim))
        self.Wh = init_param((hidden_dim, 4 * hidden_dim))
        self.b  = np.zeros((4 * hidden_dim,), dtype=np.float32)
        self.b[hidden_dim:2 * hidden_dim] = 1.0

    def params(self):
        return {f"{self.name}_Wx": self.Wx,
                f"{self.name}_Wh": self.Wh,
                f"{self.name}_b":  self.b}

    def forward(self, x_seq, mask):
        B, T, D = x_seq.shape; H = self.H
        h_t = np.zeros((B, H), dtype=np.float32)
        c_t = np.zeros((B, H), dtype=np.float32)
        h_seq    = np.zeros((B, T, H),     dtype=np.float32)
        c_seq    = np.zeros((B, T, H),     dtype=np.float32)
        ifog_seq = np.zeros((B, T, 4 * H), dtype=np.float32)
        for t in range(T):
            gates = x_seq[:, t, :] @ self.Wx + h_t @ self.Wh + self.b
            i = sigmoid(gates[:, :H]);        f = sigmoid(gates[:, H:2*H])
            o = sigmoid(gates[:, 2*H:3*H]);   g = np.tanh(gates[:, 3*H:])
            c_new = f * c_t + i * g;          h_new = o * np.tanh(c_new)
            m   = mask[:, t:t+1]
            c_t = m * c_new + (1 - m) * c_t; h_t = m * h_new + (1 - m) * h_t
            h_seq[:, t, :] = h_t;             c_seq[:, t, :] = c_t
            ifog_seq[:, t, :] = np.concatenate([i, f, o, g], axis=-1)
        return h_t, (x_seq, mask, h_seq, c_seq, ifog_seq)

    def backward(self, dh_last, cache):
        x_seq, mask, h_seq, c_seq, ifog_seq = cache
        B, T, D = x_seq.shape; H = self.H
        dWx = np.zeros_like(self.Wx); dWh = np.zeros_like(self.Wh)
        db  = np.zeros_like(self.b);  dx  = np.zeros_like(x_seq)
        dh_next = dh_last.astype(np.float32)
        dc_next = np.zeros((B, H), dtype=np.float32)
        for t in reversed(range(T)):
            m  = mask[:, t:t+1]
            dh = dh_next * m;  dc = dc_next * m
            i  = ifog_seq[:, t, :H];   f = ifog_seq[:, t, H:2*H]
            o  = ifog_seq[:, t, 2*H:3*H]; g = ifog_seq[:, t, 3*H:]
            c_t    = c_seq[:, t, :]
            c_prev = c_seq[:, t-1, :] if t > 0 else np.zeros_like(c_t)
            do     = dh * np.tanh(c_t)
            dc_tot = dc + dh * o * (1 - np.tanh(c_t)**2)
            d_gates = np.concatenate([
                dc_tot * g  * i * (1 - i),
                dc_tot * c_prev * f * (1 - f),
                do * o * (1 - o),
                dc_tot * i  * (1 - g**2),
            ], axis=-1)
            h_prev = h_seq[:, t-1, :] if t > 0 else np.zeros((B, H), np.float32)
            dWx += x_seq[:, t, :].T @ d_gates
            dWh += h_prev.T @ d_gates
            db  += d_gates.sum(0)
            dx[:, t, :] = d_gates @ self.Wx.T
            dh_next = d_gates @ self.Wh.T + dh * (1 - m)
            dc_next = dc_tot * f + dc * (1 - m)
        return dx, {f"{self.name}_Wx": dWx, f"{self.name}_Wh": dWh, f"{self.name}_b": db}


class LSTMClassifier:
    def __init__(self, vocab_size, embed_dim, hidden_dim, bidirectional=False):
        self.V, self.D, self.H, self.bi = vocab_size, embed_dim, hidden_dim, bidirectional
        self.E        = init_param((vocab_size, embed_dim), scale=0.05)
        self.lstm_fwd = LSTMLayer(embed_dim, hidden_dim, "fwd")
        out_dim = 2 * hidden_dim if bidirectional else hidden_dim
        if bidirectional:
            self.lstm_bwd = LSTMLayer(embed_dim, hidden_dim, "bwd")
        self.W_out = init_param((out_dim, 1))
        self.b_out = np.zeros((1,), dtype=np.float32)

    def forward(self, x_idx, mask):
        x_emb = self.E[x_idx]
        h_f, cf = self.lstm_fwd.forward(x_emb, mask)
        if self.bi:
            h_b, cb = self.lstm_bwd.forward(x_emb[:, ::-1, :], mask[:, ::-1])
            h = np.concatenate([h_f, h_b], -1)
        else:
            cb = None; h = h_f
        logits = (h @ self.W_out + self.b_out).squeeze(-1)
        return logits, (x_idx, x_emb, mask, h_f, cf, cb, h)

    def backward(self, d_logits, cache):
        x_idx, x_emb, mask, h_f, cf, cb, h = cache
        dl = d_logits[:, None]
        dW_out = h.T @ dl;  db_out = dl.sum(0);  dh = dl @ self.W_out.T
        if self.bi:
            H = self.H
            dx_f, gf = self.lstm_fwd.backward(dh[:, :H], cf)
            dx_b, gb = self.lstm_bwd.backward(dh[:, H:], cb)
            dx_emb = dx_f + dx_b[:, ::-1, :];  grads = {**gf, **gb}
        else:
            dx_emb, grads = self.lstm_fwd.backward(dh, cf)
        dE = np.zeros_like(self.E); np.add.at(dE, x_idx, dx_emb)
        grads.update({"E": dE, "W_out": dW_out, "b_out": db_out})
        return grads

    def params(self):
        p = {"E": self.E, "W_out": self.W_out, "b_out": self.b_out}
        p.update(self.lstm_fwd.params())
        if self.bi:
            p.update(self.lstm_bwd.params())
        return p

    def update(self, grads, opt):
        for k, p in self.params().items():
            opt.step(k, p, grads[k])


class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, clip=5.0):
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, beta1, beta2, eps, clip
        self.m, self.v, self.t = {}, {}, 0

    def step(self, key, p, g):
        gnorm = np.linalg.norm(g)
        if gnorm > self.clip:
            g = g * (self.clip / gnorm)
        if key not in self.m:
            self.m[key] = np.zeros_like(p); self.v[key] = np.zeros_like(p)
        self.t += 1
        self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * g
        self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * g**2
        m_hat = self.m[key] / (1 - self.b1**self.t)
        v_hat = self.v[key] / (1 - self.b2**self.t)
        p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def iterate_minibatches(seq, y, batch_size, shuffle=True):
    idx = np.arange(len(seq))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        sub = idx[start:start + batch_size]
        yield seq[sub], y[sub]


def eval_lstm(model, seq, y, batch_size=128):
    probs = []
    for start in range(0, len(seq), batch_size):
        xb   = seq[start:start + batch_size]
        mask = (xb != 0).astype(np.float32)
        logits, _ = model.forward(xb, mask)
        probs.append(sigmoid(logits))
    p   = np.concatenate(probs)
    acc = float(((p > 0.5).astype(np.int32) == y).mean())
    return p, acc


def train_lstm(model, seq_tr, y_tr, seq_va, y_va, name):
    opt = Adam(lr=LR, clip=CLIP)
    best_val_acc, best_state = 0.0, None
    for ep in range(1, EPOCHS + 1):
        t0 = time.time(); losses, accs = [], []
        for xb, yb in iterate_minibatches(seq_tr, y_tr, BATCH):
            mask = (xb != 0).astype(np.float32)
            logits, cache = model.forward(xb, mask)
            loss, dlogits, p = softmax_bce_logits(logits, yb.astype(np.float32))
            grads = model.backward(dlogits, cache)
            model.update(grads, opt)
            losses.append(loss); accs.append(((p > 0.5).astype(np.int32) == yb).mean())
        _, val_acc = eval_lstm(model, seq_va, y_va)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.copy() for k, v in model.params().items()}
        print(f"  [{name}] epoch {ep}/{EPOCHS}  "
              f"train_loss={np.mean(losses):.4f} train_acc={np.mean(accs):.4f}  "
              f"val_acc={val_acc:.4f}  ({time.time()-t0:.1f}s)")
    if best_state:
        for k, v in best_state.items():
            model.params()[k][...] = v


# ===========================================================================
# 主流程
# ===========================================================================
def main():
    print("=" * 65)
    print("Step 7: Joint Training (ISOT + LIAR train) → Test on LIAR test")
    print("=" * 65)

    # ---- 1. 加载 ISOT 训练数据 ----
    print("\n[1] Loading ISOT training data ...")
    npz = np.load(os.path.join(RESULTS_DIR, "preprocessed.npz"), allow_pickle=True)
    X_isot_train = npz["X_train"]
    y_isot_train = npz["y_train"].astype(np.int32)

    seq_npz     = np.load(os.path.join(RESULTS_DIR, "sequences.npz"))
    seq_isot_tr = seq_npz["seq_train"][:, :MAX_LEN].astype(np.int32)
    y_isot_tr   = seq_npz["y_train"].astype(np.int32)

    with open(os.path.join(RESULTS_DIR, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    V = len(vocab)
    print(f"    ISOT train: {len(X_isot_train)} samples  vocab: {V}")

    # ---- 2. 加载 LIAR 三个分割 ----
    print("\n[2] Loading LIAR splits ...")
    liar_train = load_liar("train")

    # valid 下载失败时，从 train 中切出 15% 作为验证集
    from sklearn.model_selection import train_test_split as _tts
    _tr_texts, _va_texts, _tr_labels, _va_labels = _tts(
        liar_train["clean"].values, liar_train["label"].values,
        test_size=0.15, random_state=42, stratify=liar_train["label"].values)
    _fallback_valid = pd.DataFrame({"clean": _va_texts, "label": _va_labels})
    _fallback_train = pd.DataFrame({"clean": _tr_texts, "label": _tr_labels})

    liar_valid = load_liar("valid", fallback_df=_fallback_valid)
    # 若使用了 fallback，liar_train 也要去掉已分出的验证部分，避免数据泄露
    if len(liar_valid) == len(_fallback_valid) and (liar_valid["clean"].values == _fallback_valid["clean"].values).all():
        liar_train = _fallback_train
        print(f"  [LIAR train] adjusted to {len(liar_train)} after removing fallback valid")

    liar_test  = load_liar("test")

    X_liar_tr = liar_train["clean"].values;  y_liar_tr = liar_train["label"].values.astype(np.int32)
    X_liar_va = liar_valid["clean"].values;  y_liar_va = liar_valid["label"].values.astype(np.int32)
    X_liar_te = liar_test["clean"].values;   y_liar_te = liar_test["label"].values.astype(np.int32)

    # ---- 3. 基线模型: ISOT + LIAR train 联合 ----
    print("\n[3] Baseline models (TF-IDF on ISOT + LIAR train, test on LIAR test) ...")
    X_combined = np.concatenate([X_isot_train, X_liar_tr])
    y_combined = np.concatenate([y_isot_train, y_liar_tr])

    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                          min_df=1, max_df=0.95, sublinear_tf=True)
    Xtr_tfidf = vec.fit_transform(X_combined)
    Xte_tfidf = vec.transform(X_liar_te)
    print(f"    TF-IDF shape: {Xtr_tfidf.shape}")

    clf_list = {
        "LogReg":       LogisticRegression(max_iter=1000, C=1.0,
                                           solver="liblinear", random_state=42),
        "LinearSVM":    LinearSVC(C=1.0, random_state=42, max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                               random_state=42),
    }
    baseline_metrics = []
    baseline_preds   = {}
    for name, clf in clf_list.items():
        t0 = time.time()
        clf.fit(Xtr_tfidf, y_combined)
        y_pred = clf.predict(Xte_tfidf)
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(Xte_tfidf)[:, 1]
        elif hasattr(clf, "decision_function"):
            sc = clf.decision_function(Xte_tfidf)
            y_prob = (sc - sc.min()) / (sc.max() - sc.min() + 1e-9)
        else:
            y_prob = y_pred.astype(float)
        m = {
            "model":     name,
            "accuracy":  round(accuracy_score(y_liar_te, y_pred), 4),
            "precision": round(precision_score(y_liar_te, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_liar_te, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_liar_te, y_pred, zero_division=0), 4),
            "auc":       round(roc_auc_score(y_liar_te, y_prob), 4),
        }
        baseline_metrics.append(m)
        baseline_preds[name] = (y_pred, y_prob)
        print(f"  {name} ({time.time()-t0:.1f}s): "
              f"acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  auc={m['auc']:.4f}")

    with open(os.path.join(RESULTS_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(baseline_metrics, f, indent=2)
    print("  [saved] baseline_metrics.json")

    # ---- 4. LSTM / BiLSTM ----
    print("\n[4] LSTM/BiLSTM (ISOT subset + LIAR train, validate on LIAR valid) ...")
    # 将 LIAR 文本转为序列
    seq_liar_tr = texts_to_sequences(X_liar_tr, vocab, MAX_LEN)
    seq_liar_va = texts_to_sequences(X_liar_va, vocab, MAX_LEN)
    seq_liar_te = texts_to_sequences(X_liar_te, vocab, MAX_LEN)

    # 从 ISOT 中均衡补充 ISOT_SUPPLEMENT 条，与 LIAR train 拼接
    seq_isot_sup, y_isot_sup = subset_balanced(seq_isot_tr, y_isot_tr,
                                               ISOT_SUPPLEMENT, seed=99)
    seq_train_mix = np.concatenate([seq_liar_tr, seq_isot_sup], axis=0)
    y_train_mix   = np.concatenate([y_liar_tr,   y_isot_sup])

    # shuffle
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(seq_train_mix))
    seq_train_mix = seq_train_mix[perm];  y_train_mix = y_train_mix[perm]

    print(f"    LSTM train: LIAR train({len(seq_liar_tr)}) + "
          f"ISOT supplement({len(seq_isot_sup)}) = {len(seq_train_mix)}")
    print(f"    LSTM valid: LIAR valid ({len(seq_liar_va)})")
    print(f"    LSTM test : LIAR test  ({len(seq_liar_te)})")

    lstm_metrics = []
    lstm_preds   = {}
    for model_name, bi in [("LSTM", False), ("BiLSTM", True)]:
        print(f"\n>>> Training {model_name} ...")
        np.random.seed(SEED)
        model = LSTMClassifier(V, EMBED_DIM, HIDDEN_DIM, bidirectional=bi)
        train_lstm(model, seq_train_mix, y_train_mix,
                   seq_liar_va, y_liar_va, name=model_name)
        prob, _ = eval_lstm(model, seq_liar_te, y_liar_te)
        pred = (prob > 0.5).astype(np.int32)
        m = {
            "model":     model_name,
            "accuracy":  round(accuracy_score(y_liar_te, pred), 4),
            "precision": round(precision_score(y_liar_te, pred, zero_division=0), 4),
            "recall":    round(recall_score(y_liar_te, pred, zero_division=0), 4),
            "f1":        round(f1_score(y_liar_te, pred, zero_division=0), 4),
            "auc":       round(roc_auc_score(y_liar_te, prob), 4),
        }
        lstm_metrics.append(m)
        lstm_preds[model_name] = (pred, prob)
        print(f"  {model_name} on LIAR test: acc={m['accuracy']:.4f}  "
              f"f1={m['f1']:.4f}  auc={m['auc']:.4f}")

    with open(os.path.join(RESULTS_DIR, "lstm_metrics.json"), "w") as f:
        json.dump(lstm_metrics, f, indent=2)
    print("  [saved] lstm_metrics.json")

    # ---- 5. 写标记文件 ----
    with open(os.path.join(RESULTS_DIR, "cross_dataset.flag"), "w") as f:
        f.write("ISOT+LIAR->LIAR")
    print("  [saved] cross_dataset.flag")

    # ---- 6. 混淆矩阵图 ----
    print("\n[5] Plotting confusion matrices ...")
    all_preds = {**baseline_preds, **lstm_preds}
    n = len(all_preds)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    for ax_, (name, (pred, _)) in zip(axes, all_preds.items()):
        cm = confusion_matrix(y_liar_te, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_,
                    xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"],
                    cbar=False, annot_kws={"size": 11})
        ax_.set_title(name, fontweight="bold")
        ax_.set_xlabel("Predicted"); ax_.set_ylabel("True")
    plt.suptitle("Confusion Matrices — Test on LIAR (Train: ISOT + LIAR train)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "12_cross_dataset_confusion.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out}")

    # ---- 7. 汇总 ----
    print("\n" + "=" * 65)
    print("Results Summary — Test on LIAR (Train: ISOT + LIAR train)")
    print("=" * 65)
    for m in baseline_metrics + lstm_metrics:
        print(f"  {m['model']:14s}  acc={m['accuracy']:.4f}  "
              f"f1={m['f1']:.4f}  auc={m['auc']:.4f}")
    print("\nDone. Run python 05_compare.py to regenerate comparison charts.")


if __name__ == "__main__":
    main()
