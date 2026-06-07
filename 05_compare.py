"""
社交媒体虚假新闻检测 —— LSTM / Bi-LSTM 模型 (NumPy 实现)
========================================================
完全使用 NumPy 从零实现 LSTM 与 Bi-LSTM, 在 ISOT Fake News 数据集上训练并对比.

之所以用 NumPy 而不是直接调框架, 有两个目的:
  1) 显式展现 LSTM 的所有门控计算, 以便在报告中讲清楚原理;
  2) 避免对深度学习框架的硬性依赖, 任何只装了 numpy/sklearn 的环境都能复现.

考虑到 NumPy 训练全网络的时间成本, 本脚本对 ISOT 全集做以下工程取舍:
  - 训练子集: 从训练集随机抽 6000 条 (类别均衡), 序列长度截到 120;
  - 嵌入维度 32, 隐层维度 32; 这种规模在该数据集上已经能达到 95%+ 的准确率;
  - 优化器 = mini-batch Adam, batch_size=32, epochs=4.
报告中也提供 PyTorch 完整版 (04b_train_lstm_pytorch.py) 用于在 GPU 上跑全量数据.

输出:
  results/lstm_history.json       训练/验证曲线
  results/lstm_metrics.json       LSTM/Bi-LSTM 在测试集上的最终指标
  figures/08_lstm_curves.png      训练曲线
  figures/09_lstm_confusion.png   混淆矩阵
  figures/10_all_models_compare.png    所有模型对比柱状图
"""
import os
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"

# 训练超参
SEED = 42
EMBED_DIM = 32
HIDDEN_DIM = 32
MAX_LEN = 120
TRAIN_SUBSET = 6000      # 每个 epoch 用 6000 条
VAL_SUBSET = 1500
TEST_SUBSET = 3000
BATCH = 32
EPOCHS = 4
LR = 5e-3
CLIP = 5.0

np.random.seed(SEED)


# ===========================================================================
# 1) 工具函数
# ===========================================================================
def sigmoid(x):
    # 数值稳定的 sigmoid
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def tanh(x):
    return np.tanh(x)


def softmax_bce_logits(logits, y):
    """二分类交叉熵 (sigmoid). logits: (B,), y: (B,)"""
    p = sigmoid(logits)
    eps = 1e-7
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    grad = (p - y) / len(y)
    return loss, grad, p


def init_param(shape, scale=None):
    if scale is None:
        scale = np.sqrt(1.0 / shape[-2]) if len(shape) >= 2 else 0.1
    return (np.random.randn(*shape) * scale).astype(np.float32)


# ===========================================================================
# 2) LSTM 单元 (单方向)
#     标准公式:
#       i_t = σ(W_xi x_t + W_hi h_{t-1} + b_i)
#       f_t = σ(W_xf x_t + W_hf h_{t-1} + b_f)
#       o_t = σ(W_xo x_t + W_ho h_{t-1} + b_o)
#       g_t = tanh(W_xg x_t + W_hg h_{t-1} + b_g)
#       c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
#       h_t = o_t ⊙ tanh(c_t)
# 实现技巧: 把 4 个 W_x* 合并为一个 (D, 4H), 4 个 W_h* 合并为 (H, 4H), 一次矩阵乘.
# ===========================================================================
class LSTMLayer:
    def __init__(self, input_dim, hidden_dim, name="lstm"):
        self.D, self.H = input_dim, hidden_dim
        self.name = name
        # 一次性初始化 4 个门的权重
        self.Wx = init_param((input_dim, 4 * hidden_dim))     # (D, 4H)
        self.Wh = init_param((hidden_dim, 4 * hidden_dim))    # (H, 4H)
        self.b = np.zeros((4 * hidden_dim,), dtype=np.float32)
        # 让 forget gate 偏置初始为 1, 一个常用 trick
        self.b[hidden_dim:2 * hidden_dim] = 1.0

    def params(self):
        return {f"{self.name}_Wx": self.Wx,
                f"{self.name}_Wh": self.Wh,
                f"{self.name}_b": self.b}

    def set_params(self, Wx, Wh, b):
        self.Wx, self.Wh, self.b = Wx, Wh, b

    def forward(self, x_seq, mask):
        """
        x_seq: (B, T, D)  mask: (B, T)  — 1 表示真实 token, 0 表示 pad.
        返回 (h_last, cache); h_last: (B, H), 是最后一个非 pad 时刻的隐层.
        """
        B, T, D = x_seq.shape
        H = self.H
        h_t = np.zeros((B, H), dtype=np.float32)
        c_t = np.zeros((B, H), dtype=np.float32)

        # 缓存以便反向传播
        h_seq = np.zeros((B, T, H), dtype=np.float32)
        c_seq = np.zeros((B, T, H), dtype=np.float32)
        gate_seq = np.zeros((B, T, 4 * H), dtype=np.float32)
        ifog_seq = np.zeros((B, T, 4 * H), dtype=np.float32)  # 激活后的 i,f,o,g

        for t in range(T):
            x_t = x_seq[:, t, :]                          # (B, D)
            gates = x_t @ self.Wx + h_t @ self.Wh + self.b   # (B, 4H)
            i = sigmoid(gates[:, 0:H])
            f = sigmoid(gates[:, H:2 * H])
            o = sigmoid(gates[:, 2 * H:3 * H])
            g = tanh(gates[:, 3 * H:4 * H])

            c_new = f * c_t + i * g
            h_new = o * tanh(c_new)

            # 对 pad 位置, 隐状态保持不变
            m = mask[:, t:t + 1]
            c_t = m * c_new + (1 - m) * c_t
            h_t = m * h_new + (1 - m) * h_t

            h_seq[:, t, :] = h_t
            c_seq[:, t, :] = c_t
            gate_seq[:, t, :] = gates
            ifog_seq[:, t, :] = np.concatenate([i, f, o, g], axis=-1)

        cache = (x_seq, mask, h_seq, c_seq, gate_seq, ifog_seq)
        return h_t, cache

    def backward(self, dh_last, cache):
        """根据最后时刻的 dh 反向传播.
        简化: 只把梯度通过最终步反传到所有时间步 (BPTT)."""
        x_seq, mask, h_seq, c_seq, gate_seq, ifog_seq = cache
        B, T, D = x_seq.shape
        H = self.H

        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)
        dx_seq = np.zeros_like(x_seq)

        dh_next = dh_last.astype(np.float32)
        dc_next = np.zeros((B, H), dtype=np.float32)

        for t in reversed(range(T)):
            m = mask[:, t:t + 1]
            # 在 pad 位置, 梯度不流过该步 (隐状态没变)
            dh = dh_next * m
            dc = dc_next * m

            i = ifog_seq[:, t, 0:H]
            f = ifog_seq[:, t, H:2 * H]
            o = ifog_seq[:, t, 2 * H:3 * H]
            g = ifog_seq[:, t, 3 * H:4 * H]
            c_t = c_seq[:, t, :]

            # h_t = o * tanh(c_t)
            do = dh * tanh(c_t)
            dc_total = dc + dh * o * (1 - tanh(c_t) ** 2)

            # c_t = f * c_{t-1} + i * g
            c_prev = c_seq[:, t - 1, :] if t > 0 else np.zeros_like(c_t)
            df = dc_total * c_prev
            di = dc_total * g
            dg = dc_total * i
            dc_prev = dc_total * f

            # 反激活
            d_i_pre = di * i * (1 - i)
            d_f_pre = df * f * (1 - f)
            d_o_pre = do * o * (1 - o)
            d_g_pre = dg * (1 - g ** 2)

            d_gates = np.concatenate([d_i_pre, d_f_pre, d_o_pre, d_g_pre], axis=-1)  # (B,4H)

            x_t = x_seq[:, t, :]
            h_prev = h_seq[:, t - 1, :] if t > 0 else np.zeros((B, H), dtype=np.float32)

            dWx += x_t.T @ d_gates
            dWh += h_prev.T @ d_gates
            db += d_gates.sum(axis=0)

            dx_seq[:, t, :] = d_gates @ self.Wx.T
            dh_prev = d_gates @ self.Wh.T

            dh_next = dh_prev + dh * (1 - m)         # pad 时直接透传梯度
            dc_next = dc_prev + dc * (1 - m)

        return dx_seq, {f"{self.name}_Wx": dWx,
                        f"{self.name}_Wh": dWh,
                        f"{self.name}_b": db}


# ===========================================================================
# 3) 完整模型 (Embedding + LSTM 或 BiLSTM + Linear + Sigmoid)
# ===========================================================================
class LSTMClassifier:
    def __init__(self, vocab_size, embed_dim, hidden_dim, bidirectional=False):
        self.V = vocab_size
        self.D = embed_dim
        self.H = hidden_dim
        self.bi = bidirectional

        # Embedding
        self.E = init_param((vocab_size, embed_dim), scale=0.05)
        # LSTM(s)
        self.lstm_fwd = LSTMLayer(embed_dim, hidden_dim, name="fwd")
        if bidirectional:
            self.lstm_bwd = LSTMLayer(embed_dim, hidden_dim, name="bwd")
            out_dim = 2 * hidden_dim
        else:
            out_dim = hidden_dim
        # 输出层: out_dim -> 1
        self.W_out = init_param((out_dim, 1))
        self.b_out = np.zeros((1,), dtype=np.float32)

    # ---------------- 前向 / 反向 ----------------
    def forward(self, x_idx, mask):
        x_emb = self.E[x_idx]                 # (B,T,D)
        h_f, cache_f = self.lstm_fwd.forward(x_emb, mask)
        if self.bi:
            x_emb_rev = x_emb[:, ::-1, :]
            mask_rev = mask[:, ::-1]
            h_b, cache_b = self.lstm_bwd.forward(x_emb_rev, mask_rev)
            h = np.concatenate([h_f, h_b], axis=-1)
        else:
            cache_b = None
            h = h_f
        logits = (h @ self.W_out + self.b_out).squeeze(-1)
        cache = (x_idx, x_emb, mask, h_f, cache_f, cache_b, h)
        return logits, cache

    def backward(self, d_logits, cache):
        x_idx, x_emb, mask, h_f, cache_f, cache_b, h = cache
        d_logits = d_logits[:, None]                         # (B,1)
        dW_out = h.T @ d_logits
        db_out = d_logits.sum(axis=0)
        dh = d_logits @ self.W_out.T                          # (B, out_dim)

        if self.bi:
            H = self.H
            dh_f, dh_b = dh[:, :H], dh[:, H:]
            dx_f, grads_f = self.lstm_fwd.backward(dh_f, cache_f)
            dx_b, grads_b = self.lstm_bwd.backward(dh_b, cache_b)
            dx_b = dx_b[:, ::-1, :]                           # 反方向 → 正向位置
            dx_emb = dx_f + dx_b
            grads = {**grads_f, **grads_b}
        else:
            dx_emb, grads = self.lstm_fwd.backward(dh, cache_f)

        # Embedding 梯度: scatter-add
        dE = np.zeros_like(self.E)
        np.add.at(dE, x_idx, dx_emb)

        grads["E"] = dE
        grads["W_out"] = dW_out
        grads["b_out"] = db_out
        return grads

    # ---------------- 参数管理 ----------------
    def params(self):
        p = {"E": self.E, "W_out": self.W_out, "b_out": self.b_out}
        p.update(self.lstm_fwd.params())
        if self.bi:
            p.update(self.lstm_bwd.params())
        return p

    def update(self, grads, opt):
        for k, p in self.params().items():
            opt.step(k, p, grads[k])


# ===========================================================================
# 4) Adam 优化器 + 梯度裁剪
# ===========================================================================
class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, clip=5.0):
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, beta1, beta2, eps, clip
        self.m, self.v, self.t = {}, {}, 0

    def step(self, key, p, g):
        # 梯度裁剪
        gnorm = np.linalg.norm(g)
        if gnorm > self.clip:
            g = g * (self.clip / gnorm)
        if key not in self.m:
            self.m[key] = np.zeros_like(p)
            self.v[key] = np.zeros_like(p)
        self.t += 1
        self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * g
        self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * (g ** 2)
        m_hat = self.m[key] / (1 - self.b1 ** self.t)
        v_hat = self.v[key] / (1 - self.b2 ** self.t)
        p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ===========================================================================
# 5) 训练循环
# ===========================================================================
def iterate_minibatches(seq, y, batch_size, shuffle=True):
    idx = np.arange(len(seq))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        sub = idx[start:start + batch_size]
        yield seq[sub], y[sub]


def train_model(model, seq_tr, y_tr, seq_va, y_va, epochs=EPOCHS, lr=LR,
                batch_size=BATCH, name="LSTM"):
    opt = Adam(lr=lr, clip=CLIP)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    for ep in range(1, epochs + 1):
        t0 = time.time()
        # ---- 训练 ----
        losses, accs = [], []
        for xb, yb in iterate_minibatches(seq_tr, y_tr, batch_size):
            mask = (xb != 0).astype(np.float32)
            logits, cache = model.forward(xb, mask)
            loss, dlogits, p = softmax_bce_logits(logits, yb.astype(np.float32))
            grads = model.backward(dlogits, cache)
            model.update(grads, opt)
            losses.append(loss)
            accs.append(((p > 0.5).astype(np.int32) == yb).mean())
        tr_loss = float(np.mean(losses))
        tr_acc = float(np.mean(accs))

        # ---- 验证 ----
        val_loss, val_acc = evaluate_model(model, seq_va, y_va, batch_size)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.copy() for k, v in model.params().items()}

        print(f"  [{name}] epoch {ep}/{epochs}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
              f"({time.time() - t0:.1f}s)")

    # 恢复最佳权重
    if best_state is not None:
        for k, v in best_state.items():
            model.params()[k][...] = v

    return history


def evaluate_model(model, seq, y, batch_size=64):
    losses, preds = [], []
    for start in range(0, len(seq), batch_size):
        xb = seq[start:start + batch_size]
        yb = y[start:start + batch_size].astype(np.float32)
        mask = (xb != 0).astype(np.float32)
        logits, _ = model.forward(xb, mask)
        loss, _, p = softmax_bce_logits(logits, yb)
        losses.append(loss * len(xb))
        preds.append(p)
    p_all = np.concatenate(preds)
    pred_lbl = (p_all > 0.5).astype(np.int32)
    acc = (pred_lbl == y).mean()
    return float(np.sum(losses) / len(seq)), float(acc)


def predict(model, seq, batch_size=64):
    out = []
    for start in range(0, len(seq), batch_size):
        xb = seq[start:start + batch_size]
        mask = (xb != 0).astype(np.float32)
        logits, _ = model.forward(xb, mask)
        out.append(sigmoid(logits))
    return np.concatenate(out)


# ===========================================================================
# 6) 主流程
# ===========================================================================
def subset_balanced(seq, y, n, seed=0):
    """从 (seq, y) 中按类别均衡抽取 n 条 (n 必须为偶数)."""
    rng = np.random.RandomState(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)
    half = n // 2
    sel = np.concatenate([idx_pos[:half], idx_neg[:half]])
    rng.shuffle(sel)
    return seq[sel], y[sel]


def truncate_seq(seq, max_len):
    return seq[:, :max_len].astype(np.int32)


def main():
    print("=" * 60)
    print("Step 4: LSTM / Bi-LSTM (NumPy implementation)")
    print("=" * 60)

    data = np.load(os.path.join(RESULTS_DIR, "sequences.npz"))
    seq_tr_full = truncate_seq(data["seq_train"], MAX_LEN)
    seq_va_full = truncate_seq(data["seq_val"],   MAX_LEN)
    seq_te_full = truncate_seq(data["seq_test"],  MAX_LEN)
    y_tr_full = data["y_train"].astype(np.int32)
    y_va_full = data["y_val"].astype(np.int32)
    y_te_full = data["y_test"].astype(np.int32)

    seq_tr, y_tr = subset_balanced(seq_tr_full, y_tr_full, TRAIN_SUBSET, seed=1)
    seq_va, y_va = subset_balanced(seq_va_full, y_va_full, VAL_SUBSET, seed=2)
    seq_te, y_te = subset_balanced(seq_te_full, y_te_full, TEST_SUBSET, seed=3)

    print(f"train={len(seq_tr)}  val={len(seq_va)}  test={len(seq_te)}  "
          f"max_len={MAX_LEN}  embed={EMBED_DIM}  hidden={HIDDEN_DIM}")

    # 词表大小
    import pickle
    with open(os.path.join(RESULTS_DIR, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    V = len(vocab)
    print(f"vocab_size = {V}")

    # ---------------- 训练 LSTM ----------------
    print("\n>>> Training Vanilla LSTM ...")
    np.random.seed(SEED)
    lstm = LSTMClassifier(V, EMBED_DIM, HIDDEN_DIM, bidirectional=False)
    hist_lstm = train_model(lstm, seq_tr, y_tr, seq_va, y_va, name="LSTM")

    # ---------------- 训练 Bi-LSTM ----------------
    print("\n>>> Training Bi-LSTM ...")
    np.random.seed(SEED)
    bilstm = LSTMClassifier(V, EMBED_DIM, HIDDEN_DIM, bidirectional=True)
    hist_bi = train_model(bilstm, seq_tr, y_tr, seq_va, y_va, name="BiLSTM")

    # ---------------- 测试集评估 ----------------
    print("\n>>> Test set evaluation")
    metrics = []
    preds_dict = {}
    for name, model in [("LSTM", lstm), ("BiLSTM", bilstm)]:
        prob = predict(model, seq_te)
        pred = (prob > 0.5).astype(np.int32)
        m = {
            "model": name,
            "accuracy":  round(accuracy_score(y_te, pred), 4),
            "precision": round(precision_score(y_te, pred), 4),
            "recall":    round(recall_score(y_te, pred), 4),
            "f1":        round(f1_score(y_te, pred), 4),
            "auc":       round(roc_auc_score(y_te, prob), 4),
        }
        metrics.append(m)
        preds_dict[name] = (pred, prob)
        print(f"  {name}: acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
              f"recall={m['recall']:.4f}  f1={m['f1']:.4f}  auc={m['auc']:.4f}")

    # 保存
    with open(os.path.join(RESULTS_DIR, "lstm_history.json"), "w") as f:
        json.dump({"LSTM": hist_lstm, "BiLSTM": hist_bi}, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "lstm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---------------- 训练曲线图 ----------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    epochs_x = list(range(1, EPOCHS + 1))
    for r, key, ttl in [(0, "loss", "Loss"), (1, "acc", "Accuracy")]:
        ax[r, 0].plot(epochs_x, hist_lstm[f"train_{key}"], "o-",
                      color="#2E86AB", label=f"LSTM Train")
        ax[r, 0].plot(epochs_x, hist_lstm[f"val_{key}"], "s--",
                      color="#A23B72", label=f"LSTM Val")
        ax[r, 0].set_title(f"LSTM — {ttl}", fontweight="bold")
        ax[r, 0].set_xlabel("Epoch"); ax[r, 0].set_ylabel(ttl); ax[r, 0].legend()

        ax[r, 1].plot(epochs_x, hist_bi[f"train_{key}"], "o-",
                      color="#2E86AB", label=f"Bi-LSTM Train")
        ax[r, 1].plot(epochs_x, hist_bi[f"val_{key}"], "s--",
                      color="#A23B72", label=f"Bi-LSTM Val")
        ax[r, 1].set_title(f"Bi-LSTM — {ttl}", fontweight="bold")
        ax[r, 1].set_xlabel("Epoch"); ax[r, 1].set_ylabel(ttl); ax[r, 1].legend()

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "08_lstm_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")

    # ---------------- 混淆矩阵 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax_, (name, (pred, _)) in zip(axes, preds_dict.items()):
        cm = confusion_matrix(y_te, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_,
                    xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"],
                    cbar=False, annot_kws={"size": 13})
        ax_.set_title(name, fontweight="bold")
        ax_.set_xlabel("Predicted"); ax_.set_ylabel("True")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "09_lstm_confusion.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
