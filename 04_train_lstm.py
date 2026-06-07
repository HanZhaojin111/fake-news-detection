"""
社交媒体虚假新闻检测 —— PyTorch 版 LSTM / Bi-LSTM
==================================================
本脚本是 04_train_lstm.py (NumPy 版) 的工业级对应版本.
NumPy 版用于在没有深度学习框架的环境演示原理 (沙箱内已运行并写入报告);
本脚本则用于在装有 PyTorch 的本地/服务器环境上跑全量数据.

依赖:
    pip install torch  (CPU 版即可, GPU 版更快)

运行:
    python 04b_train_lstm_pytorch.py

输入:
    results/sequences.npz  (由 02_preprocess.py 生成)
    results/vocab.pkl

输出:
    results/lstm_pt_metrics.json
    figures/12_lstm_pt_curves.png
"""
import os
import json
import time
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[torch] device = {DEVICE}")


# =========================================================================
# 模型
# =========================================================================
class LSTMTextClassifier(nn.Module):
    """单/双向 LSTM 文本分类器.
    Embedding -> LSTM -> 取最后非 pad 时刻 (或双向首尾) -> Dropout -> Linear -> Sigmoid
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128,
                 num_layers=1, bidirectional=False, dropout=0.3, pad_idx=0):
        super().__init__()
        self.bi = bidirectional
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        feat_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feat_dim, 1)

    def forward(self, x, lengths):
        emb = self.embedding(x)                                # (B, T, D)
        # 用 packed sequence 处理变长 (忽略 pad)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)                        # h_n: (num_dirs, B, H)
        if self.bi:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)          # 拼接首尾方向
        else:
            h = h_n[-1]
        h = self.dropout(h)
        logits = self.fc(h).squeeze(-1)
        return logits


# =========================================================================
# 训练
# =========================================================================
def make_loader(seq, y, batch_size, shuffle, max_len):
    seq = seq[:, :max_len].astype(np.int64)
    lengths = (seq != 0).sum(axis=1).astype(np.int64)
    lengths = np.clip(lengths, 1, max_len)                     # 防止全 pad
    ds = TensorDataset(torch.from_numpy(seq),
                       torch.from_numpy(lengths),
                       torch.from_numpy(y.astype(np.float32)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_one(model, train_loader, val_loader, epochs, lr, name):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}
    best_state, best_val = None, -1.0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses, accs = [], []
        for xb, lens, yb in train_loader:
            xb, lens, yb = xb.to(DEVICE), lens.to(DEVICE), yb.to(DEVICE)
            logits = model(xb, lens)
            loss = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
            accs.append(((torch.sigmoid(logits) > 0.5).float() == yb).float().mean().item())

        val_loss, val_acc, _, _, _ = evaluate(model, val_loader)
        history["train_loss"].append(np.mean(losses))
        history["train_acc"].append(np.mean(accs))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"  [{name}] epoch {ep}/{epochs}  "
              f"train_loss={np.mean(losses):.4f} train_acc={np.mean(accs):.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
              f"({time.time() - t0:.1f}s)")

    if best_state:
        model.load_state_dict(best_state)
    return history


def evaluate(model, loader):
    model.eval()
    crit = nn.BCEWithLogitsLoss(reduction="sum")
    losses, ys, probs = 0.0, [], []
    n = 0
    with torch.no_grad():
        for xb, lens, yb in loader:
            xb, lens, yb = xb.to(DEVICE), lens.to(DEVICE), yb.to(DEVICE)
            logits = model(xb, lens)
            losses += crit(logits, yb).item()
            n += len(yb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(yb.cpu().numpy())
    probs = np.concatenate(probs); ys = np.concatenate(ys)
    pred = (probs > 0.5).astype(np.int32)
    return losses / n, accuracy_score(ys, pred), pred, probs, ys


# =========================================================================
# 主流程
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=300)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print("=" * 60)
    print("Step 4b: PyTorch LSTM / Bi-LSTM (full data)")
    print("=" * 60)

    data = np.load(os.path.join(RESULTS_DIR, "sequences.npz"))
    with open(os.path.join(RESULTS_DIR, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    V = len(vocab)
    print(f"vocab_size = {V}")

    train_loader = make_loader(data["seq_train"], data["y_train"],
                               args.batch, True, args.max_len)
    val_loader = make_loader(data["seq_val"], data["y_val"],
                             args.batch, False, args.max_len)
    test_loader = make_loader(data["seq_test"], data["y_test"],
                              args.batch, False, args.max_len)
    print(f"train batches: {len(train_loader)}, "
          f"val: {len(val_loader)}, test: {len(test_loader)}")

    metrics = []
    histories = {}
    for bidi, name in [(False, "LSTM"), (True, "BiLSTM")]:
        print(f"\n>>> Training {name} ...")
        torch.manual_seed(42)
        model = LSTMTextClassifier(
            V, embed_dim=args.embed, hidden_dim=args.hidden,
            bidirectional=bidi, dropout=0.3,
        ).to(DEVICE)
        hist = train_one(model, train_loader, val_loader,
                         args.epochs, args.lr, name)
        histories[name] = hist

        # test
        _, _, pred, prob, ys = evaluate(model, test_loader)
        m = {
            "model": name,
            "accuracy":  round(accuracy_score(ys, pred), 4),
            "precision": round(precision_score(ys, pred), 4),
            "recall":    round(recall_score(ys, pred), 4),
            "f1":        round(f1_score(ys, pred), 4),
            "auc":       round(roc_auc_score(ys, prob), 4),
        }
        metrics.append(m)
        print(f"  {name}: {m}")

        # save model weights
        torch.save(model.state_dict(),
                   os.path.join(RESULTS_DIR, f"{name.lower()}_pt.pth"))

    # save metrics
    with open(os.path.join(RESULTS_DIR, "lstm_pt_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # plot curves
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    epochs_x = list(range(1, args.epochs + 1))
    for r, key, ttl in [(0, "loss", "Loss"), (1, "acc", "Accuracy")]:
        for c, name in enumerate(["LSTM", "BiLSTM"]):
            h = histories[name]
            ax[r, c].plot(epochs_x, h[f"train_{key}"], "o-",
                          color="#2E86AB", label="Train")
            ax[r, c].plot(epochs_x, h[f"val_{key}"], "s--",
                          color="#A23B72", label="Val")
            ax[r, c].set_title(f"{name} — {ttl}", fontweight="bold")
            ax[r, c].set_xlabel("Epoch")
            ax[r, c].set_ylabel(ttl)
            ax[r, c].legend()
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "12_lstm_pt_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
