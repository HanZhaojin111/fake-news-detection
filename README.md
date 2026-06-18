# 社交媒体虚假新闻检测

> 基于 LSTM/Bi-LSTM 与传统机器学习的端到端虚假新闻检测项目，含跨域泛化研究（三阶段实验）

## 项目简介

本项目以 [ISOT Fake News Dataset](https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/) (44,898 条新闻) 为研究对象，构建了一套完整的虚假新闻自动检测系统，包含：

- 探索性数据分析 (EDA)
- 文本预处理与序列化
- 5 个模型的训练与对比 (Logistic Regression / Linear SVM / Random Forest / LSTM / Bi-LSTM)
- 推理 demo（命令行 + Gradio Web 交互界面）
- **跨域泛化研究（三阶段实验）**：ISOT 内部测试 → 纯跨域评估（ISOT→LIAR）→ 联合训练补充实验（ISOT+LIAR→LIAR）
- 完整的 Word/LaTeX 实验报告

## 核心发现

ISOT 数据集的真实新闻几乎全部来自 Reuters，存在潜在的"来源风格泄漏"问题——模型可能学到的是来源写作风格，而非新闻真伪的本质特征。为验证这一假设，本项目设计了三阶段递进式实验：

### 阶段一：ISOT 内部测试（同源）

| 模型 | 类别 | Accuracy | F1 | AUC |
| :-- | :-- | :-: | :-: | :-: |
| Logistic Regression | Baseline (TF-IDF) | 0.9923 | 0.9919 | 0.9994 |
| **Linear SVM** | Baseline (TF-IDF) | **0.9982** | **0.9981** | 0.9999 |
| Random Forest | Baseline (TF-IDF) | 0.9978 | 0.9977 | 1.0000 |
| LSTM | Deep Learning | 0.9037 | 0.9047 | 0.9489 |
| **Bi-LSTM** | Deep Learning | **0.9980** | **0.9980** | 0.9995 |

五个模型在 ISOT 测试集上 F1 普遍超过 90%，Linear SVM 与 Bi-LSTM 接近满分。

### 阶段二：纯跨域评估（ISOT 训练 → LIAR 测试，未做任何调整）

| 模型 | Accuracy | Precision | Recall | F1 | AUC |
| :-- | :-: | :-: | :-: | :-: | :-: |
| LogReg | 0.4593 | 0.6230 | 0.1064 | 0.1818 | 0.5503 |
| Linear SVM | 0.4427 | 0.5634 | 0.0560 | 0.1019 | 0.5581 |
| Random Forest | 0.4356 | 0.0000 | 0.0000 | 0.0000 | 0.5280 |
| LSTM | 0.5115 | 0.5854 | 0.4608 | **0.5157** | 0.5197 |
| Bi-LSTM | 0.4617 | 0.6154 | 0.1232 | 0.2054 | 0.4958 |

*（half-true 归为 Real，测试样本 1,267 条）*

**F1 普遍降至 0~52%，AUC 接近随机水平（0.50~0.56）**，Random Forest 甚至未能识别出任何一条真实新闻。这一结果证实了 ISOT 内部高分主要依赖来源风格的可分性，而非新闻真伪的本质特征。

### 阶段三：联合训练补充实验（ISOT+LIAR 联合训练 → LIAR 测试）

| 模型 | Accuracy | Precision | Recall | F1 | AUC |
| :-- | :-: | :-: | :-: | :-: | :-: |
| LogReg | 0.6158 | 0.5909 | 0.4633 | 0.5194 | 0.6621 |
| Linear SVM | 0.6248 | 0.5901 | 0.5323 | 0.5597 | 0.6702 |
| Random Forest | 0.6267 | 0.6404 | 0.3808 | 0.4777 | 0.6698 |
| LSTM | 0.5379 | 0.4764 | 0.3140 | 0.3785 | 0.5209 |
| **Bi-LSTM** | 0.6158 | 0.5705 | 0.5768 | **0.5736** | 0.6658 |

*（half-true 排除，测试样本 1,002 条；与阶段二标签空间不同，不可直接数值对比，仅作趋势参考）*

引入目标域（LIAR）训练数据后，F1 回升至 38~57%，AUC 回升至 0.52~0.67，但仍远低于阶段一的 ISOT 内部水平——说明政治声明类文本的真伪判别本质上更依赖事实核查，而非写作风格，是比 ISOT 更具挑战性的任务。

## 项目结构

```
project/
├── 01_eda.py                       # 数据探索
├── 02_preprocess.py                # 数据预处理
├── 03_train_baseline.py            # TF-IDF + 三个基线模型（阶段一）
├── 04_train_lstm.py                # NumPy 实现的 LSTM/Bi-LSTM（阶段一）
├── 04b_train_lstm_pytorch.py       # PyTorch 实现的 LSTM/Bi-LSTM
├── 05_compare.py                   # 模型综合对比
├── 06_predict.py                   # 推理 demo（命令行）
├── 07_cross_dataset_test.py        # 跨数据集评测（阶段二：纯跨域 / 阶段三：联合训练）
├── app.py                          # Gradio Web 交互界面
├── .gitignore
├── results/                        # 模型权重、指标、序列化数据
│   ├── baseline_metrics.json       # 当前为最近一次运行结果（阶段一/二/三视运行而定）
│   ├── lstm_metrics.json
│   ├── cross_dataset_summary.csv   # 跨数据集评测汇总（5 模型 × 5 指标）
│   ├── best_baseline.pkl
│   ├── preprocessed.npz
│   ├── sequences.npz
│   ├── vocab.pkl
│   └── liar_test.tsv               # LIAR 测试集（跨域评估用）
├── figures/                        # 可视化图表
│   ├── 01_label_distribution.png ~ 11_models_radar.png   # 阶段一 EDA / 模型对比图
│   └── 12_cross_dataset_confusion.png                    # 跨数据集 5 模型混淆矩阵
└── README.md
```

## 数据集

**ISOT 训练数据**请从以下地址下载，将 `True.csv` 和 `fake.csv` 放置于项目根目录：

> [ISOT Fake News Dataset](https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/)

**LIAR 数据集**（跨域评估用）由脚本自动下载，镜像地址：

```python
"https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/"
```

若网络环境无法访问，可手动下载 `test.tsv`（及 `train.tsv`、`valid.tsv`，联合训练需要）放入 `results/` 目录。

## 运行环境

最小依赖（沙箱内已经跑通的版本）：

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

可选 (运行 PyTorch 版本)：

```bash
pip install torch
```

可选 (运行 Web 界面)：

```bash
pip install gradio
```

## 复现实验

```bash
# 1) 数据探索
python 01_eda.py

# 2) 预处理
python 02_preprocess.py

# 3) 训练 3 个基线（阶段一）
python 03_train_baseline.py

# 4) 训练 LSTM / Bi-LSTM（阶段一，NumPy 版, 任何环境可跑）
python 04_train_lstm.py

# 4b) 可选 - PyTorch 全量数据版本
python 04b_train_lstm_pytorch.py --epochs 4 --max_len 300

# 5) 综合对比图（阶段一）
python 05_compare.py

# 6) 推理 demo（命令行）
python 06_predict.py --text "WASHINGTON (Reuters) - The U.S. Senate ..."

# 7) 跨数据集评测（阶段二：纯跨域 / 阶段三：联合训练，两种模式见脚本内说明）
python 07_cross_dataset_test.py

# 8) Web 交互界面
python app.py
# 启动后浏览器访问 http://127.0.0.1:7860
```

**注意**：运行 `07_cross_dataset_test.py` 会覆盖 `results/baseline_metrics.json` 和 `results/lstm_metrics.json`。若需保留阶段一（ISOT 内部测试）的原始结果用于对比，请在运行前手动备份这两个文件。

## 推理示例

**命令行：**

```
$ python 06_predict.py --text "WASHINGTON (Reuters) - The U.S. Senate on Tuesday..."
[loaded] best baseline = LinearSVM
Prediction: Real  (confidence ≈ 0.8758)

$ python 06_predict.py --text "BREAKING: Hillary Clinton was caught running a child trafficking ring..."
[loaded] best baseline = LinearSVM
Prediction: Fake  (confidence ≈ 0.8849)
```

**Web 界面：**

运行 `python app.py` 后在浏览器打开 `http://127.0.0.1:7860`，在文本框粘贴新闻内容，点击「🔍 检测」即可获得结果与置信度。界面内置 4 条典型样例，支持一键测试。

## 关键发现

1. **ISOT 数据集存在来源风格泄漏**：真实新闻几乎全来自 Reuters，TF-IDF + 线性模型即可达到 99.8% F1，但模型很可能学到的是来源格式而非真假本质。
2. **纯跨域泛化能力极差**：ISOT 训练的模型在 LIAR 数据集上 F1 暴跌至 0~52%，AUC 接近随机水平，证实了来源风格泄漏问题。
3. **联合训练可部分缓解，但无法根本解决**：引入 LIAR 训练数据后 F1 回升至 38~57%，但仍远低于 ISOT 内部水平，说明政治声明判断本质上更难，且 LIAR 学术界 SOTA 也仅在 70~77% 区间。
4. **Bi-LSTM 大幅优于单向 LSTM（ISOT 内部）**：从 90.4% 提升至 99.8%，印证了双向上下文建模的价值；但在 LIAR 短文本+元数据拼接场景下，传统模型（Linear SVM / Random Forest）的 AUC 反而与 Bi-LSTM 相当甚至略优，说明深度模型的优势高度依赖任务特性，并非普遍适用。

## 后续工作建议

- 引入 BERT / RoBERTa 等预训练语言模型，提升跨域语义理解与泛化能力
- 探索基于事实核查/声明验证（claim verification）的方法，而非单纯依赖文本分类
- 引入图片、视频、转发图等多模态信息
- 增加可解释性 (LIME / SHAP / Attention 可视化)
- 评估对抗鲁棒性；进一步探索对抗域适应等更系统的域适应方法
