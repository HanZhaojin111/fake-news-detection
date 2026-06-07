# 社交媒体虚假新闻检测

> 基于 LSTM / Bi-LSTM 与传统机器学习的端到端虚假新闻检测项目

## 项目简介

本项目以 [ISOT Fake News Dataset](https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/) (44,898 条新闻) 为研究对象，构建了一套完整的虚假新闻自动检测系统，包含：

- 探索性数据分析 (EDA)
- 文本预处理与序列化
- 5 个模型的训练与对比 (Logistic Regression / Linear SVM / Random Forest / LSTM / Bi-LSTM)
- 推理 demo
- 完整的 Word 实验报告

## 核心结果

| 模型 | 类别 | Accuracy | F1 | AUC |
| :-- | :-- | :-: | :-: | :-: |
| Logistic Regression | Baseline (TF-IDF) | 0.9923 | 0.9919 | 0.9994 |
| **Linear SVM** | Baseline (TF-IDF) | **0.9982** | **0.9981** | 0.9999 |
| Random Forest | Baseline (TF-IDF) | 0.9978 | 0.9977 | 1.0000 |
| LSTM | Deep Learning | 0.9037 | 0.9047 | 0.9489 |
| **Bi-LSTM** | Deep Learning | **0.9980** | **0.9980** | 0.9995 |

## 项目结构

```
project/
├── src/                                # 源代码
│   ├── 01_eda.py                       # 数据探索
│   ├── 02_preprocess.py                # 数据预处理
│   ├── 03_train_baseline.py            # TF-IDF + 三个基线模型
│   ├── 04_train_lstm.py                # NumPy 实现的 LSTM/Bi-LSTM (沙箱可跑)
│   ├── 04b_train_lstm_pytorch.py       # PyTorch 实现的 LSTM/Bi-LSTM (本地跑全量)
│   ├── 05_compare.py                   # 模型综合对比
│   ├── 06_predict.py                   # 推理 demo
│   └── build_report.js                 # 报告生成 (docx-js)
├── results/                            # 模型权重、指标、序列化数据
│   ├── baseline_metrics.json
│   ├── baseline_results.csv
│   ├── best_baseline.pkl               # 训练好的最佳基线 (Linear SVM)
│   ├── lstm_history.json
│   ├── lstm_metrics.json
│   ├── all_models_metrics.csv
│   ├── preprocessed.npz                # 清洗后的文本与标签
│   ├── sequences.npz                   # 整数序列 (供 LSTM 使用)
│   └── vocab.pkl                       # 词表
├── figures/                            # 11 张可视化图表
│   ├── 01_label_distribution.png
│   ├── 02_subject_distribution.png
│   ├── 03_text_length.png
│   ├── 04_top_words.png
│   ├── 05_time_distribution.png
│   ├── 06_baseline_confusion.png
│   ├── 07_baseline_roc.png
│   ├── 08_lstm_curves.png
│   ├── 09_lstm_confusion.png
│   ├── 10_all_models_compare.png
│   └── 11_models_radar.png
└── report/
    └── 实验报告_社交媒体虚假新闻检测.docx
```

## 运行环境

最小依赖（沙箱内已经跑通的版本）：

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

可选 (运行 PyTorch 版本)：

```bash
pip install torch
```

## 复现实验

```bash
# 1) 数据探索
python src/01_eda.py

# 2) 预处理
python src/02_preprocess.py

# 3) 训练 3 个基线
python src/03_train_baseline.py

# 4) 训练 LSTM / Bi-LSTM (NumPy 版, 任何环境可跑)
python src/04_train_lstm.py

# 4b) 可选 - PyTorch 全量数据版本
python src/04b_train_lstm_pytorch.py --epochs 4 --max_len 300

# 5) 综合对比图
python src/05_compare.py

# 6) 推理 demo
python src/06_predict.py --text "WASHINGTON (Reuters) - The U.S. Senate ..."
```

## 推理示例

```
$ python src/06_predict.py --text "WASHINGTON (Reuters) - The U.S. Senate on Tuesday..."
[loaded] best baseline = LinearSVM
Prediction: Real  (confidence ≈ 0.8758)

$ python src/06_predict.py --text "BREAKING: Hillary Clinton was caught running a child trafficking ring..."
[loaded] best baseline = LinearSVM
Prediction: Fake  (confidence ≈ 0.8849)
```

## 关键发现

1. **数据集本身可分性极高**：来源差异 (Reuters vs. 不可靠站点) 使得 TF-IDF 配上线性模型即可达到 99% 准确率。
2. **Bi-LSTM 大幅优于单向 LSTM**：从 90.4% 提升至 99.8%，印证了双向上下文建模的价值。
3. **效率最优解是 Linear SVM**：F1=0.9981、训练 2.7 秒、推理毫秒级，是该任务的工程最优选择。
4. **Bi-LSTM 是深度方案的良好起点**：可便捷扩展到 Attention、Transformer，以及多模态架构。

## 后续工作建议

- 引入 LIAR、FakeNewsNet 等多源数据集，验证跨源泛化能力
- 升级到 BERT / RoBERTa 等预训练模型
- 引入图片、视频、转发图等多模态信息
- 增加可解释性 (LIME / SHAP / Attention 可视化)
- 评估对抗鲁棒性
