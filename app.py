"""
社交媒体虚假新闻检测 —— Web 交互界面
运行: python app.py
"""
import os
import pickle
import re
import math

import gradio as gr

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best_baseline.pkl")

STOPWORDS = set("""a an the and or but if then else of in on at to for with by from
about as is are was were be been being have has had do does did will would should could
this that these those it its he she they them their his her our your you we i me my
not no nor so than too very s t don should now also can just over more most other some
such only own same so than too very can will just don should re ve ll d m o""".split())


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


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model_pkg = load_model()


def predict(text: str):
    if not text or not text.strip():
        return "请输入新闻文本", 0.0

    cleaned = clean_text(text)
    if not cleaned:
        return "清洗后文本为空，请提供更多内容", 0.0

    X = model_pkg["vectorizer"].transform([cleaned])
    clf = model_pkg["model"]
    pred = int(clf.predict(X)[0])

    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(X)[0]
        confidence = float(prob[pred])
    elif hasattr(clf, "decision_function"):
        score = float(clf.decision_function(X)[0])
        confidence = 1 / (1 + math.exp(-abs(score)))
    else:
        confidence = None

    label = "✅ Real（真实新闻）" if pred == 1 else "🚨 Fake（虚假新闻）"
    conf_str = f"{confidence:.4f} ({confidence * 100:.2f}%)" if confidence else "N/A"
    return label, conf_str


examples = [
    ["WASHINGTON (Reuters) - The U.S. Senate on Tuesday approved a $1.2 trillion bipartisan infrastructure bill, delivering a victory for President Joe Biden's agenda."],
    ["BREAKING: Hillary Clinton was caught running a child trafficking ring hidden beneath a Washington D.C. pizzeria. Multiple sources confirm that law enforcement has uncovered evidence of this massive conspiracy."],
    ["Scientists at Stanford University have published a new study on the effectiveness of mRNA vaccines against emerging variants of the coronavirus."],
    ["You won't believe what this celebrity did! Click here to see the shocking truth that the mainstream media doesn't want you to know about!"],
]

with gr.Blocks(title="社交媒体虚假新闻检测", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🗞️ 社交媒体虚假新闻检测
        输入一段新闻文本，系统将判断其是 **真实新闻** 还是 **虚假新闻**。
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="新闻文本",
                placeholder="在此粘贴或输入新闻内容...",
                lines=8,
            )
            with gr.Row():
                predict_btn = gr.Button("🔍 检测", variant="primary", size="lg")
                clear_btn = gr.Button("清空", size="lg")

        with gr.Column(scale=2):
            result_label = gr.Textbox(label="检测结果", lines=2, interactive=False)
            result_confidence = gr.Textbox(label="置信度", lines=1, interactive=False)

    gr.Examples(examples=examples, inputs=text_input, label="试试这些例子")

    predict_btn.click(fn=predict, inputs=text_input, outputs=[result_label, result_confidence])
    clear_btn.click(fn=lambda: ("", ""), outputs=[result_label, result_confidence])
    clear_btn.click(fn=lambda: "", outputs=[text_input])

    gr.Markdown(
        """
        ---
        **模型**: Linear SVM (TF-IDF) | 准确率: 99.82% | F1: 0.9981
        """
    )

if __name__ == "__main__":
    demo.launch()
