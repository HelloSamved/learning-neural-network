"""Portfolio-ready Streamlit interface for the NumPy MNIST project."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import mnist_prediction as model


PROJECT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_DIR / "mnist_weights.npz"
SAMPLES_PATH = PROJECT_DIR / "mnist_samples.npz"

st.set_page_config(
    page_title="Neural Ink | MNIST Classifier",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #090d18; color: #eef2ff; }
    [data-testid="stSidebar"] { background: #0d1323; border-right: 1px solid #26304a; }
    .hero {
        padding: 2.8rem 3rem; border: 1px solid #293554; border-radius: 24px;
        background: radial-gradient(circle at 85% 20%, #6d4aff55, transparent 32%),
                    linear-gradient(135deg, #111a30, #0b1020);
        margin-bottom: 1.3rem;
    }
    .eyebrow { color: #9b8cff; font-weight: 700; letter-spacing: .16em; font-size: .75rem; }
    .hero h1 { font-size: 3.5rem; line-height: 1; margin: .65rem 0 1rem; }
    .hero p { color: #aab5cf; max-width: 720px; font-size: 1.08rem; }
    .chip {
        display: inline-block; padding: .35rem .7rem; border-radius: 99px;
        background: #1b2540; color: #c7d2fe; margin: .35rem .35rem 0 0; font-size: .8rem;
    }
    .result {
        border: 1px solid #6d5dfc; border-radius: 20px; padding: 1.5rem;
        background: linear-gradient(145deg, #171b36, #10162a); text-align: center;
    }
    .digit { font-size: 6rem; font-weight: 800; line-height: 1; color: #a999ff; }
    .muted { color: #9aa7c2; }
    div[data-testid="stMetric"] {
        background: #11182a; border: 1px solid #25304b; padding: 1rem; border-radius: 16px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: .5rem; }
    .stTabs [data-baseweb="tab"] { background: #11182a; border-radius: 10px; padding: 0 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_weights():
    return model.load_model(WEIGHTS_PATH)


@st.cache_data
def load_samples():
    samples = np.load(SAMPLES_PATH)
    return samples["images"], samples["labels"]


def probability_chart(probabilities):
    chart_data = pd.DataFrame(
        {"Digit": [str(digit) for digit in range(10)], "Confidence": probabilities}
    ).set_index("Digit")
    st.bar_chart(chart_data, color="#8b7cf6", height=280)


def prediction_panel(image, inputs, parameters, source_label):
    probabilities = model.predict_proba(inputs, parameters)[:, 0]
    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction])

    left, right = st.columns([0.85, 1.45], gap="large")
    with left:
        st.image(image.resize((280, 280)), caption=source_label, width=280)
    with right:
        st.markdown(
            f"""
            <div class="result">
                <div class="muted">NETWORK PREDICTION</div>
                <div class="digit">{prediction}</div>
                <strong>{confidence:.1%} confidence</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Confidence by class")
        probability_chart(probabilities)


if not WEIGHTS_PATH.exists():
    st.error(
        "Model weights are missing. Run `python mnist_prediction.py` once to train "
        "and create `mnist_weights.npz`."
    )
    st.stop()

parameters = load_weights()
saved_model = np.load(WEIGHTS_PATH)
validation_accuracy = float(saved_model.get("validation_accuracy", 0))
training_accuracy = float(saved_model.get("training_accuracy", 0))

with st.sidebar:
    st.markdown("## ✦ Neural Ink")
    st.caption("A handwritten digit classifier built from scratch")
    st.divider()
    st.markdown("**Architecture**")
    st.code("784 → 28 → 28 → 10", language=None)
    st.markdown("**Built with**")
    st.markdown(
        '<span class="chip">NumPy</span><span class="chip">Pandas</span>'
        '<span class="chip">Streamlit</span><span class="chip">Pillow</span>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption(
        "No TensorFlow. No PyTorch. Forward propagation, backpropagation, "
        "and gradient descent are implemented directly with NumPy."
    )

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">FROM-SCRATCH NEURAL NETWORK</div>
        <h1>Pixels in.<br>Predictions out.</h1>
        <p>
            Explore how a compact dense neural network learns to recognize
            handwritten digits from 784 raw pixel values.
        </p>
        <span class="chip">42,000 images</span>
        <span class="chip">3 dense layers</span>
        <span class="chip">10 output classes</span>
    </section>
    """,
    unsafe_allow_html=True,
)

metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.metric("Validation accuracy", f"{validation_accuracy:.1%}")
metric_b.metric("Training accuracy", f"{training_accuracy:.1%}")
metric_c.metric("Trainable parameters", "23,082")
metric_d.metric("Framework", "NumPy only")

predict_tab, explore_tab, story_tab = st.tabs(
    ["Try the model", "Explore MNIST", "How it works"]
)

with predict_tab:
    st.markdown("### Upload a handwritten digit")
    st.caption(
        "For the best result, use a dark digit on a light background or a light "
        "digit on a dark background. The image is centered and resized to 28×28."
    )
    uploaded_file = st.file_uploader(
        "Choose a PNG, JPG, JPEG, or WEBP image",
        type=["png", "jpg", "jpeg", "webp"],
    )
    if uploaded_file:
        uploaded_image = Image.open(uploaded_file)
        processed_image, inputs = model.preprocess_image(uploaded_image)
        prediction_panel(processed_image, inputs, parameters, "Model input · 28×28")
    else:
        st.info("Upload an image above, or use the MNIST explorer to test a sample.")

with explore_tab:
    images, labels = load_samples()
    st.markdown("### Test a real MNIST sample")
    sample_index = st.slider("Sample index", 0, len(images) - 1, 42)
    selected_image = images[sample_index]
    selected_label = labels[sample_index]
    sample_inputs = selected_image.reshape(784, 1).astype(np.float64) / 255.0
    prediction_panel(
        Image.fromarray(selected_image),
        sample_inputs,
        parameters,
        f"Dataset label · {selected_label}",
    )

with story_tab:
    st.markdown("### What the notebook is doing")
    st.write(
        "The original notebook reads Kaggle's MNIST CSV, normalizes each grayscale "
        "pixel from 0–255 to 0–1, and sends the resulting 784-value vector through "
        "three fully connected layers."
    )
    layer_a, layer_b, layer_c, layer_d = st.columns(4)
    layer_a.metric("Input", "784", "28 × 28 pixels")
    layer_b.metric("Hidden layer 1", "28", "ReLU")
    layer_c.metric("Hidden layer 2", "28", "ReLU")
    layer_d.metric("Output", "10", "Softmax")
    st.markdown("#### Learning loop")
    st.code(
        """1. Forward propagation computes class probabilities
2. Cross-entropy measures prediction error
3. Backpropagation calculates parameter gradients
4. Mini-batch gradient descent updates weights and biases
5. The loop repeats until predictions improve""",
        language=None,
    )
    st.markdown("#### Improvements made for this demo")
    st.write(
        "The portfolio app keeps the notebook's from-scratch approach while using "
        "softmax for the output layer, stable probability calculations, correctly "
        "shaped bias gradients, mini-batch training, and saved weights for fast deployment."
    )
