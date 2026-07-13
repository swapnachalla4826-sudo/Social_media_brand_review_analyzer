import re
import pickle
 
import numpy as np
import streamlit as st
import tensorflow as tf
 
# ==========================================================
# Page Config
# ==========================================================
 
st.set_page_config(
    page_title="Social Media Brand Review Analyzer",
    page_icon="💬",
    layout="centered"
)
 
# ==========================================================
# Load Artifacts (cached so they load only once)
# ==========================================================
 
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("sentiment_model.keras")
 
    with open("embedding_lookup.pkl", "rb") as f:
        embedding_lookup = pickle.load(f)
 
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
 
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
 
    return model, embedding_lookup, scaler, label_encoder
 
 
model, embedding_lookup, scaler, label_encoder = load_artifacts()
 
EMBEDDING_DIM = 200
 
# ==========================================================
# Preprocessing (must exactly match training notebook)
# ==========================================================
 
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " @ ", text)
    text = re.sub(r"[^a-z@ ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()
 
 
def sentence_vector(text, embedding_lookup, dim=EMBEDDING_DIM):
    words = clean_text(text)
    valid_vectors = [embedding_lookup[w] for w in words if w in embedding_lookup]
 
    if len(valid_vectors) == 0:
        return np.zeros(dim)
 
    # gensim's KeyedVectors.get_mean_vector() (used in training) normalizes
    # each word vector to unit length BEFORE averaging (pre_normalize=True
    # by default). This must be replicated exactly, or the resulting vector
    # scale won't match what the StandardScaler and model expect.
    unit_vectors = [v / np.linalg.norm(v) for v in valid_vectors]
 
    return np.mean(unit_vectors, axis=0)
 
 
def predict_sentiment(text):
    vector = sentence_vector(text, embedding_lookup)
    vector = vector.reshape(1, -1)
    vector_scaled = scaler.transform(vector)
 
    probabilities = model.predict(vector_scaled, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
 
    class_probs = {
        label_encoder.classes_[i]: float(probabilities[i])
        for i in range(len(label_encoder.classes_))
    }
 
    return predicted_label, class_probs
 
 
# ==========================================================
# UI
# ==========================================================
 
st.title("💬 Social Media Brand Review Analyzer")
st.write(
    "Enter a tweet, comment, or brand review below to analyze its sentiment "
    "(Positive, Negative, Neutral, or Irrelevant)."
)
 
SENTIMENT_STYLE = {
    "Positive": ("🟢", "success"),
    "Negative": ("🔴", "error"),
    "Neutral": ("🟡", "warning"),
    "Irrelevant": ("⚪", "info"),
}
 
if "text_input_area" not in st.session_state:
    st.session_state["text_input_area"] = ""
 
text_input = st.text_area(
    "Enter text to analyze",
    placeholder="e.g. I absolutely love the new update, it's so much faster now!",
    height=120,
    key="text_input_area"
)
 
analyze_clicked = st.button("Analyze Sentiment", type="primary")
 
if analyze_clicked:
    if not text_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            predicted_label, class_probs = predict_sentiment(text_input)
 
        emoji, style = SENTIMENT_STYLE.get(predicted_label, ("ℹ️", "info"))
        message = f"{emoji} Predicted Sentiment: **{predicted_label}**"
 
        if style == "success":
            st.success(message)
        elif style == "error":
            st.error(message)
        elif style == "warning":
            st.warning(message)
        else:
            st.info(message)
 
        st.subheader("Confidence Scores")
        sorted_probs = dict(
            sorted(class_probs.items(), key=lambda item: item[1], reverse=True)
        )
        for label, prob in sorted_probs.items():
            st.write(f"{label}: {prob:.2%}")
            st.progress(prob)
 
st.divider()
 
with st.expander("ℹ️ About this app"):
    st.write(
        "This app classifies text into four sentiment categories — Positive, "
        "Negative, Neutral, and Irrelevant — using a neural network trained on "
        "the Twitter Sentiment Analysis dataset. Text is converted into a "
        "200-dimensional vector by averaging pretrained GloVe (glove-twitter-200) "
        "word embeddings, scaled with StandardScaler, and passed through a "
        "feed-forward neural network for classification."
    )
 
with st.sidebar:
    st.header("Try an example")
    examples = [
        "This product is amazing, best purchase I've made all year!",
        "Worst customer service ever, I'm never buying from this brand again.",
        "The event starts at 6pm at the downtown venue.",
        "Not sure how I feel about the new packaging design.",
    ]
    for example in examples:
        if st.button(example, key=example):
            st.session_state["text_input_area"] = example
            st.rerun()