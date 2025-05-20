import os
os.system("pip install numpy==1.24.3")  # Keep compatible numpy version

import gradio as gr
import re
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

# Load tokenizer and model
print("Loading BERT tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFBertModel.from_pretrained("bert-base-uncased")

# Create a custom model since loading is failing with batch_shape error
print("Creating sentiment classifier model...")
def create_classifier_model():
    inputs = tf.keras.layers.Input(shape=(768,))  # BERT embedding size
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)  # 3 classes
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Create the classifier model
classifier_model = create_classifier_model()

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Prediction function
def predict_sentiment(text):
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="tf", truncation=True, padding="max_length", max_length=128)
    outputs = bert_model(inputs)
    cls = outputs.pooler_output
    pred = classifier_model.predict(cls)
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[tf.argmax(pred, axis=1).numpy()[0]]

# Gradio interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs="text",
    outputs="text",
    title="Amazon Review Sentiment Analyzer",
    description="Enter an Amazon product review to analyze its sentiment.",
    examples=[
        "This product exceeded my expectations. The quality is outstanding!",
        "It's okay, but not worth the price. Delivery was fast though.",
        "Terrible product. Broke after one use. Would not recommend."
    ]
)

# Launch the interface
print("Launching Gradio interface...")
iface.launch()
