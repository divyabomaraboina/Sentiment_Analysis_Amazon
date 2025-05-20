import streamlit as st
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

# Load your saved model (assuming you saved it already)
model = tf.keras.models.load_model('bert_sentiment_classifier.h5')

# Load BERT tokenizer and model (for CLS embeddings)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Clean text function (reuse from your project)
import re
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Define prediction function
def predict_sentiment(review_text):
    cleaned_text = clean_text(review_text)
    inputs = tokenizer(cleaned_text, max_length=128, padding='max_length', truncation=True, return_tensors='tf')
    outputs = bert_model(inputs)
    cls_embedding = outputs.pooler_output
    prediction = model.predict(cls_embedding)
    predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
    label_map_reverse = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return label_map_reverse[predicted_class]

# Build the app UI
st.title("Amazon Review Sentiment Analyzer")

user_input = st.text_area("Enter a product review:")
if st.button("Predict Sentiment"):
    result = predict_sentiment(user_input)
    st.write(f"### Predicted Sentiment: {result}")