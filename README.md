# Amazon Review Sentiment Analyzer

## Overview
This project is a sentiment analysis tool that uses BERT (Bidirectional Encoder Representations from Transformers) to classify Amazon product reviews as positive, negative, or neutral. The application provides a user-friendly web interface built with Gradio, allowing users to input review text and receive instant sentiment analysis results.

## Features
- **BERT-based Sentiment Analysis**: Leverages the power of BERT for contextual understanding of review text
- **Text Preprocessing**: Cleans and normalizes input text to improve analysis accuracy
- **User-friendly Interface**: Simple web UI for easy interaction with the model
- **Real-time Analysis**: Instant sentiment classification of input text

## Technical Architecture
The application consists of two main components:
1. **BERT Model**: Pre-trained BERT model from Hugging Face's Transformers library for text embedding
2. **Sentiment Classifier**: A neural network that takes BERT embeddings and classifies sentiment

## Requirements
- Python 3.8+
- TensorFlow 2.x
- Hugging Face Transformers
- Gradio
- NumPy 1.24.3

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/amazon-review-sentiment-analyzer.git
cd amazon-review-sentiment-analyzer

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install numpy==1.24.3
pip install tensorflow transformers gradio
```

## Usage

```bash
# Run the application
python app.py
```

Once the application is running, open your web browser and navigate to the URL displayed in the terminal. Enter an Amazon product review in the text box and click "Submit" to see the sentiment analysis result.

## How It Works

1. **Text Preprocessing**: The input review text is cleaned by removing URLs, HTML tags, punctuation, and numbers.
2. **BERT Embedding**: The cleaned text is tokenized and processed through the BERT model to generate embeddings.
3. **Sentiment Classification**: The embeddings are passed to a classifier neural network that predicts sentiment.
4. **Result Display**: The predicted sentiment (Positive, Negative, or Neutral) is displayed to the user.

## Model Details
- **BERT Model**: bert-base-uncased from Hugging Face
- **Classifier**: A neural network with a dense hidden layer and softmax output layer
- **Classes**: 3 (Positive, Neutral, Negative)

## Future Improvements
- Add support for batch processing of reviews
- Implement confidence scores for predictions
- Expand to support multiple languages
- Add aspect-based sentiment analysis
- Fine-tune the BERT model on a larger Amazon review dataset

## License
MIT License

## Contact
Your Name - divya.bomaraboina22@gmail.com

Project Link: https://huggingface.co/spaces/bomaraboinadivya/sentiment-analyzer
## Acknowledgments
- Hugging Face for the Transformers library
- The BERT research team
- Gradio for the simple web interface framework

