import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gensim
from gensim.models import CoherenceModel
import joblib
from flask import Flask, request, jsonify

#loading pickeled files
model = joblib.load('lda_model.pkl')

#loading tfidf
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

embed_model = tf.saved_model.load('embed_model')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from request
        input_text = request.json['text']

        # Preprocess the input text (e.g., tokenize, clean, and vectorize)

        # Make predictions using your model
        predictions = model.predict([input_text])

        # Format the predictions as needed

        # Return the predictions as JSON response
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)})

    
if __name__ == '__main__':
    # Run the Flask application
       app.run(debug=True)