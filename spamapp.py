from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Training data
texts = [
    "Win money now",
    "Claim your prize",
    "Limited time offer",
    "Hello how are you",
    "Let's meet tomorrow",
    "Are you coming to class"
]

labels = [1, 1, 1, 0, 0, 0]

# Train model
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# API route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data['message']

    vector = vectorizer.transform([message])
    prediction = model.predict(vector)[0]

    return jsonify({
        "message": message,
        "prediction": "spam" if prediction == 1 else "not spam"
    })

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)