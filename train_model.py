from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Data
texts = [
    "Win money now",
    "Claim your prize",
    "Limited time offer",
    "Hello how are you",
    "Let's meet tomorrow",
    "Are you coming to class"
]

labels = [1, 1, 1, 0, 0, 0]

# Train
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# Save
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved ✅")