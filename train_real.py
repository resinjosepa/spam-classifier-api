import csv

texts = []
labels = []

with open("spam.csv", "r", encoding="latin-1") as file:
    reader = csv.reader(file)
    next(reader)  # skip header

    for row in reader:
        label = row[0]
        message = row[1]

        texts.append(message)

        if label == "spam":
            labels.append(1)
        else:
            labels.append(0)

print("Total messages:", len(texts))
print("First 5 messages:", texts[:5])
print("First 5 labels:", labels[:5])
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Convert text to numbers
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(texts)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained and saved with real data ✅")