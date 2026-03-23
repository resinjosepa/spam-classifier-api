from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

texts = [
    "Win money now",
    "Claim your prize",
    "Limited time offer",
    "Hello how are you",
    "Let's meet tomorrow",
    "Are you coming to class"
]

labels = [1, 1, 1, 0, 0, 0]

vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

test_text = ["free meeting offer today"]
test_vector = vectorizer.transform(test_text)

prediction = model.predict(test_vector)

print("Spam" if prediction[0] == 1 else "Not Spam")
print("\nWord Probabilities:\n")

words = vectorizer.get_feature_names_out()
log_probs = model.feature_log_prob_

for i, word in enumerate(words):
    print(f"{word}: spam={log_probs[1][i]:.2f}, not_spam={log_probs[0][i]:.2f}")