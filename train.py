import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading datasets...")

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true], ignore_index=True)

data["text"] = data["title"].fillna("") + " " + data["text"].fillna("")

data = data[["text", "label"]]

data = data.dropna()

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total articles: {len(data)}")

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=100000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training Logistic Regression model...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train_tfidf, y_train)

print("Evaluating model...")

predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print("MODEL RESULTS")
print("==============================")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Fake", "Real"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel saved successfully!")
print("fake_news_model.pkl")
print("tfidf_vectorizer.pkl")