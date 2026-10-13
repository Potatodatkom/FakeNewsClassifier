import joblib

# Load trained model and TF-IDF vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("================================")
print("       FAKE NEWS DETECTOR")
print("================================")

# Get article from user
text = input("\nPaste a news article: ")

# Convert text into TF-IDF features
X = vectorizer.transform([text])

# Make prediction
prediction = model.predict(X)[0]

# Get prediction probabilities
probabilities = model.predict_proba(X)[0]

# Convert prediction to label
if prediction == 0:
    label = "FAKE"
else:
    label = "REAL"

# Get confidence
confidence = probabilities[prediction] * 100

print("\n================================")
print(f"Prediction: {label}")
print(f"Confidence: {confidence:.2f}%")
print("================================")