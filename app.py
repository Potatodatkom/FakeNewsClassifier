from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    headline = ""
    article = ""

    if request.method == "POST":

        headline = request.form.get("headline", "").strip()
        article = request.form.get("article", "").strip()

        combined_text = headline + " " + article

        if combined_text.strip():

            X = vectorizer.transform([combined_text])

            result = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]

            prediction = "FAKE" if result == 0 else "REAL"
            confidence = round(probabilities[result] * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        headline=headline,
        article=article
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)