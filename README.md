**TruthScan — Fake News Classifier**



TruthScan is a machine learning web application that classifies news articles as potentially fake or real based on text patterns learned from a labeled dataset.



**Technologies Used**

\- Python

\- Flask

\- Pandas

\- Scikit-learn

\- TF-IDF Vectorization

\- Logistic Regression

\- HTML and CSS

\- Jupyter Notebook



**Dataset**

This project uses the Fake and Real News Dataset by Clément Bisaillon.



Dataset source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset



Download the dataset and place `Fake.csv` and `True.csv` inside the `dataset/` folder before training the model.



**How to Run**



1\. Install the required packages:



 `pip install -r requirements.txt`



2\. Make sure the dataset files are in the `dataset/` folder.



3\. Train the model if the saved model files are missing:



 `python train.py`



4\. Start the web application:



 `python app.py`



5\. Open `http://127.0.0.1:5000` in your browser.



**Important Note**

The classifier identifies patterns learned from its training data. Its prediction is not a guarantee that a news article is true or false. Always verify news using reliable sources.

