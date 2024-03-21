import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
import re
import tkinter as tk
from tkinter import Label, Entry, Button, StringVar, messagebox


import nltk
nltk.download('punkt')
nltk.download('stopwords')


df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None, names=["target", "ids", "date", "flag", "user", "text"])


df['target'] = df['target'].map({4: 'positive', 0: 'negative'})


stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def improved_preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', "", text)
    tokens = word_tokenize(text)
    tokens = [ps.stem(word) for word in tokens if word.isalnum()]
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['cleaned_text'] = df['text'].apply(improved_preprocess_text)


X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['target'], test_size=0.2, random_state=42)


tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


word2vec_model = Word2Vec(sentences=df['cleaned_text'].apply(lambda x: x.split()), vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.save("sentiment.model")


def text_to_word2vec(text):
    words = text.split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    return sum(word_vectors) / len(word_vectors) if word_vectors else [0] * 100

X_train_word2vec = X_train.apply(text_to_word2vec)
X_test_word2vec = X_test.apply(text_to_word2vec)


classifier_word2vec = LogisticRegression(max_iter=1000)
classifier_word2vec.fit(list(X_train_word2vec), y_train)
y_pred_word2vec = classifier_word2vec.predict(list(X_test_word2vec))
accuracy_word2vec = accuracy_score(y_test, y_pred_word2vec)


def analyze_sentiment():
    user_input = entry.get()
    if user_input:
        user_input_processed = improved_preprocess_text(user_input)
        
        
        user_input_word2vec = text_to_word2vec(user_input_processed)
        prediction_word2vec = classifier_word2vec.predict([user_input_word2vec])
        accuracy_label_word2vec.config(text=f"Word2Vec Predicted Sentiment: {prediction_word2vec[0]}, Accuracy: {accuracy_word2vec:.2f}")
    else:
        messagebox.showwarning("Input Error", "Please enter a sentiment text.")


root = tk.Tk()
root.title("Sentiment Analysis")

label = Label(root, text="Enter your sentiment:")
label.pack(pady=10)

entry = Entry(root, width=50)
entry.pack(pady=10)

analyze_button = Button(root, text="Analyze Sentiment", command=analyze_sentiment)
analyze_button.pack(pady=10)

accuracy_label_word2vec = Label(root, text="")
accuracy_label_word2vec.pack(pady=5)

root.mainloop()
