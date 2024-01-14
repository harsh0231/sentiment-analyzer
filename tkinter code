import tkinter as tk
from tkinter import Label, Entry, Button, messagebox
import pyttsx3
from ttkthemes import ThemedStyle
from gensim.models import Word2Vec

word2vec_model = Word2Vec.load("sentiment.model")

def text_to_word2vec(text):
    words = text.split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    return sum(word_vectors) / len(word_vectors) if word_vectors else [0] * 100

def analyze_sentiment():
    user_input = entry.get()
    if user_input:
        user_input_processed = improved_preprocess_text(user_input)
        user_input_word2vec = text_to_word2vec(user_input_processed)
        prediction_word2vec = classifier_word2vec.predict([user_input_word2vec])
        speak_result(prediction_word2vec[0])
        accuracy_label_word2vec.config(text=f"Word2Vec Predicted Sentiment: {prediction_word2vec[0]}")
    else:
        messagebox.showwarning("Input Error", "Please enter a sentiment text.")

def speak_welcome():
    welcome_message = "Welcome to this interface. Type your thoughts here, and I will analyze your sentiment."
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.say(welcome_message)
    engine.runAndWait()

def speak_result(sentiment):
    result_message = "Your sentiment is positive. Have a great day!" if sentiment == 'positive' else "Your sentiment is negative. Cheer up!"
    engine = pyttsx3.init()
    engine.say(result_message)
    engine.runAndWait()

root = tk.Tk()
root.title("Sentiment Analysis")
style = ThemedStyle(root)
style.set_theme("plastik")

label = Label(root, text="Enter your sentiment:")
label.pack(pady=10)

entry = Entry(root, width=50)
entry.pack(pady=10)

analyze_button = Button(root, text="Analyze Sentiment", command=analyze_sentiment)
analyze_button.pack(pady=10)

accuracy_label_word2vec = Label(root, text="")
accuracy_label_word2vec.pack(pady=5)

speak_welcome()
root.mainloop()
