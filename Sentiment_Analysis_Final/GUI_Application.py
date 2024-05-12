import tkinter as tk
from tkinter import ttk
import speech_recognition as sr
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('CSV/preprocessed_dataset.csv')


class SentimentAnalyzerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Sentiment Analyzer")
        
        self.load_data()
        self.vectorize_data()
        self.train_model()
        
        self.create_widgets()

    def load_data(self):
        self.df_text = df['clean_sentences']
        self.df_react = df['category']

    def vectorize_data(self):
        self.vectorizer = CountVectorizer()
        self.X = self.vectorizer.fit_transform(self.df_text)
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.df_react)

    def train_model(self):
        self.model = MultinomialNB()
        self.model.fit(self.X, self.y)

    def recognize_speech(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("I am listening...:")
            audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            self.input_text.delete(1.0, tk.END)
            self.input_text.insert(tk.END, text)
            self.analyze_sentiment()
        except sr.UnknownValueError:
            print("Could not hear")
        except sr.RequestError as e:
            print("Could not request; {0}".format(e))

    def analyze_sentiment(self):
        text = self.input_text.get(1.0, tk.END).strip()
        if text == "":
            self.result_label.config(text="Please speak or text.")
            return
        text_vect = self.vectorizer.transform([text])
        sentiment = self.label_encoder.inverse_transform(self.model.predict(text_vect))[0]
        self.result_label.config(text=f"Sentiment: {sentiment}")

    def create_widgets(self):
        self.input_label = ttk.Label(self.master, text="Input:")
        self.input_label.grid(row=0, column=0, padx=10, pady=10)

        self.input_text = tk.Text(self.master, height=5, width=40)
        self.input_text.grid(row=0, column=1, padx=10, pady=10)

        self.analyze_button = ttk.Button(self.master, text="Analyze Text", command=self.analyze_sentiment)
        self.analyze_button.grid(row=0, column=2, padx=10, pady=10)

        self.speech_button = ttk.Button(self.master, text="Use Speech", command=self.recognize_speech)
        self.speech_button.grid(row=0, column=3, padx=10, pady=10)

        self.result_label = ttk.Label(self.master, text="")
        self.result_label.grid(row=1, columnspan=4, padx=10, pady=10)

def main():
    root = tk.Tk()
    app = SentimentAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
