import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tkinter import *
from tkinter import messagebox
import os
import csv
# Setup
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
# Load and Clean Dataset
df = pd.read_csv("news_dataset.csv")
df = df.dropna()
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    stop_words = stopwords.words('english')
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text
df['clean_text'] = df['text'].apply(clean_text)
# Prepare Data
X = df['clean_text']
y = df['label'].map({'real': 0, 'fake': 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = LogisticRegression()
model.fit(X_train_vec, y_train)
# GUI Functions
is_dark = False
def predict_news():
    input_text = entry.get("1.0", END).strip()
    if not input_text:
        output_label.config(text="Please enter some news text.")
        return
    cleaned = clean_text(input_text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0][prediction]
    result = "🛑 Fake News" if prediction == 1 else "✅ Real News"
    confidence_percent = round(probability * 100, 2)
    output = f"{result} ({confidence_percent}% confidence)"
    output_label.config(text=output)
    # Append to CSV
    with open("predictions_log.csv", mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(['Input Text', 'Prediction', 'Confidence (%)'])  # Write header if new file
        writer.writerow([input_text, result, confidence_percent])
def clear_text():
    entry.delete("1.0", END)
    output_label.config(text="")
def toggle_dark_mode():
    global is_dark
    is_dark = not is_dark
    bg = "#2E2E2E" if is_dark else "white"
    fg = "white" if is_dark else "black"
    btn_bg = "#444" if is_dark else "lightblue"
    clear_bg = "#666" if is_dark else "lightgrey"
    window.config(bg=bg)
    title.config(bg=bg, fg=fg)
    entry.config(bg=fg, fg=bg)
    output_label.config(bg=bg, fg=fg)
    footer.config(bg=bg, fg="lightgrey" if is_dark else "grey")
    predict_button.config(bg=btn_bg, fg=fg)
    clear_button.config(bg=clear_bg, fg=fg)
    dark_button.config(bg=clear_bg, fg=fg)
# GUI Setup
window = Tk()
window.title("Fake News Detector")
window.geometry("600x520")
window.config(bg="white")
# Title
title = Label(window, text="📰 Fake News Detector", font=("Arial", 20, "bold"), bg="white", fg="darkblue")
title.pack(pady=15)
# Text Input
entry = Text(window, height=10, width=65, font=("Arial", 12), bg="white", fg="black")
entry.pack(pady=10)
# Button Frame
button_frame = Frame(window, bg="white")
button_frame.pack(pady=5)
predict_button = Button(button_frame, text="Check News", command=predict_news, font=("Arial", 12), bg="lightblue", width=15)
predict_button.grid(row=0, column=0, padx=10)
clear_button = Button(button_frame, text="Clear", command=clear_text, font=("Arial", 12), bg="lightgrey", width=15)
clear_button.grid(row=0, column=1, padx=10)
dark_button = Button(button_frame, text="🌙 Toggle Dark Mode", command=toggle_dark_mode, font=("Arial", 12), bg="lightgrey", width=20)
dark_button.grid(row=0, column=2, padx=10)
# Output
output_label = Label(window, text="", font=("Arial", 14, "bold"), bg="white", fg="black")
output_label.pack(pady=20)
# Footer
footer = Label(window, text="Built with ❤️ using Python, ML & Tkinter", font=("Arial", 10), bg="white", fg="grey")
footer.pack(side=BOTTOM, pady=10)
# Run App
window.mainloop()
