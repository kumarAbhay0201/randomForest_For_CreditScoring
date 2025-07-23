# 💳 Credit Risk Predictor

#this app is live on "https://creditscoringtoolbyabhaykumar.streamlit.app/"

Welcome to the **Credit Risk Predictor** — a simple, interactive web app built using **Streamlit** that helps estimate whether a person is a **Good** or **Bad** credit risk based on various financial attributes.

> ⚠️ *Disclaimer: This app is created for educational purposes only using a well-known public dataset and is not meant for real-world credit assessment.*

---

## 🔍 About the Project

This project demonstrates how machine learning models can be integrated into web applications to make predictions in real-time. The model used here is a **Random Forest Classifier** trained on the [German Credit Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).

The goal is to help understand how various features (like age, employment status, savings, and credit history) affect someone's creditworthiness.

---

## 🧠 Machine Learning Behind the Scenes

- 📊 **Dataset**: German Credit Data (numeric format)
- 🧪 **Model Used**: Random Forest Classifier
- 🔧 **Metrics**: Accuracy, Confusion Matrix, Classification Report
- 🧼 **Preprocessing**: Feature encoding, train-test split
- 🎯 **Output**: Binary prediction — "Good" or "Bad" credit risk

---

## 🚀 Features

- ✅ Easy-to-use UI for entering customer financial data
- 📈 Real-time prediction on button click
- 📊 Visualization of the **Top 10 most important features**
- 🧠 Model trained and saved using `joblib`
- 🔐 Input validation for dropdowns to prevent errors

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Seaborn / Matplotlib
- Joblib

---

## 📦 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/credit-risk-predictor.git
   cd credit-risk-predictor

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt

  3.Run the app:
  ```bash
    streamlit run app.py
