# SMS Spam Classifier

A machine learning model that classifies SMS messages as **spam** or **ham** (not spam), built using the classic SMS Spam Collection Dataset.

## 🚀 Features
- Cleans and prepares real-world SMS message data
- Converts text into numerical features using **TF-IDF**
- Classifies messages using a **Naive Bayes** classifier
- Achieves **96% accuracy** on unseen test data
- Includes a script to test the model on custom, user-written messages
- Trained model is saved and reusable without retraining

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** pandas, scikit-learn, joblib

## 📂 Project Structure
- `explore.py` — initial exploration of the raw dataset
- `clean.py` — cleans the data and checks class balance
- `train.py` — trains the TF-IDF + Naive Bayes model and evaluates it
- `predict.py` — loads the trained model and tests it on new messages
- `spam.csv` — the dataset (SMS Spam Collection Dataset)
- `spam_model.pkl` / `vectorizer.pkl` — the saved, trained model and vectorizer

## ⚙️ How to Run

1. Install dependencies:
   ```bash
   pip install pandas scikit-learn joblib
   ```

2. Train the model:
   ```bash
   python train.py
   ```

3. Test it on your own messages by editing the `test_messages` list in `predict.py`, then run:
   ```bash
   python predict.py
   ```

## 📊 Results
The model achieves **96% accuracy** on the test set, with strong precision and recall for both spam and ham classes after correcting for class imbalance (see below).

## 📌 What I Learned
- Cleaning and preparing real-world text data for machine learning
- Converting text into numerical features using TF-IDF vectorization
- Training and evaluating a Naive Bayes classifier
- **Handling class imbalance:** the dataset is ~86% ham and ~14% spam. By default, Naive Bayes leans toward the majority class (ham) unless the evidence for spam is very strong, which caused obvious spam messages to be misclassified during testing. Setting `fit_prior=False` corrected this by treating both classes as equally likely up front, significantly improving spam detection.
- Saving and reusing trained models with `joblib`, rather than retraining every time

## 📄 License
This project is licensed under the MIT License.
