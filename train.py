import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load and clean the data
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Split into features (X) and target (y)
X = data['message']
y = data['label']

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text into TF-IDF numerical features
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
# fit_prior=False tells the model to treat spam and ham as equally likely
# up front, rather than being biased toward the majority class (ham).
model = MultinomialNB(fit_prior=False)
model.fit(X_train_vec, y_train)

# Evaluate on the test set
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.4f}")
print("\nDetailed report:")
print(classification_report(y_test, predictions))

# Save the trained model and vectorizer for later use
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("\nModel and vectorizer saved successfully.")
