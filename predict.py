import joblib

# Load the trained model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Try your own messages here
test_messages = [
    "Congratulations! You've won a free iPhone, click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account will be suspended. Verify your details immediately.",
    "Can you send me the notes from today's class?"
]

# Convert messages to the same TF-IDF format used in training
test_vec = vectorizer.transform(test_messages)

# Predict
predictions = model.predict(test_vec)

# Show results
for message, prediction in zip(test_messages, predictions):
    print(f"[{prediction.upper()}] {message}")
