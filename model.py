import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('data.csv')

df['statement'] = df['statement'].fillna('')  # Replace NaN with empty string

# Prepare features (X) and target (y)
vectorizer = TfidfVectorizer(min_df=2, stop_words='english')
X = vectorizer.fit_transform(df['statement'])

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(df['status'])

# Train model
model = LogisticRegression()
model.fit(X, y)

# Function to make predictions
def predict_status(text):
    # Transform text to TF-IDF features
    text_features = vectorizer.transform([text])
    # Make prediction and convert back to label
    prediction = model.predict(text_features)
    rounded_pred = int(np.round(prediction[0]))
    return le.inverse_transform([rounded_pred])[0]

# Example usage
test_text = "I need help."
predicted_status = predict_status(test_text)
print(f"Predicted status: {predicted_status}")

# Print model accuracy
train_score = model.score(X, y)
print(f"Training accuracy: {train_score:.2f}")
