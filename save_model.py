# save_model.py
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load and train model (your existing code)
df = pd.read_csv('data.csv')
df['statement'] = df['statement'].fillna('')

status_labels = df['status'].unique()


vectorizer = TfidfVectorizer(min_df=2, stop_words='english')
X = vectorizer.fit_transform(df['statement'])

le = LabelEncoder()
y = le.fit_transform(df['status'])

model = LogisticRegression()
model.fit(X, y)

# Save trained model and encoders
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
with open('status_labels.pkl', 'wb') as f:
    pickle.dump(status_labels, f)
