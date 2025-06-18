import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
df = pd.read_csv('data/mock_prs.csv')
df['text'] = df['title'] + " " + df['body']

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, 'model/model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

print("✅ Model trained and saved.")
