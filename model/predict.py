import joblib
import sys

title = input("Enter PR title: ")
body = input("Enter PR body: ")
text = title + " " + body

model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')
X = vectorizer.transform([text])

label = model.predict(X)[0]
print(f"\n🔖 Predicted Label: {label}")
