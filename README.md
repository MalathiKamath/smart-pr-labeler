
# Smart PR Labeler 🤖

Automatically classify GitHub pull requests into labels using AI.

Smart PR Labeler is an AI-powered assistant that automatically classifies GitHub pull requests based on their title and body. It suggests labels like bug, feature, refactor, and documentation — helping DevOps teams reduce manual triage, maintain consistent workflows, and speed up code reviews.

Built using NLP and machine learning (TF-IDF + Logistic Regression), this tool transforms noisy, unlabeled PRs into actionable, structured entries in your DevOps pipeline.

Whether you're a solo developer or a team processing dozens of PRs per day, Smart PR Labeler helps you shift labeling left — right at the point of code submission.

✨ Why It Matters

📌 Eliminates manual labeling of pull requests

🚦 Improves CI/CD routing, triage, and automation

🧪 Lays the foundation for future GitHub bot automation

🧠 Built with real AI, not rule-based shortcuts

⚡ It's not just another ML demo — it's a working DevOps utility.

🛠️ Built With

Python, scikit-learn, TF-IDF vectorizer

Gradio for interactive UI

Joblib for model serialization

Mock GitHub PR data for rapid prototyping

## Labels Supported
- bug
- feature
- refactor
- documentation

## Tech Stack
- Python
- scikit-learn (baseline)
- Gradio (UI)
- pandas
  
## How to Run
python model/train_model.py
python model/predict.py
pip install -r requirements.txt
python app/interface.py

## Demo
Coming soon...

## Author
Malathi Kamath | AI + DevOps Builder

