from github import Github
import joblib
import argparse

# Load model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def classify(text):
    X = vectorizer.transform([text])
    label = model.predict(X)[0]
    proba = model.predict_proba(X).max()
    return label, proba

def fetch_and_label_prs(token, repo_name, max_prs=5):
    g = Github(token)
    repo = g.get_repo(repo_name)
    pulls = repo.get_pulls(state='all', sort='created')[:max_prs]

    for pr in pulls:
        text = (pr.title or "") + " " + (pr.body or "")
        label, confidence = classify(text)
        print(f"[PR #{pr.number}] {pr.title} - > Label: {label} ({confidence:.2%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--repo", required=True)  # e.g., "username/repo"
    parser.add_argument("--max", type=int, default=5)
    args = parser.parse_args()

    fetch_and_label_prs(args.token, args.repo, args.max)
