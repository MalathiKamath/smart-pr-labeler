import gradio as gr
import joblib

model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def predict_label(title, body):
    text = title + " " + body
    X = vectorizer.transform([text])
    label = model.predict(X)[0]
    proba = model.predict_proba(X).max()
    return f"{label} ({proba:.2%} confidence)"

demo = gr.Interface(
    fn=predict_label,
    inputs=[
        gr.Textbox(label="PR Title"),
        gr.Textbox(label="PR Body")
    ],
    outputs=gr.Textbox(label="Predicted Label"),
    title="🔖 Smart PR Labeler",
    description="Enter a GitHub PR title and body to get an AI-suggested label"
)

if __name__ == "__main__":
    demo.launch()
