import gradio as gr
import pickle
from fastapi import FastAPI
import os

# Load the model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), "hate_speech_model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Model files not found! Make sure 'hate_speech_model.pkl' and 'vectorizer.pkl' are uploaded.")

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Prediction function
def predict_hate_speech(tweet):
    tweet_vect = vectorizer.transform([tweet])
    prediction = model.predict(tweet_vect)
    return "💔 Hate Speech" if prediction == 1 else "💖 Non-Hate Speech"

# FastAPI setup
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hate Speech Detection API is running!"}

# Gradio UI
iface = gr.Interface(
    fn=predict_hate_speech,
    inputs=gr.Textbox(placeholder="Type your tweet here...", label="📝 Enter Tweet"),
    outputs=gr.Label(label="🔍 Prediction"),
    title="🌟 Hate Speech Detection System 🌟",
    description="This tool analyzes tweets to determine if they contain hate speech.",
    examples=[["I love the world!"], ["I hate you!"], ["You are amazing!"]],
    live=True
)

app = gr.mount_gradio_app(app, iface, path="/")
