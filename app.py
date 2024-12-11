from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = load_model('saved_model.h5')

# Define maximum sequence length (same as used during training)
SEQUENCE_LENGTH = 100

# Example training data for the tokenizer (replace this with your actual training data)
texts = [
    "This is a fake news article.",
    "This article is genuine."
]

# Initialize and train the tokenizer
tokenizer = Tokenizer(num_words=10000)  # Limit to 10,000 most common words
tokenizer.fit_on_texts(texts)

# Initialize FastAPI app
app = FastAPI()

# Input model for API
class PredictRequest(BaseModel):
    text: str

# Prediction endpoint
@app.post("/predict")
def predict(request: PredictRequest):
    try:
        # Preprocess the input text
        input_text = [request.text]
        sequences = tokenizer.texts_to_sequences(input_text)
        padded = pad_sequences(sequences, maxlen=SEQUENCE_LENGTH)

        # Make prediction
        prediction = model.predict(padded)
        label = "Real News" if prediction[0] >= 0.5 else "Fake News"
        confidence = float(prediction[0])

        # Return result as JSON
        return {
            "text": request.text,
            "prediction": label,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
def home():
    return {"message": "Fake News Detection API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)