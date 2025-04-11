from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from model import FakeNewsDetector
import uvicorn
import torch
import requests

app = FastAPI(
    title="Fake News Detector API",
    description="API for detecting fake news using BERT/RoBERTa",
    version="1.0.0"
)

class NewsItem(BaseModel):
    text: str

class NewsBatch(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    prediction: int  # 0 for fake, 1 for real
    confidence: float
    text: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

# Load the model
try:
    print("🔄 Attempting to load the model...")
    # Try multiple potential model locations
    model_locations = [
        "best_model_checkpoint",  # Directory name
        "best_model_checkpoint.pt",  # File name
        "saved_model",  # Default directory name
        "saved_model/best_model_checkpoint.pt",  # File in subdirectory
        "bestmodel_checkpoint.pt",  # Default file name
    ]
    
    detector = None
    for location in model_locations:
        try:
            print(f"Trying to load model from: {location}")
            detector = FakeNewsDetector.load_model(location)
            print(f"✓ Successfully loaded model from {location}")
            break
        except Exception as model_error:
            print(f"Could not load model from {location}: {str(model_error)}")
    
    if detector is None:
        print("❌ Failed to load model from any location. API will start but prediction endpoints will fail.")
except Exception as e:
    import traceback
    print(f"❌ Error loading model: {str(e)}")
    print(traceback.format_exc())
    detector = None

@app.get("/")
async def root():
    return {"message": "Welcome to the Fake News Detector API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(news_item: NewsItem):
    if detector is None:
        print("Error: Model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Verify that the model is correctly initialized
    if not hasattr(detector, 'model') or not hasattr(detector, 'tokenizer'):
        print("Error: Model or tokenizer not properly initialized")
        raise HTTPException(status_code=500, detail="Model not properly initialized")
    
    try:
        # Get prediction
        print(f"Attempting to predict text: {news_item.text[:100]}...")
        predictions = detector.predict([news_item.text])
        prediction = predictions[0]
        print(f"Raw prediction result: {prediction}")
        
        # Get confidence scores
        try:
            with torch.no_grad():
                print("Tokenizing text...")
                tokenized = detector.tokenize_texts([news_item.text])
                print("Running model...")
                outputs = detector.model(**tokenized)
                print("Calculating probabilities...")
                probabilities = torch.softmax(outputs.logits, dim=1)
                # Get confidence for the prediction
                confidence = probabilities[0][prediction].item()
                print(f"Confidence: {confidence}")
                
                # Print all probabilities for debugging
                print(f"All probabilities: {probabilities[0].tolist()}")
                print(f"Class 0 (Fake) prob: {probabilities[0][0].item()}")
                print(f"Class 1 (Real) prob: {probabilities[0][1].item()}")
        except Exception as inner_e:
            print(f"Error during confidence calculation: {str(inner_e)}")
            # If confidence calculation fails, set a default confidence
            confidence = 0.5
        
        # Map the prediction correctly
        # In the model: 0=fake, 1=real
        # Ensure the prediction is correctly represented
        result = {
            "prediction": int(prediction),  # 0=fake, 1=real
            "confidence": float(confidence),
            "text": str(news_item.text)
        }
        print("API Response:", result)
        
        return result
    except Exception as e:
        import traceback
        print(f"Error in predict_single: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(news_batch: NewsBatch):
    if detector is None:
        print("Error: Model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Verify that the model is correctly initialized
    if not hasattr(detector, 'model') or not hasattr(detector, 'tokenizer'):
        print("Error: Model or tokenizer not properly initialized")
        raise HTTPException(status_code=500, detail="Model not properly initialized")
    
    try:
        # Get predictions
        print(f"Processing batch with {len(news_batch.texts)} texts")
        predictions = detector.predict(news_batch.texts)
        print(f"Raw predictions: {predictions}")
        
        # Get confidence scores
        try:
            with torch.no_grad():
                print("Tokenizing batch...")
                outputs = detector.model(**detector.tokenize_texts(news_batch.texts))
                print("Calculating probabilities...")
                probabilities = torch.softmax(outputs.logits, dim=1)
                
                # Get confidence for each prediction
                confidences = []
                for i, pred in enumerate(predictions):
                    confidences.append(probabilities[i][pred].item())
                
                print(f"Confidences: {confidences}")
        except Exception as inner_e:
            print(f"Error during batch confidence calculation: {str(inner_e)}")
            # If confidence calculation fails, set default confidences
            confidences = [0.5] * len(predictions)
        
        # Map predictions correctly (0=fake, 1=real)
        response_items = [
            PredictionResponse(
                prediction=int(pred),  # 0=fake, 1=real
                confidence=float(conf),
                text=text
            )
            for pred, conf, text in zip(predictions, confidences, news_batch.texts)
        ]
        
        return BatchPredictionResponse(predictions=response_items)
    except Exception as e:
        import traceback
        print(f"Error in predict_batch: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing batch request: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 