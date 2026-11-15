from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import io
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Model lives in <repo>/models/ (not committed - see models/README.md)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "pneumonia_detection_model.keras")

app = FastAPI(title="PneumoScan - Chest X-Ray Classifier")

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Load model on startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image to EXACTLY match training pipeline.

    Training used ImageDataGenerator with:
      - rescale=1/255
      - samplewise_center=True       → subtract per-image mean
      - samplewise_std_normalization=True  → divide by per-image std

    Skipping any of these 3 steps causes wrong predictions every time.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))

    # Step 1: to float32
    img_array = np.array(img, dtype=np.float32)

    # Step 2: rescale to [0, 1]
    img_array = img_array / 255.0

    # Step 3: samplewise_center — subtract per-image mean
    img_array -= np.mean(img_array)

    # Step 4: samplewise_std_normalization — divide by per-image std
    std = np.std(img_array)
    if std == 0:
        std = 1e-7
    img_array /= std

    # Step 5: add batch dimension → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model

    if model is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Model not loaded. Place pneumonia_detection_model.keras in the repo's models/ folder (see models/README.md).",
                "status": "error"
            }
        )

    # Validate file type
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": "Please upload an image file.", "status": "error"}
        )

    try:
        image_bytes = await file.read()
        img_array = preprocess_image(image_bytes)

        prediction = model.predict(img_array)

        # Handle both binary and sigmoid output
        if prediction.shape[-1] == 1:
            # Binary classification: sigmoid output
            prob_pneumonia = float(prediction[0][0])
            prob_normal = 1 - prob_pneumonia
        else:
            # Softmax output: [normal, pneumonia]
            prob_normal = float(prediction[0][0])
            prob_pneumonia = float(prediction[0][1])

        is_pneumonia = prob_pneumonia > 0.5
        confidence = max(prob_pneumonia, prob_normal) * 100

        return JSONResponse(content={
            "status": "success",
            "prediction": "PNEUMONIA" if is_pneumonia else "NORMAL",
            "confidence": round(confidence, 2),
            "prob_pneumonia": round(prob_pneumonia * 100, 2),
            "prob_normal": round(prob_normal * 100, 2),
            "is_pneumonia": is_pneumonia
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Prediction failed: {str(e)}", "status": "error"}
        )

@app.get("/health")
async def health():
    return {
        "status": "running",
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
