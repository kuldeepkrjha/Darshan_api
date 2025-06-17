from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os

app = FastAPI(title="Fundus Image Disease Detection")

# Allow CORS (allow frontend to call the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify ["http://localhost:3000"] for Next.js
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
MODEL_PATH = "Kaggle_1000/fundus_efficientnetv2b0_final.h5"

CLASS_NAMES = [
    'Bietti Crystalline Dystrophy', 'Blur Fundus With Suspected Pdiabetic Retinopathy',
    'Blur Fundus Without Pdiabetic Retinopathy', 'Branch Retinal Vein Occlusion',
    'Central Retinal Vein Occlusion', 'Central Serous Chorioretinopathy',
    'Chorioretinal Atrophy-Coloboma', 'Congenital Disc Abnormality', 'Cotton-Wool Spots',
    'Diabetic Retinopathy1', 'Diabetic Retinopathy2', 'Diabetic Retinopathy3',
    'Disc Swelling And Elevation', 'Dragged Disc', 'Epiretinal Membrane', 'Fibrosis',
    'Fundus Neoplasm', 'Large Optic Cup', 'Laser Spots', 'Macular Hole', 'Maculopathy',
    'Massive Hard Exudates', 'Myelinated Nerve Fiber', 'Normal', 'Optic Atrophy',
    'Pathological Myopia', 'Peripheral Retinal Degeneration And Break', 'Possible Glaucoma',
    'Preretinal Hemorrhage', 'Retinal Artery Occlusion', 'Retinitis Pigmentosa',
    'Rhegmatogenous Retinal Detachment', 'Severe Hypertensive Retinopathy',
    'Silicon Oil In Eye', 'Tessellated Fundus', 'Vessel Tortuosity', 'Vitreous Particles',
    'Vogt-Koyanagi-Harada Disease Disease', 'Yellow-White Spots-Flecks'
]

# Global model
model = None

@app.on_event("startup")
def load_model_on_startup():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"❌ Model not found at path: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")

# Image preprocessing
def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((380, 380))  # Resize to match input shape
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Root health check
@app.get("/")
def root():
    return {"status": "healthy", "framework": "Keras / TensorFlow"}

# Predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        image_bytes = await file.read()
        img_tensor = preprocess_image(image_bytes)
        predictions = model.predict(img_tensor)[0]

        pred_idx = int(np.argmax(predictions))
        confidence = float(predictions[pred_idx])
        predicted_class = CLASS_NAMES[pred_idx]

        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "class_index": pred_idx,
            "probabilities": predictions.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
