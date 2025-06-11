from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
import io
import pickle
import numpy as np
import tempfile
import asyncio
import time
import logging

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras import Input

from image_caption.models.model_test import generate_caption
from image_caption.data.data_preprocessing import preprocess_image  # (optional if custom)

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# === Load everything at startup ===
@app.on_event("startup")
def load_components():
    global model, tokenizer, max_length, cnn_model, compressor

    # Load captioning model
    model = load_model("models/best_model.keras")

    # Load tokenizer
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Load max_length
    with open("models/max_length.txt", "r") as f:
        max_length = int(f.read())

    # CNN Feature extractor
    base_model = InceptionV3(weights='imagenet')
    cnn_model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    # Dense 2048→256 compressor
    input_tensor = Input(shape=(2048,))
    output_tensor = Dense(256, activation='relu')(input_tensor)
    compressor = Model(inputs=input_tensor, outputs=output_tensor)

# === Image Preprocessing ===
def preprocess_for_inception(image_path):
    img = keras_image.load_img(image_path, target_size=(299, 299))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# === Prediction Endpoint ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()

    # File checks
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Invalid file type. Only images allowed.")

    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image file too large (>5MB)")

    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
            temp.write(contents)
            temp_path = temp.name

        # Preprocess and extract feature vector
        image = preprocess_for_inception(temp_path)
        raw_feature = await asyncio.to_thread(cnn_model.predict, image)
        feature_vector = await asyncio.to_thread(compressor.predict, raw_feature)

        # Generate caption
        caption = await asyncio.to_thread(generate_caption, model, tokenizer, feature_vector, max_length)
        latency = round((time.time() - start) * 1000)

        return JSONResponse(content={
            "caption": caption,
            "processing_time_ms": latency
        })

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not identify image format.")
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
