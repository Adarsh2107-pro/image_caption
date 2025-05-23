import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from image_caption.models.model_test import generate_caption

# --- Paths ---
model_path = "../models/best_model.keras"
tokenizer_path = "../models/tokenizer.pkl"
features_path = "../models/features.pkl"
max_length_path = "../models/max_length.txt"

# --- Load Model ---
print(f"Loading model from: {model_path}")
model = load_model(model_path)

# --- Load Tokenizer ---
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)
print("Tokenizer loaded.")

# --- Load Features ---
with open(features_path, "rb") as f:
    features = pickle.load(f)
print("Features loaded.")

# --- Load max_length ---
with open(max_length_path, "r") as f:
    max_length = int(f.read())
print(f"max_length loaded: {max_length}")

# --- Select an image ---
image_id = list(features.keys())[40]
photo = features[image_id]
photo = np.expand_dims(photo, axis=0)

# --- Generate Caption ---
caption = generate_caption(model, tokenizer, photo, max_length)
print("Generated caption:", caption)
