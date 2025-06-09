import pickle

import numpy as np
import pytest
from tensorflow.keras.models import load_model

from image_caption.models.model_test import generate_caption

# Change as needed for loading objects
model_path = "./models/best_model.keras"
tokenizer_path = "./models/tokenizer.pkl"
features_path = "./models/features.pkl"
max_length_path = "./models/max_length.txt"

@pytest.fixture
def model():
    model = load_model(model_path)
    return model

def test_inference(model):
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(features_path, "rb") as f:
        features = pickle.load(f)

    with open(max_length_path, "r") as f:
        max_length = int(f.read())

    image_id = list(features.keys())[40]
    photo = features[image_id]
    photo = np.expand_dims(photo, axis=0)

    caption = generate_caption(model, tokenizer, photo, max_length)

    # A string caption should be generated
    assert isinstance(caption, str)
