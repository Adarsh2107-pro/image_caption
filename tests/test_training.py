import numpy as np
import pytest
import yaml

from image_caption.data.data_preprocessing import preprocess_image
from image_caption.data.exploring_dataset import DatasetLoader
from image_caption.models.training_model import create_cnn_model, create_lstm_model, extract_features

# Define the dataset path
with open("config/conf/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
    # Shorten train time
    cfg['epochs'] = 1

@pytest.fixture
def data():
    return DatasetLoader(cfg)

@pytest.fixture
def cnn_model():
    return create_cnn_model()

@pytest.fixture
def lstm_model():
    vocab_size = 814
    max_length = 27
    return create_lstm_model(vocab_size, max_length)

def test_cnn_features_extraction(cnn_model):

    # Extract features from a single image
    features = extract_features(
        cnn_model,
        '',
        {cfg['example_img_path']: ["This is a fake caption for testing."]},
        preprocess_image
    )

    feature_shape = next(iter(features.values())).shape
    expected_shape = (256,)

    assert feature_shape == expected_shape

def test_lstm_training(lstm_model):
    lstm_model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Dummy data
    batch_size = 1
    vocab_size = 814
    max_length = 27
    X1 = np.random.rand(batch_size, 256).astype(np.float32)
    X2 = np.random.randint(1, vocab_size, size=(batch_size, max_length))
    y = np.zeros((batch_size, vocab_size))
    y[np.arange(batch_size), np.random.randint(0, vocab_size, batch_size)] = 1

    # Fit model (should not cause any errors)
    results = lstm_model.fit([X1, X2], y, epochs=1, batch_size=batch_size, verbose=0)
    assert results.history['loss']
