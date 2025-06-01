import numpy as np
import pytest

from image_caption.models.training_model import create_cnn_model, create_lstm_model


@pytest.fixture
def cnn_model():
    return create_cnn_model()

@pytest.fixture
def lstm_model():
    vocab_size = 814
    max_length = 27
    return create_lstm_model(vocab_size, max_length)

def test_cnn_model_output_shape(cnn_model):
    # Dummy input shape
    input_shape = (1, 224, 224, 3)
    expected_output_shape = (1, 256)

    input = np.random.rand(*input_shape)
    output = cnn_model.predict(input)

    assert output.shape == expected_output_shape

def test_lstm_model(lstm_model):
    # Dummy inputs
    vocab_size = 814
    max_length = 27
    img_features = np.random.rand(1, 256) # batch size of 1
    seq_input = np.random.randint(1, vocab_size, size=(1, max_length))

    output = lstm_model.predict([img_features, seq_input])

    expected_output_shape = (1, vocab_size)
    assert output.shape == expected_output_shape
