import gradio as gr
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

from image_caption.models.model_test import generate_caption

# Load models and tokenizer
model = load_model("models/best_model.keras")
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("models/max_length.txt", "r") as f:
    max_length = int(f.read())

# Load InceptionV3
base_model = InceptionV3(weights='imagenet')
cnn_model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
compressor_input = Dense(256, activation='relu')(cnn_model.output)
compressor = Model(inputs=cnn_model.input, outputs=compressor_input)

def preprocess_for_inception(img: Image.Image):
    img = img.resize((299, 299))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

def caption_image(image):
    image_array = preprocess_for_inception(image)
    features = compressor.predict(image_array)
    caption = generate_caption(model, tokenizer, features, max_length)
    return caption

gr.Interface(fn=caption_image, inputs=gr.Image(type="pil"), outputs="text", title="Image Captioning App").launch()
