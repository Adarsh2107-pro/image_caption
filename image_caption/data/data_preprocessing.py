import string

import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def preprocess_image(image_path: str) -> 'np.ndarray':
    # Load the image
    img = load_img(image_path, target_size=(224, 224))  # ResNet50 expects 224x224 images

    # Convert the image to a numpy array
    img = img_to_array(img)

    # If the image has 4 channels (RGBA), we discard the alpha channel
    if img.shape[-1] == 4:
        img = img[..., :3]

    # Preprocess the image as per ResNet50's requirements
    img = preprocess_input(img)

    return img


def clean_captions(captions_mapping: dict[str, list[str]]) -> None:
    table = str.maketrans('', '', string.punctuation)
    for img_id, captions in captions_mapping.items():
        for i, caption in enumerate(captions):
            # Tokenize
            caption = caption.lower()
            caption = caption.translate(table)
            caption = caption.strip()
            caption = ' '.join([word for word in caption.split() if len(word) > 1])
            # Add start and end tokens
            caption = 'startseq ' + caption + ' endseq'
            captions[i] = caption
