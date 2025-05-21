# Standard imports
import importlib.util
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer

import image_caption.data.exploring_dataset as ds
from image_caption.config import logger

# Custom imports
from image_caption.data.data_preprocessing import clean_captions, preprocess_image
from image_caption.data.exploring_dataset import DatasetLoader
from image_caption.models.model_test import generate_caption
from image_caption.models.training_model import create_cnn_model, create_lstm_model, create_sequences, extract_features
from image_caption.visualizations.model_visualization import evaluate_model

importlib.reload(ds)

@hydra.main(version_base=None, config_path="../data/conf", config_name="config.yaml")
def main(cfg: DictConfig):
    # Load all captions mapping
    loader= DatasetLoader(cfg)
    all_captions_mapping = loader.load_captions()
    logger.info(f"Total images: {len(all_captions_mapping)}")

    # Selecting the first 100 images

    all_captions_mapping.pop('image', None)
    captions_mapping = {k: all_captions_mapping[k] for k in list(all_captions_mapping.keys())[:cfg['n_captions']]}

    # Build a list of all captions
    clean_captions(captions_mapping)

    all_captions = []
    for captions in captions_mapping.values():
        all_captions.extend(captions)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)

    vocab_size = len(tokenizer.word_index) + 1
    logger.info(f"Vocabulary Size: {vocab_size}")

    # Maximum length of a caption
    max_length = max(len(caption.split()) for caption in all_captions)
    logger.info(f"Maximum caption length: {max_length}")

    cnn_model = create_cnn_model()
    cnn_model.summary()

    img_path = cfg['example_img_path']

    image = preprocess_image(img_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    feature_vector = cnn_model.predict(image)
    logger.info(f"Feature vector shape: {feature_vector.shape}")
    feature_vector = cnn_model.predict(image)
    logger.info(f"Feature vector shape: {feature_vector.shape}")

    lstm_model = create_lstm_model(vocab_size, max_length)
    lstm_model.summary()

    lstm_model.compile(loss=cfg['loss'], optimizer=cfg['optimizer'])

    # Assume cnn_model, captions_mapping, and preprocess_image are already defined
    features = extract_features(
        cnn_model,
        cfg['img_path'],
        captions_mapping,
        preprocess_image
    )


    logger.info(f"Extracted features for {len(features)} images")

    # Prepare training data
    X1, X2, y = [], [], []

    for img_id, captions_list in captions_mapping.items():
        feature = features[img_id.split('.')[0]]
        xi1, xi2, yi = create_sequences(tokenizer, max_length, captions_list, feature, vocab_size)
        X1.extend(xi1)
        X2.extend(xi2)
        y.extend(yi)


    X1 = np.array(X1)
    X2 = np.array(X2)
    y = np.array(y)
    print(f"X1 shape: {X1.shape}, X2 shape: {X2.shape}, y shape: {y.shape}")

    # filepath = "../models/model-ep{epoch:03d}-loss{loss:.3f}.keras"
    filepath = f"models/{cfg['model_id']}.keras" # So we do not save way too many models

    # Might need some work here. I think it should overwrite the model each time the loss improves,
    # but for the checkpoint we'd want to save a model under a different name every n epochs (Josh)
    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor=cfg['loss_type'],       # or 'val_loss' if you're using validation_split
        verbose=cfg['verbose'],
        save_best_only=cfg['save_best_only'],  # saves only if loss improves
        save_weights_only=cfg['save_weights_only'],
        mode=cfg['mode']
    )
    # Fit model
    lstm_model.fit([X1, X2], y, epochs=cfg.epochs, batch_size=cfg['batch_size'],
                   callbacks=[checkpoint], verbose=cfg['verbose'])

    # Load an image
    image_id = list(captions_mapping.keys())[40].split('.')[0]  # Get the 45th image id
    photo = features[image_id]
    photo = np.expand_dims(photo, axis=0)

    # Generate caption
    caption = generate_caption(lstm_model, tokenizer, photo, max_length)

    # Correctly access the image path
    # Use raw string (r"...") and join paths properly
    base_image_dir = cfg['img_path']
    image_file = list(captions_mapping.keys())[40]
    image_path = os.path.join(base_image_dir, image_file)


    # Open the image using PIL
    img = Image.open(image_path)

    # Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Print the generated caption
    logger.info(f"Generated caption: {caption}")

    # Call the function to evaluate the model
    evaluate_model(lstm_model, captions_mapping, features, tokenizer, max_length)

if __name__ == "__main__":
    main()
