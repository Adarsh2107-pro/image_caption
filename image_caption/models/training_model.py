from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,Embedding,LSTM,concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


import os
import numpy as np

def create_cnn_model():
    # Input layer for images
    inputs = Input(shape=(224, 224, 3))

    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)

    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Flatten and Dense Layers
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = x
    model = Model(inputs, outputs, name='CNN_Model')
    return model


def create_sequences(tokenizer, max_length, captions_list, image_id, features,vocab_size):
    X1, X2, y = [], [], []
    for caption in captions_list:
        seq = tokenizer.texts_to_sequences([caption])[0]
        for i in range(1, len(seq)):
            # Input sequence
            in_seq = seq[:i]
            # Output word
            out_seq = seq[i]
            # Pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # Encode output word as one-hot vector
            out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
            # Store
            X1.append(features)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def create_lstm_model(vocab_size, max_length):
    
    # Image feature input
    inputs1 = Input(shape=(256,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # Sequence input
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    # Decoder (combine features)
    decoder1 = concatenate([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # Define the model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

def extract_features(cnn_model, images_directory, captions_mapping, preprocess_image):
    features = {}
    files = [x for x in captions_mapping.keys()]
    for img_name in files:
        img_path = os.path.join(images_directory, img_name)
        img = preprocess_image(img_path)
        img = np.expand_dims(img, axis=0)
        feature = cnn_model.predict(img, verbose=0)
        img_id = img_name.split('.')[0]
        features[img_id] = feature[0]
    return features