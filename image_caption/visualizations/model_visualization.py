from typing import Any

import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu

from image_caption.models.model_test import generate_caption


def evaluate_model(
        model: tf.keras.Model,
        captions_mapping: dict[str, list[str]],
        features: Any,
        tokenizer: Any,
        max_length: int,
    ) -> None:
    actual, predicted = [], []
    for img_id, captions_list in captions_mapping.items():
        y_pred = generate_caption(model, tokenizer, features[img_id.split('.')[0]].reshape(1, -1), max_length)
        references = [caption.split() for caption in captions_list]
        y_pred_tokens = y_pred.split()
        actual.append(references)
        predicted.append(y_pred_tokens)

    # Calculate BLEU score (BLEU-1)
    bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))

    # Print BLEU score
    print(f'BLEU-1: {bleu1:.4f}')
