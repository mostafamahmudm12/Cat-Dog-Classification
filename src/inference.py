import numpy as np
import io
import os
from typing import List, Union
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from .config import MODEL, IDX2LABEL
from .schemas import PredictionResponse, PredictionsResponses

IDX2LABEL = {
    0: "cats",
    1: "dogs"
}

class ImageClassifier:
    """Class for image classification operations"""

    def __init__(self, model=MODEL, idx2label=IDX2LABEL, target_size=(150, 150)):
        self.model = model
        self.idx2label = idx2label
        self.target_size = target_size

    def _preprocess_image(self, image: Union[str, bytes]) -> np.ndarray:
        """Preprocess image data to be fed into the model."""
        try:
            if isinstance(image, str):
                # Load image from file path
                image = tf.keras.utils.load_img(image, target_size=self.target_size)
            else:
                # Load image from raw bytes
                image = tf.keras.utils.load_img(io.BytesIO(image), target_size=self.target_size)

            # Convert to array and preprocess
            image_array = tf.keras.utils.img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)
            return image_array

        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")

    def predict_batch(self, images_data: List[Union[str, bytes]]) -> PredictionsResponses:
        """Make predictions on a batch of images."""

        if not images_data:
            raise ValueError("No images provided for prediction")

        basenames = []
        for i, img_data in enumerate(images_data):
            if isinstance(img_data, str) and os.path.exists(img_data):
                basenames.append(os.path.basename(img_data))
            else:
                basenames.append(f"image_{i}")

        # Preprocess all images
        preprocessed_images = []
        for img_data in images_data:
            preprocessed_images.append(self._preprocess_image(img_data))

        # Stack all into a batch: shape (N, H, W, 3)
        batch_images = np.vstack(preprocessed_images)

        # Make predictions
        predictions = self.model.predict(batch_images)

        # Get predicted class indices and confidence scores
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = [float(predictions[i][idx] * 100) for i, idx in enumerate(predicted_classes)]
        predicted_class_names = [self.idx2label[idx] for idx in predicted_classes]

        # Create list of PredictionResponse objects
        prediction_responses = [
            PredictionResponse(
                base_name=basename,
                class_index=int(idx),
                class_name=name,
                confidence=round(score, 2),
            )
            for basename, idx, name, score in zip(
                basenames, predicted_classes, predicted_class_names, confidence_scores
            )
        ]

        return PredictionsResponses(predictions=prediction_responses)
