import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from pretraitement import pipelineData
import tensorflow as tf


def evaluate():
    model = tf.keras.models.load_model(
        "models/googlenet_best.keras"
    )

    y_true = []
    y_pred = []

    _, test_ds, _, class_names = pipelineData()

    for images, labels in test_ds:
        probabilities = model.predict(images, verbose=0)
        predictions = np.argmax(probabilities, axis=1)

        if len(labels.shape) > 1 and labels.shape[-1] > 1:
            labels = np.argmax(labels.numpy(), axis=1)
        else:
            labels = labels.numpy().reshape(-1)

        y_true.extend(labels.astype(int))
        y_pred.extend(predictions.astype(int))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy : {accuracy:.4f}")

    print("\nRapport de classification :")
    print(classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0
    ))


if __name__ == "__main__":
    evaluate()