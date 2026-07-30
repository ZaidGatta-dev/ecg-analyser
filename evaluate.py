"""Evaluates the trained model on the held-out PTB-XL test fold."""

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, classification_report

import config


def evaluate():
    model = tf.keras.models.load_model(config.MODEL_DIR + "ecg_cnn.keras")
    test_data = np.load(config.PROCESSED_DIR + "test_split.npz")
    X_test, y_test = test_data["X_test"], test_data["y_test"]

    y_pred_probs = model.predict(X_test)

    # Per-class ROC-AUC over plain accuracy: NORM dominates the label distribution,
    # so accuracy alone would reward a model that ignores rarer classes.
    aucs = roc_auc_score(y_test, y_pred_probs, average=None)
    for cls, auc in zip(config.DIAGNOSTIC_CLASSES, aucs):
        print(f"{cls:6s}: {auc:.3f}")
    print(f"Macro AUC: {roc_auc_score(y_test, y_pred_probs, average='macro'):.3f}")

    y_pred_labels = (y_pred_probs >= 0.5).astype(int)  # threshold chosen for illustration only
    print(classification_report(y_test, y_pred_labels,
                                 target_names=config.DIAGNOSTIC_CLASSES, zero_division=0))


if __name__ == "__main__":
    evaluate()
