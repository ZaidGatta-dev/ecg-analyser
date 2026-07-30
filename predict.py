"""Runs inference on a single new ECG recording.

Usage: python predict.py path/to/record_name  (no .dat/.hea extension)
"""

import pickle
import sys

import numpy as np
import tensorflow as tf
import wfdb

import config
import preprocess


def predict_single_ecg(record_path: str) -> dict:
    signal, _ = wfdb.rdsamp(record_path)

    # Must match training-time preprocessing exactly, or predictions are meaningless.
    signal = preprocess.filter_signal(signal)
    signal_batch = preprocess.normalize_signals(np.expand_dims(signal, axis=0))

    model = tf.keras.models.load_model(config.MODEL_DIR + "ecg_cnn.keras")
    with open(config.PROCESSED_DIR + "label_binarizer.pkl", "rb") as f:
        label_binarizer = pickle.load(f)

    probs = model.predict(signal_batch)[0]
    for cls, prob in zip(label_binarizer.classes_, probs):
        flag = " <-- above threshold" if prob >= 0.5 else ""
        print(f"{cls:6s}: {prob:.3f}{flag}")

    return dict(zip(label_binarizer.classes_, probs))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/record_name")
        sys.exit(1)
    predict_single_ecg(sys.argv[1])
