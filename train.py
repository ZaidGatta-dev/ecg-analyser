"""Trains the ECG CNN end-to-end: load -> preprocess -> train -> save."""

import os
import pickle

import numpy as np
import tensorflow as tf

import config
import data_loader
import preprocess
from model import build_cnn_model


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    metadata = data_loader.load_metadata()
    diag_map = data_loader.load_diagnostic_mapping()
    metadata["diagnostic_superclass"] = metadata.scp_codes.apply(
        lambda codes: data_loader.scp_codes_to_superclass(codes, diag_map)
    )
    metadata = metadata[metadata.diagnostic_superclass.apply(len) > 0]

    signals = data_loader.load_raw_signals(metadata)
    signals = preprocess.filter_all_signals(signals)
    signals = preprocess.normalize_signals(signals)

    labels, label_binarizer = preprocess.encode_labels(metadata.diagnostic_superclass)
    with open(config.PROCESSED_DIR + "label_binarizer.pkl", "wb") as f:
        pickle.dump(label_binarizer, f)

    splits = preprocess.split_by_fold(metadata, signals, labels)
    print(f"train={splits['X_train'].shape[0]} val={splits['X_val'].shape[0]} "
          f"test={splits['X_test'].shape[0]}")

    model = build_cnn_model(input_shape=splits["X_train"].shape[1:])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )

    # AUC-monitored early stopping: accuracy is misleading on this imbalanced label set.
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=5, restore_best_weights=True
    )

    model.fit(
        splits["X_train"], splits["y_train"],
        validation_data=(splits["X_val"], splits["y_val"]),
        epochs=50, batch_size=64, callbacks=[early_stop],
    )

    model.save(config.MODEL_DIR + "ecg_cnn.keras")
    np.savez(config.PROCESSED_DIR + "test_split.npz",
              X_test=splits["X_test"], y_test=splits["y_test"])


if __name__ == "__main__":
    main()
