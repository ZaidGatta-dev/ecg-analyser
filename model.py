"""1D CNN architecture for multi-label ECG classification."""

from tensorflow import keras
from tensorflow.keras import layers

import config


def build_cnn_model(input_shape=(1000, config.NUM_LEADS),
                     num_classes=len(config.DIAGNOSTIC_CLASSES)) -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="ecg_signal")

    x = layers.Conv1D(32, kernel_size=7, activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, kernel_size=5, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # Sigmoid (not softmax): each class is an independent yes/no, more than one can fire.
    outputs = layers.Dense(num_classes, activation="sigmoid", name="diagnosis")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="ecg_cnn")


if __name__ == "__main__":
    build_cnn_model().summary()
