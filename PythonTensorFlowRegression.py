#!/usr/bin/env python3
"""
PythonTensorFlowRegression.py

Architecture: 2 → 6 → 2 → 2 → 8 → 2 → 2 → 6 → 2
Output: 2 continuous values (regression)

Target y:
  shape: (N, 2)
  dtype: float32
  values: any real numbers

Loss: MeanSquaredError (MSE)
"""

import numpy as np
import tensorflow as tf


def build_model(lr: float = 1e-3) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(2,), name="x")

    x = tf.keras.layers.Dense(6, activation="relu", name="dense_2_6")(inputs)
    x = tf.keras.layers.Dense(2, activation="relu", name="dense_6_2")(x)
    x = tf.keras.layers.Dense(2, activation="relu", name="dense_2_2_a")(x)
    x = tf.keras.layers.Dense(8, activation="relu", name="dense_2_8")(x)
    x = tf.keras.layers.Dense(2, activation="relu", name="dense_8_2")(x)
    x = tf.keras.layers.Dense(2, activation="relu", name="dense_2_2_b")(x)
    x = tf.keras.layers.Dense(6, activation="relu", name="dense_2_6_b")(x)

    outputs = tf.keras.layers.Dense(2, activation=None, name="y")(x)  # linear output

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="YoungNet_regression")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def make_dummy_data(n: int = 512, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)).astype(np.float32)
    y = rng.normal(size=(n, 2)).astype(np.float32)
    return x, y


def main():
    tf.get_logger().setLevel("ERROR")

    model = build_model()
    model.summary()

    x, y = make_dummy_data()
    model.fit(x, y, epochs=5, batch_size=64, validation_split=0.2)

    x_test = np.array([[0.1, -0.2], [1.0, 0.5]], dtype=np.float32)
    print("\nPredictions:")
    print(model.predict(x_test))


if __name__ == "__main__":
    main()
