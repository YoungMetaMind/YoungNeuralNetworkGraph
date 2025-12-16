#!/usr/bin/env python3
"""
PythonTensorFlowSoftmax.py

Architecture: 2 → 6 → 2 → 2 → 8 → 2 → 2 → 6 → 2
Output: 2-class softmax (mutually exclusive classification)

Target y:
  shape: (N,)
  dtype: int32 / int64
  values: 0 or 1
  NOTE: this uses SparseCategoricalCrossentropy so targets are NOT one-hot.

Loss: SparseCategoricalCrossentropy
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

    outputs = tf.keras.layers.Dense(2, activation="softmax", name="y")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="YoungNet_softmax2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model


def make_dummy_data(n: int = 512, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)).astype(np.float32)
    y = rng.integers(0, 2, size=(n,), dtype=np.int32)
    return x, y


def main():
    tf.get_logger().setLevel("ERROR")

    model = build_model()
    model.summary()

    x, y = make_dummy_data()
    model.fit(x, y, epochs=5, batch_size=64, validation_split=0.2)

    x_test = np.array([[0.1, -0.2], [1.0, 0.5]], dtype=np.float32)
    print("\nClass probabilities:")
    print(model.predict(x_test))


if __name__ == "__main__":
    main()
