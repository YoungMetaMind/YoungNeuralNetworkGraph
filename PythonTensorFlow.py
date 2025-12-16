#!/usr/bin/env python3
"""
PythonTensorFlow.py

Implements the MLP architecture:
2 → 6 → 2 → 2 → 8 → 2 → 2 → 6 → 2

Supports 3 output modes:
A) regression   : 2 continuous outputs
B) softmax      : 2-class classification (mutually exclusive)
C) multilabel   : 2 independent probabilities
"""

from __future__ import annotations
import argparse
import numpy as np
import tensorflow as tf


def build_model(mode: str, lr: float = 1e-3) -> tf.keras.Model:
    """
    mode:
      - "regression"
      - "softmax"
      - "multilabel"
    """
    mode = mode.lower()
    if mode not in {"regression", "softmax", "multilabel"}:
        raise ValueError("mode must be one of: regression, softmax, multilabel")

    inputs = tf.keras.Input(shape=(2,), name="x")

    x = tf.keras.layers.Dense(6, activation="relu", name="dense_2_6")(inputs)
    x = tf.keras.layers.Dense(2, activation="relu", name="dense_6_2")(x)
    x = tf.keras.layers.Dense(2, activation="relu", name="dense_2_2_a")(x)
    x = tf.keras.layers.Dense(8, activation="relu", name="dense_2_8")(x)
    x = tf.keras.layers.Dense(2, activation="relu", name="dense_8_2")(x)
    x = tf.keras.layers.Dense(2, activation="relu", name="dense_2_2_b")(x)
    x = tf.keras.layers.Dense(6, activation="relu", name="dense_2_6_b")(x)

    if mode == "regression":
        outputs = tf.keras.layers.Dense(2, activation=None, name="y")(x)
        loss = tf.keras.losses.MeanSquaredError()
        metrics = [tf.keras.metrics.MeanAbsoluteError(name="mae")]

    elif mode == "softmax":
        outputs = tf.keras.layers.Dense(2, activation="softmax", name="y")(x)
        # targets: shape (N,), int {0,1}
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]

    else:  # multilabel
        outputs = tf.keras.layers.Dense(2, activation="sigmoid", name="y")(x)
        # targets: shape (N,2), float {0,1}
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [tf.keras.metrics.BinaryAccuracy(name="bin_acc")]

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"YoungNet_{mode}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=metrics,
    )
    return model


def make_dummy_data(mode: str, n: int = 512, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)).astype(np.float32)

    mode = mode.lower()
    if mode == "regression":
        y = rng.normal(size=(n, 2)).astype(np.float32)
    elif mode == "softmax":
        y = rng.integers(0, 2, size=(n,), dtype=np.int32)
    elif mode == "multilabel":
        y = rng.integers(0, 2, size=(n, 2)).astype(np.float32)
    else:
        raise ValueError("Invalid mode")

    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["regression", "softmax", "multilabel"],
        default="regression",
        help="Output interpretation",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    tf.get_logger().setLevel("ERROR")

    model = build_model(args.mode, lr=args.lr)
    model.summary()

    x, y = make_dummy_data(args.mode)

    model.fit(
        x,
        y,
        epochs=args.epochs,
        batch_size=args.batch,
        validation_split=0.2,
    )

    # Example inference
    x_test = np.array(
        [[0.1, -0.2],
         [1.0,  0.5]],
        dtype=np.float32,
    )

    preds = model.predict(x_test)
    print("\nPredictions:")
    print(preds)

    print("\nTarget formatting reminder:")
    if args.mode == "regression":
        print("  y shape: (N,2) float32, arbitrary real values")
    elif args.mode == "softmax":
        print("  y shape: (N,) int32 with values {0,1}")
    else:
        print("  y shape: (N,2) float32 with values {0,1}")


if __name__ == "__main__":
    main()
