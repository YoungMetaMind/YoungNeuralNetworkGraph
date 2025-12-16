# PythonTensorFlowMultiLabel.py
import tensorflow as tf

def build_model() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(2,), name="input"),
            tf.keras.layers.Dense(6, activation="relu", name="dense_6"),
            tf.keras.layers.Dense(2, activation="relu", name="dense_2a"),
            tf.keras.layers.Dense(2, activation="relu", name="dense_2b"),
            tf.keras.layers.Dense(8, activation="relu", name="dense_8"),
            tf.keras.layers.Dense(2, activation="relu", name="dense_2c"),
            tf.keras.layers.Dense(2, activation="relu", name="dense_2d"),
            tf.keras.layers.Dense(6, activation="relu", name="dense_6b"),
            tf.keras.layers.Dense(2, activation="sigmoid", name="output_sigmoid"),
        ],
        name="nn_2_6_2_2_8_2_2_6_2_multilabel",
    )

def compile_model(model: tf.keras.Model) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="bin_acc", threshold=0.5),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

def main() -> None:
    tf.random.set_seed(42)

    model = build_model()
    compile_model(model)
    model.summary()

    # Dummy data (replace with your real dataset)
    x = tf.random.normal(shape=(256, 2))
    y = tf.cast(
        tf.random.uniform(shape=(256, 2), minval=0, maxval=2, dtype=tf.int32),
        tf.float32,
    )  # two independent binary labels

    model.fit(x, y, epochs=5, batch_size=32, verbose=1)

if __name__ == "__main__":
    main()

