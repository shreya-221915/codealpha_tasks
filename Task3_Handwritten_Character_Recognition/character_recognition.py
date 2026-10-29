
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


def build_emnist_cnn():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    print("=== Task 3: Handwritten Character Recognition ===")

    # Load the MNIST handwritten digit dataset
    (X_train, y_train), (X_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Normalize pixel values and reshape images
    X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    model = build_emnist_cnn()
    model.summary()

    # Train the CNN
    model.fit(
        X_train,
        y_train,
        epochs=1,
        validation_data=(X_test, y_test),
        batch_size=64
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Save the trained model
    model.save("character_recognition.keras")
    print("Model saved successfully!")
