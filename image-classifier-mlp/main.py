import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # load dataset
    fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()

    # split dataset
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
    X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
    X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

    # scaling pixel intensities down to the 0 - 1 range
    X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.

    # defining class names
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    tf.random.set_seed(42)
    # creating the model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.Dense(300, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    # compiling the model
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])

    # plotting the learning rate
    history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
    pd.DataFrame(history.history).plot(
        figsize=(8, 5), xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="Epoch", style=["r--", "r--.", "b-", "b-*"]
    )
    plt.show()

    # predicting 'new' instances
    X_new = X_test[:3]
    y_proba = model.predict(X_new)

    # getting the probability for each class for each instance
    print(y_proba.round(2))

    # getting the class with the highest probability for each instance
    y_pred = y_proba.argmax(axis=-1)
    print(np.array(class_names)[y_pred])




if __name__ == '__main__':
    main()