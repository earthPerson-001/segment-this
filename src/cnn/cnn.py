from sklearn.datasets import load_sample_images
import tensorflow as tf
from functools import partial

from sklearn.datasets import fetch_openml

import matplotlib.pyplot as plt

from keras.utils import to_categorical  # one-hot encode target column

import sys

DefaultConv2D = partial(
    tf.keras.layers.Conv2D,
    kernel_size=3,
    padding="same",
    activation="relu",
    kernel_initializer="he_normal",
)


def get_large_cnn_model():
    return tf.keras.Sequential(
        [
            DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
            tf.keras.layers.MaxPool2D(),
            DefaultConv2D(filters=128),
            DefaultConv2D(filters=128),
            tf.keras.layers.MaxPool2D(),
            DefaultConv2D(filters=256),
            DefaultConv2D(filters=256),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=128, activation="relu", kernel_initializer="he_normal"
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                units=64, activation="relu", kernel_initializer="he_normal"
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(units=10, activation="softmax"),
        ]
    )


def get_small_cnn_model():
    # create model
    model = tf.keras.Sequential()  # add model layers
    model.add(
        tf.keras.layers.Conv2D(
            64, kernel_size=3, activation="relu", input_shape=(28, 28, 1)
        )
    )
    model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    return model


def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")


def start():
    # mnist_784 dataset consists of handwritten digits
    mnist = fetch_openml("mnist_784", as_frame=False)

    data, target = (mnist.data, mnist.target)

    # some_digit = data[0]
    # plot_digit(some_digit)
    # plt.show()

    # compiling the model
    cnn_model = get_small_cnn_model()
    cnn_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    # for changing the input shape
    model_input_shape = cnn_model.layers[0].input_shape

    # train test dataset
    X_train, X_test, y_train, y_test = (
        data[:60000],
        data[60000:],
        target[:60000],
        target[60000:],
    )

    # changing the dimensions for passing through CNN
    input_shape_train = X_train.shape[0], model_input_shape[1], model_input_shape[2]
    input_shape_test = X_test.shape[0], model_input_shape[1], model_input_shape[2]

    reshaped_X_train, reshaped_X_test = (
        X_train.reshape(input_shape_train),
        X_test.reshape(input_shape_test),
    )

    # one hot encode (change to columns with 1 in the place corresponding to the number)
    # 6 => [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    categorical_y_train = to_categorical(y_train)
    categorical_y_test = to_categorical(y_test)

    # training the model
    cnn_model.fit(
        reshaped_X_train,
        categorical_y_train,
        validation_data=(reshaped_X_test, categorical_y_test),
        epochs=3,
    )


if __name__ == "__main__":
    start()
