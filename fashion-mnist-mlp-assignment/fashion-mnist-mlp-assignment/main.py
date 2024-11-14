from pathlib import Path
from datetime import datetime

import tensorflow as tf
import keras_tuner as kt
from keras.src.optimizers.schedules import ExponentialDecay

import config


def main():
    """The main starting point for the application."""
    # getting the dataset with validation set of size 5000
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_dataset(5000)

    # scaling the features
    X_train, X_val, X_test = scale_features([X_train, X_val, X_test], 255.)

    # searches for the optimal hyperparameters
    search(get_tuner(), X_train, y_train, X_val, y_val)


def search(tuner, X_train, y_train, X_val, y_val):
    """Searches for the optimal hyperparameters for a given tuner."""
    tuner.search(
        X_train, y_train, epochs=config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[
            get_tensorboard_cb(tuner.project_dir), get_early_stopping_cb()
        ]
    )


def get_tuner():
    """Gets the tuner to use for searching."""
    return kt.Hyperband(
        build_model, objective="val_accuracy", seed=config.RND_SEED,
        max_epochs=config.MAX_EPOCHS, factor=config.FACTOR, hyperband_iterations=config.HYPERBAND_ITERATIONS,
        overwrite=False, directory=config.ROOT_LOG_DIR, project_name=config.PROJECT_NAME
    )


def build_model(hp):
    """Build the MLP model."""
    # creating model and adding layers
    model = add_layers(tf.keras.Sequential(), hp)

    # compiling model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=get_optimizer(optimizers_space(hp), get_lr_schedule(hp)),
        metrics=["accuracy"]
    )

    return model


def add_layers(model, hp):
    """Adds layers to the model using batch normalization before activation."""
    # adding input layer
    model.add(tf.keras.layers.Input(shape=config.INPUT_SHAPE))
    model.add(tf.keras.layers.Flatten())

    for _ in range(n_hidden_space(hp)):
        # no need for bias as batch normalization includes one offset parameter per input
        model.add(tf.keras.layers.Dense(n_neurons_space(hp), kernel_initializer=initializers_space(hp), use_bias=False))
        # adding batch normalization before activation
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(activation_space(hp)))

    # adding output layer
    model.add(tf.keras.layers.Dense(config.N_OUTPUTS, activation="softmax"))

    return model


def get_early_stopping_cb():
    """Generates an early stopping callback."""
    return tf.keras.callbacks.EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)


def get_tensorboard_cb(project_dir):
    """Generates a TensorBoard callback."""
    return tf.keras.callbacks.TensorBoard(log_dir=generate_logdir(project_dir))


def generate_logdir(project_dir):
    """Generates a logging directory for the current run and returns it."""
    # generates a timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # generates an appropriate directory
    logdir = Path(project_dir) / "tensorboard" / timestamp
    logdir.mkdir(parents=True, exist_ok=True)

    return logdir


def get_lr_schedule(hp):
    """Generates an exponential decay learning rate schedule."""
    return ExponentialDecay(
        initial_learning_rate=initial_lr_space(hp),
        decay_steps=config.LR_DECAY_STEPS,
        decay_rate=config.LR_DECAY_RATE
    )


def get_optimizer(name, initial_lr):
    """Returns an optimizer based on the give name and initial learning rate."""
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=initial_lr)

    elif name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=initial_lr)


def n_hidden_space(hp):
    """Defines the hyperparameter space for the number of hidden layers."""
    return hp.Int("n_hidden", min_value=config.N_HIDDEN_MIN, max_value=config.N_HIDDEN_MAX,
                      default=config.N_HIDDEN_DEFAULT)


def n_neurons_space(hp):
    """Defines the hyperparameter space for the number of neurons in each hidden layer."""
    return hp.Int("n_neurons", min_value=config.N_NEURONS_MIN, max_value=config.N_NEURONS_MAX)


def activation_space(hp):
    """Defines the hyperparameter space for the activation function used in hidden layers."""
    return hp.Choice("activation", values=config.ACTIVATION_CHOICES)


def optimizers_space(hp):
    """Defines the hyperparameter space for the optimizer used in training."""
    return hp.Choice("optimizer", values=config.OPTIMIZER_CHOICES)


def initializers_space(hp):
    """Defines the hyperparameter space for the initializers used in training."""
    return hp.Choice("initializer", values=config.INITIALIZER_CHOICES)


def initial_lr_space(hp):
    """Defines the hyperparameter space for the initial learning rate."""
    return hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")


def get_dataset(n_val):
    """Fetches the dataset and splits it into training, validation, and test sets."""
    fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist

    return ((X_train_full[:-n_val], y_train_full[:-n_val]),
            (X_train_full[-n_val:], y_train_full[-n_val:]),
            (X_test, y_test))


def scale_features(features, factor):
    """Scales the features by some factor."""
    for feature in features:
        feature = feature / factor

    return features


if __name__ == "__main__":
    main()
