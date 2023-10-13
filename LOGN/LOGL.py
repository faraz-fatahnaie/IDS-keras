import tensorflow as tf
from keras.layers import Layer, Dense, Concatenate, Input
from keras.models import Model
import pandas as pd
import numpy as np
from model.CNNATTENTION import CNNATTENTION1D
from utils import parse_data


class LogarithmicNeuron(Layer):
    def __init__(self, **kwargs):
        super(LogarithmicNeuron, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mi = self.add_weight(name='mi',
                                  shape=(1,),
                                  initializer='ones',
                                  trainable=True)
        super(LogarithmicNeuron, self).build(input_shape)

    def call(self, x):
        result = tf.math.log(tf.keras.activations.relu(self.mi) * x + 1.0)
        return result

    def compute_output_shape(self, input_shape):
        return input_shape


# Function to create the Logarithmic layer for a given input shape
def create_logarithmic_layer(in_shape):
    mi_inputs = [Input(shape=(1,)) for _ in range(in_shape[-1])]
    neurons = [LogarithmicNeuron()(mi_input) for mi_input in mi_inputs]
    concatenated = Concatenate()(neurons)
    return mi_inputs, concatenated


if __name__ == "__main__":

    train = pd.read_csv(
        'C:\\Users\\Faraz\\PycharmProjects\\IDS-keras\\dataset\\UNSW_NB15\\train_binary_withoutNorm_2neuron.csv')

    test = pd.read_csv(
        'C:\\Users\\Faraz\\PycharmProjects\\IDS-keras\\dataset\\UNSW_NB15\\test_binary_withoutNorm_2neuron.csv')

    X_train, y_train = parse_data(train, 'UNSW_NB15', 'binary')
    X_test, y_test = parse_data(test, 'UNSW_NB15', 'binary')
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    input_shape = (196,)  # Adjust this to match your input shape
    epochs = 20
    batch_size = 1

    # Create a model with the logarithmic layer
    mi_inputs, logarithmic_layer = create_logarithmic_layer(input_shape)
    # output = CNNATTENTION1D()(logarithmic_layer)
    x_loss = Dense(64)(logarithmic_layer)
    x = tf.keras.activations.tanh(x_loss)
    x = Dense(128, activation='tanh')(x)
    output = Dense(2)(x)
    output = tf.keras.activations.sigmoid(output)

    model = Model(inputs=mi_inputs, outputs=output)
    input_data = [X_train[:, i:i + 1].astype('float32') for i in range(X_train.shape[1])]
    X_test = [X_test[:, i:i + 1].astype('float32') for i in range(X_test.shape[1])]

    # Compile and train the model
    opt = tf.keras.optimizers.legacy.Adam
    initial_learning_rate = 0.001
    final_learning_rate = 0.00001
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / epochs)
    steps_per_epoch = int(len(X_train) / batch_size)

    lr_schedule = {
        'ExponentialDecay': tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=steps_per_epoch,
            decay_rate=learning_rate_decay_factor,
            staircase=True),

        'ReduceLROnPlateau': tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=2,
            verbose=0,
            mode="auto",
            min_delta=0.01,
            cooldown=0,
            min_lr=0
        )}

    criterion = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=opt(lr_schedule['ExponentialDecay']), loss=criterion, metrics=['accuracy'])
    model.fit(input_data, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
