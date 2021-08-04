from logisds.utils import get_console_logger

import tensorflow as tf


class TSModel:
    def __init__(self, N_steps, N_features, lr):
        self.N_steps = N_steps
        self.N_features = N_features
        self.lr = lr

    def get_model(self):
        layer_input = tf.keras.layers.Input(
            shape=(self.N_steps, self.N_features)
        )
        layer_gru, _, _ = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                units=8,
                return_sequences=True,
                return_state=True
            )
        )(layer_input)
        layer_flatten = tf.keras.layers.Flatten()(layer_gru)
        layer_dropout1 = tf.keras.layers.Dropout(0.3)(layer_flatten)
        layer_dense_output_time = tf.keras.layers.Dense(1, activation="relu")(
            layer_dropout1)
        layer_dense_output_xy = tf.keras.layers.Dense(2, activation=None)(
            layer_dropout1)
        output = tf.keras.layers.Concatenate()(
            [layer_dense_output_time, layer_dense_output_xy])

        model = tf.keras.Model(inputs=layer_input, outputs=output)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        loss = tf.keras.losses.mean_squared_error
        model.compile(optimizer=optimizer, loss=loss)
        return model


