# dl/lstm_multitask.py
"""
Multi-horizon BiLSTM: hem classification (h1, h3, h5) hem de regression (logret_h3, logret_h5).
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Dense

def build_lstm_multitask(input_shape,
                         lstm_units=(64, 32),
                         dropout_rate=0.3):
    inp = Input(shape=input_shape, name="sequence_input")
    x = Bidirectional(LSTM(lstm_units[0], return_sequences=True), name="bi_lstm_1")(inp)
    x = Dropout(dropout_rate, name="drop_1")(x)
    x = Bidirectional(LSTM(lstm_units[1]), name="bi_lstm_2")(x)
    x = Dropout(dropout_rate, name="drop_2")(x)
    x = Dense(32, activation="relu", name="shared_dense")(x)

    # Classification heads
    out_h1 = Dense(1, activation="sigmoid", name="h1")(x)
    out_h3 = Dense(1, activation="sigmoid", name="h3")(x)
    out_h5 = Dense(1, activation="sigmoid", name="h5")(x)
    # Regression heads
    out_r3 = Dense(1, activation="linear", name="r3")(x)
    out_r5 = Dense(1, activation="linear", name="r5")(x)

    model = Model(inputs=inp, outputs=[out_h1, out_h3, out_h5, out_r3, out_r5], name="lstm_multitask")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss={
            'h1': 'binary_crossentropy',
            'h3': 'binary_crossentropy',
            'h5': 'binary_crossentropy',
            'r3': 'mean_squared_error',
            'r5': 'mean_squared_error'
        },
        loss_weights={'h1': 0.5, 'h3': 1.0, 'h5': 1.0, 'r3': 0.2, 'r5': 0.2},
        metrics={
            'h1': [tf.keras.metrics.AUC(name='auc_h1')],
            'h3': [tf.keras.metrics.AUC(name='auc_h3')],
            'h5': [tf.keras.metrics.AUC(name='auc_h5')],
            'r3': [tf.keras.metrics.MeanAbsoluteError(name='mae_r3')],
            'r5': [tf.keras.metrics.MeanAbsoluteError(name='mae_r5')]
        }
    )
    return model

if __name__ == '__main__':
    model = build_lstm_multitask(input_shape=(60, 33))
    model.summary()
