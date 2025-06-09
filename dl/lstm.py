# dl/lstm.py

"""
Defines a baseline Bidirectional LSTM model for binary classification of 3-day >2% price jumps.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense


def build_lstm(input_shape, n_outputs=1, dropout_rate=0.3, lstm_units=(64, 32)):
    """
    Builds and compiles a Bi-directional LSTM model.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input sequence (SEQ_LEN, n_features).
    n_outputs : int
        Number of output units (1 for binary classification).
    dropout_rate : float
        Dropout rate between LSTM layers.
    lstm_units : tuple of int
        Number of units in the first and second LSTM layers.

    Returns
    -------
    model : tf.keras.Model
        Compiled Keras model ready for training.
    """
    model = Sequential()
    # First Bi-LSTM layer with sequence output
    model.add(
        Bidirectional(
            LSTM(lstm_units[0], return_sequences=True),
            input_shape=input_shape
        )
    )
    model.add(Dropout(dropout_rate))

    # Second Bi-LSTM layer without returning sequences
    model.add(Bidirectional(LSTM(lstm_units[1])))
    model.add(Dropout(dropout_rate))

    # Dense layers for classification
    model.add(Dense(32, activation="relu"))
    model.add(Dense(n_outputs, activation="sigmoid"))

    # Compile with binary crossentropy and relevant metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )
    return model
