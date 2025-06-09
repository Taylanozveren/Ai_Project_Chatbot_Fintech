# dl/cnn.py

"""
Defines a baseline 1D-CNN model for binary classification of 3-day >2% price jumps.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dropout, Dense


def build_cnn(input_shape, n_outputs=1, dropout_rate=0.3,
              filters=(32, 64), kernel_sizes=(5, 3)):
    """
    Builds and compiles a simple 1D-CNN model.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input sequence (SEQ_LEN, n_features).
    n_outputs : int
        Number of output units (1 for binary classification).
    dropout_rate : float
        Dropout rate between Conv layers.
    filters : tuple of int
        Number of filters for first and second Conv1D layers.
    kernel_sizes : tuple of int
        Kernel sizes for the two Conv1D layers.

    Returns
    -------
    model : tf.keras.Model
        Compiled Keras model ready for training.
    """
    model = Sequential()
    # First Conv1D block
    model.add(
        Conv1D(filters[0], kernel_sizes[0], activation='relu',
               input_shape=input_shape)
    )
    model.add(Dropout(dropout_rate))

    # Second Conv1D block
    model.add(
        Conv1D(filters[1], kernel_sizes[1], activation='relu')
    )
    model.add(Dropout(dropout_rate))

    # Pooling and dense heads
    model.add(GlobalMaxPooling1D())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))

    # Compile with binary crossentropy and relevant metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )
    return model
