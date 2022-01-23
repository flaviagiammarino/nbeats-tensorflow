import pandas as pd
import numpy as np
import tensorflow as tf

def cast_target_to_array(y):

    '''
    Cast the time series to np.array.

    Parameters:
    __________________________________
    y: np.array, pd.Series, list.
        Time series as numpy array, pandas series or list.

    Returns:
    __________________________________
    y: np.array.
        Time series as numpy array.
    '''

    if type(y) not in [pd.core.series.Series, np.ndarray, list]:
        raise ValueError('The input time series must be either a pandas series, a numpy array, or a list.')

    elif pd.isna(y).sum() != 0:
        raise ValueError('The input time series cannot contain missing values.')

    elif np.std(y) == 0:
        raise ValueError('The input time series cannot be constant.')

    else:

        if type(y) == pd.core.series.Series:
            y = y.values

        elif type(y) == list:
            y = np.array(y)

        return y


def get_training_sequences(y, t, H):

    '''
    Split the time series into input and output sequences. These are used for training the model.
    See Section 2 in the N-BEATS paper.

    Parameters:
    __________________________________
    y: np.array.
        Time series.

    t: int.
        Length of input sequences (lookback period).

    H: int.
        Length of output sequences (forecast period).

    Returns:
    __________________________________
    X: np.array.
        Input sequences, 2-dimensional array with shape (N - t - H + 1, t) where N is the length
        of the time series.

    Y: np.array.
       Output sequences, 2-dimensional array with shape (N - t - H + 1, H) where N is the length
       of the time series.
    '''

    if H < 1:
        raise ValueError('The length of the forecast period should be greater than one.')

    if t < H:
        raise ValueError('The lookback period cannot be shorter than the forecast period.')

    if t + H >= len(y):
        raise ValueError('The combined length of the forecast and lookback periods cannot exceed the length of the time series.')

    X = []
    Y = []

    for T in range(t, len(y) - H + 1):
        X.append(y[T - t: T])
        Y.append(y[T: T + H])

    X = np.array(X)
    Y = np.array(Y)

    return X, Y


def get_time_indices(t, H):

    '''
    Generate the time indices corresponding to the input and output sequences. These are used for estimating
    the trend and seasonality components of the model. See Section 3.3 in the N-BEATS paper.

    Parameters:
    __________________________________
    t: int.
        Length of input sequences (lookback period).

    H: int.
        Length of output sequences (forecast period).

    Returns:
    __________________________________
    t_b: tf.Tensor.
        Input time index, 1-dimensional tensor with length t used for backcasting.

    t_f: tf.Tensor.
        Output time index, 1-dimensional tensor with length H used for forecasting.
    '''

    # Full time index, 1-dimensional tensor with length t + H.
    t_ = tf.cast(tf.range(0, t + H), dtype=tf.float32) / (t + H)

    # Input time index, 1-dimensional tensor with length t.
    t_b = t_[:t]

    # Output time index, 1-dimensional tensor with length H.
    t_f = t_[t:]

    return t_b, t_f
