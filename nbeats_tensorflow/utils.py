import pandas as pd
import numpy as np
import tensorflow as tf
import warnings

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
    y: np.array
        Time series.

    t: int
        Length of input sequences (lookback period).

    H: int
        Length of output sequences (forecast period).

    Returns:
    __________________________________
    X: np.array
        Input sequences, 2-dimensional array of with shape (N - t - H + 1, t) where N is the length
        of the time series.

    Y: np.array
       Output sequences, 2-dimensional array with shape (N - t - H + 1, H) where N is the length
       of the time series.
    '''

    if H < 1:
        raise ValueError('The length of the forecast period should be greater than or equal to one.')

    elif t <= 1:
        raise ValueError('The length of the lookback period should be greater than one.')

    elif t + H >= len(y):
        raise ValueError('The combined length of the forecast and lookback periods cannot exceed the '
                         'length of the time series.')

    else:

        if len(y) - t - H + 1 < 100:
            warnings.warn('Only {} sequences are available for training the model, consider decreasing '
                          'the length of the forecast and lookback periods.'.format(len(y) - t - H + 1))

        if t % H != 0:
            warnings.warn('It is recommended to set the length of the lookback period equal to a multiple '
                          'of the length of the forecast period, this multiple is usually between 2 and 7.')

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
    t: int
        Length of input sequences (lookback period).

    H: int
        Length of output sequences (forecast period).

    Returns:
    __________________________________
    t_b: tf.Tensor
        Input time index, 1-dimensional tensor with shape t used for backcasting.

    t_f: tf.Tensor
        Output time index, 1-dimensional tensor with shape H used for forecasting.
    '''

    # Full time index, tensor with shape t + H.
    t_ = tf.cast(tf.range(0, t + H), dtype=tf.float32) / (t + H)

    # Input time index, tensor with shape t.
    t_b = t_[:t]

    # Output time index, tensor with shape H.
    t_f = t_[t:]

    return t_b, t_f


def get_loss_function(loss, alpha):

    '''
    Define the loss function.

    Parameters:
    __________________________________
    loss: str
        Loss type, one of:
         - 'mse': mean squared error,
         - 'mae': mean absolute error,
         - 'mape': mean absolute percentage error,
         - 'smape': symmetric mean absolute percentage error.

    alpha: float
        Backcast loss weight, must be between 0 and 1.

    Returns:
    __________________________________
    loss_fun: function.
        Loss function taking as input the actual and predicted values,
        both of which are of type tf.Tensor.
    '''

    if alpha < 0 or alpha > 1:
        raise ValueError('The backcast loss weight must be between zero and one.')

    else:

        if loss == 'mse':

            def loss_fun(y_true, y_pred):
                backcast_loss = tf.reduce_mean((y_true[0] - y_pred[0]) ** 2)
                forecast_loss = tf.reduce_mean((y_true[1] - y_pred[1]) ** 2)
                return alpha * backcast_loss + (1 - alpha) * forecast_loss

            return loss_fun

        elif loss == 'mae':

            def loss_fun(y_true, y_pred):
                backcast_loss = tf.reduce_mean(tf.math.abs(y_true[0] - y_pred[0]))
                forecast_loss = tf.reduce_mean(tf.math.abs(y_true[1] - y_pred[1]))
                return alpha * backcast_loss + (1 - alpha) * forecast_loss

            return loss_fun

        elif loss == 'mape':

            def loss_fun(y_true, y_pred):
                backcast_loss = tf.reduce_mean(tf.math.abs((y_true[0] - y_pred[0]) / y_true[0]))
                forecast_loss = tf.reduce_mean(tf.math.abs((y_true[1] - y_pred[1]) / y_true[1]))
                return alpha * backcast_loss + (1 - alpha) * forecast_loss

            return loss_fun

        elif loss == 'smape':

            def loss_fun(y_true, y_pred):
                backcast_loss = 2 * tf.reduce_mean(tf.math.abs((y_true[0] - y_pred[0]) / (y_true[0] + y_pred[0])))
                forecast_loss = 2 * tf.reduce_mean(tf.math.abs((y_true[1] - y_pred[1]) / (y_true[1] + y_pred[1])))
                return alpha * backcast_loss + (1 - alpha) * forecast_loss

            return loss_fun

        else:

            raise ValueError('Undefined loss function {}'.format(loss))
