import numpy as np
import tensorflow as tf

def trend_model(theta, t_, p):

    '''
    Generate the trend backcast or forecast by multiplying the matrix of basis expansion coefficients
    by the matrix of powers of the time index. See Section 3.3 in the N-BEATS paper.

    Parameters:
    __________________________________
    theta: tf.Tensor.
        Basis expansion coefficients, 2-dimensional tensor with shape (N, p) where N is the batch
        size and p is the number of polynomial terms.

    t_: tf.Tensor.
        Time index, 1-dimensional tensor with length t for the trend backcast or with length H for
        the trend forecast.

    p: int.
        Number of polynomial terms.

    Returns:
    __________________________________
    tf.Tensor.
        Predicted trend, 2-dimensional tensor with shape (N, t) when backcasting and (N, H) when
        forecasting where N is the batch size.
    '''

    # Generate the matrix of polynomial terms. The shape of this matrix is (p, t) when backcasting
    # and (p, H) when forecasting.
    T_ = tf.map_fn(lambda p: t_ ** p, tf.range(p, dtype=tf.float32))

    # Calculate the dot product of the matrix of basis expansion coefficients and of the matrix
    # of polynomial terms. The shape of this matrix is (N, p) x (p, t) = (N, t) when backcasting
    # and (N, p) x (p, H) = (N, H) when forecasting where N is the batch size.
    return tf.tensordot(theta, T_, axes=1)


def seasonality_model(theta, t_, p):

    '''
    Generate the seasonality backcast or forecast by multiplying the matrix of basis expansion
    coefficients by the matrix of Fourier terms. See Section 3.3 in the N-BEATS paper.

    Parameters:
    __________________________________
    theta: tf.Tensor.
        Basis expansion coefficients, 2-dimensional tensor with shape (N, 2 * p) where N is the
        batch size and 2 * p is the total number of Fourier terms, that is p sine functions plus
        p cosine functions.

    t_: tf.Tensor.
        Time index, 1-dimensional tensor with length t for the seasonality backcast or with
        length H for the seasonality forecast.

    p: int.
        Number of Fourier terms.

    Returns:
    __________________________________
    tf.Tensor.
        Predicted seasonality, 2-dimensional tensor with shape (N, t) when backcasting and
        (N, H) when forecasting where N is the batch size.
    '''

    # Generate the matrix of Fourier terms. The shape of this matrix is (2 * p, t) when
    # backcasting and (2 * p, H) when forecasting.
    s1 = tf.map_fn(lambda p_: tf.math.cos(2 * np.pi * p_ * t_), tf.range(p, dtype=tf.float32))
    s2 = tf.map_fn(lambda p_: tf.math.sin(2 * np.pi * p_ * t_), tf.range(p, dtype=tf.float32))
    S = tf.concat([s1, s2], axis=0)

    # Calculate the dot product of the matrix of basis expansion coefficients and of the matrix of
    # Fourier terms. The shape of this matrix is (N, 2 * p) x (2 * p, t) = (N, t) when backcasting
    # and (N, 2 * p) x (2 * p, H) = (N, H) when forecasting where N is the batch size.
    return tf.tensordot(theta, S, axes=1)


def trend_block(h, p, t_b, t_f, share_theta):

    '''
    Derive the trend basis expansion coefficients and generate the trend backcast and forecast.
    See Section 3.1 and Section 3.3 in the N-BEATS paper.

    Parameters:
    __________________________________
    h: tf.Tensor.
        Output of 4-layer fully connected stack, 2-dimensional tensor with shape (N, k) where
        N is the batch size and k is the number of hidden units of each fully connected layer.
        Note that all fully connected layers have the same number of units.

    p: int.
       Number of polynomial terms.

    t_b: tf.Tensor.
        Input time index, 1-dimensional tensor with length t used for generating the backcast.

    t_f: tf.Tensor.
        Output time index, 1-dimensional tensor with length H used for generating the forecast.

    share_theta: bool.
        True if the backcast and forecast should share the same basis expansion coefficients,
        False otherwise.

    Returns:
    __________________________________
    backcast: tf.Tensor.
        Trend backcast, 2-dimensional tensor with shape (N, t) where N is the batch size and
        t is the length of the lookback period.

    forecast: tf.Tensor.
        Trend forecast, 2-dimensional tensor with shape (N, H) where N is the batch size and
        H is the length of the forecast period.
    '''

    if share_theta:

        # If share_theta is true, use the same basis expansion
        # coefficients for backcasting and forecasting.
        theta = tf.keras.layers.Dense(units=p, activation='linear', use_bias=False)(h)

        # Obtain the backcast and forecast as the dot product of their common
        # basis expansion coefficients and of the matrix of polynomial terms.
        backcast = tf.keras.layers.Lambda(function=trend_model, arguments={'p': p, 't_': t_b})(theta)
        forecast = tf.keras.layers.Lambda(function=trend_model, arguments={'p': p, 't_': t_f})(theta)

    else:

        # If share_theta is false, use different basis expansion
        # coefficients for backcasting and forecasting.
        theta_b = tf.keras.layers.Dense(units=p, activation='linear', use_bias=False)(h)
        theta_f = tf.keras.layers.Dense(units=p, activation='linear', use_bias=False)(h)

        # Obtain the backcast and forecast as the dot product of their respective
        # basis expansion coefficients and of the matrix of polynomial terms.
        backcast = tf.keras.layers.Lambda(function=trend_model, arguments={'p': p, 't_': t_b})(theta_b)
        forecast = tf.keras.layers.Lambda(function=trend_model, arguments={'p': p, 't_': t_f})(theta_f)

    return backcast, forecast


def seasonality_block(h, p, t_b, t_f, share_theta):

    '''
    Derive the seasonality basis expansion coefficients and generate the seasonality
    backcast and forecast. See Section 3.1 and Section 3.3 in the N-BEATS paper.

    Parameters:
    __________________________________
    h: tf.Tensor.
        Output of 4-layer fully connected stack, 2-dimensional tensor with shape (N, k) where
        N is the batch size and k is the number of hidden units of each fully connected layer.
        Note that all fully connected layers have the same number of units.

    p: int.
        Number of Fourier terms.

    t_b: tf.Tensor.
        Input time index, 1-dimensional tensor with length t used for generating the backcast.

    t_f: tf.Tensor.
        Output time index, 1-dimensional tensor with length H used for generating the forecast.

    share_theta: bool.
        True if the backcast and forecast should share the same basis expansion coefficients,
        False otherwise.

    Returns:
    __________________________________
    backcast: tf.Tensor.
        Seasonality backcast, 2-dimensional tensor with shape (N, t) where N is the batch size and
        t is the length of the lookback period.

    forecast: tf.Tensor.
        Seasonality forecast, 2-dimensional tensor with shape (N, H) where N is the batch size and
        H is the length of the forecast period.
    '''

    if share_theta:

        # If share_theta is true, use the same basis expansion
        # coefficients for backcasting and forecasting.
        theta = tf.keras.layers.Dense(units=2 * p, activation='linear', use_bias=False)(h)

        # Obtain the backcast and forecast as the dot product of their common
        # basis expansion coefficients and of the matrix of Fourier terms.
        backcast = tf.keras.layers.Lambda(function=seasonality_model, arguments={'p': p, 't_': t_b})(theta)
        forecast = tf.keras.layers.Lambda(function=seasonality_model, arguments={'p': p, 't_': t_f})(theta)

    else:

        # If share_theta is false, use different basis expansion
        # coefficients for backcasting and forecasting.
        theta_b = tf.keras.layers.Dense(units=2 * p, activation='linear', use_bias=False)(h)
        theta_f = tf.keras.layers.Dense(units=2 * p, activation='linear', use_bias=False)(h)

        # Obtain the backcast and forecast as the dot product of their respective
        # basis expansion coefficients and of the matrix of Fourier terms.
        backcast = tf.keras.layers.Lambda(function=seasonality_model, arguments={'p': p, 't_': t_b})(theta_b)
        forecast = tf.keras.layers.Lambda(function=seasonality_model, arguments={'p': p, 't_': t_f})(theta_f)

    return backcast, forecast


def generic_block(h, p, t_b, t_f, share_theta):

    '''
    Derive the generic basis expansion coefficients and generate the backcast and
    forecast. See Section 3.1 and Section 3.3 in the N-BEATS paper.

    Parameters:
    __________________________________
    h: tf.Tensor.
        Output of 4-layer fully connected stack, 2-dimensional tensor with shape (N, k) where
        N is the batch size and k is the number of hidden units of each fully connected layer.
        Note that all fully connected layers have the same number of units.

    p: int.
        Number of linear terms.

    t_b: tf.Tensor.
        Input time index, 1-dimensional tensor with length t used for generating the backcast.

    t_f: tf.Tensor.
        Output time index, 1-dimensional tensor with length H used for generating the forecast.

    share_theta: bool.
        True if the backcast and forecast should share the same basis expansion coefficients,
        False otherwise.

    Returns:
    __________________________________
    backcast: tf.Tensor.
        Generic backcast, 2-dimensional tensor with shape (N, t) where N is the batch size and
        t is the length of the lookback period.

    forecast: tf.Tensor.
        Generic forecast, 2-dimensional tensor with shape (N, H) where N is the batch size and
        H is the length of the forecast period.
    '''

    if share_theta:

        # If share_theta is true, use the same basis expansion
        # coefficients for backcasting and forecasting.
        theta = tf.keras.layers.Dense(units=p, activation='linear', use_bias=False)(h)

        # Obtain the backcast and forecast as a linear function of
        # their common basis expansion coefficients.
        backcast = tf.keras.layers.Dense(units=len(t_b), activation='linear')(theta)
        forecast = tf.keras.layers.Dense(units=len(t_f), activation='linear')(theta)

    else:

        # If share_theta is false, use different basis expansion
        # coefficients for backcasting and forecasting.
        theta_b = tf.keras.layers.Dense(units=p, activation='linear', use_bias=False)(h)
        theta_f = tf.keras.layers.Dense(units=p, activation='linear', use_bias=False)(h)

        # Obtain the backcast and forecast as a linear function of
        # their respective basis expansion coefficients.
        backcast = tf.keras.layers.Dense(units=len(t_b), activation='linear')(theta_b)
        forecast = tf.keras.layers.Dense(units=len(t_f), activation='linear')(theta_f)

    return backcast, forecast
