import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Dense, Subtract, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
pd.options.mode.chained_assignment = None

from nbeats_tensorflow.utils import cast_target_to_array, get_training_sequences, get_time_indices
from nbeats_tensorflow.blocks import trend_block, seasonality_block, generic_block

class NBeats():

    def __init__(self,
                 target,
                 forecast_period,
                 lookback_period,
                 num_trend_coefficients=3,
                 num_seasonal_coefficients=5,
                 num_generic_coefficients=7,
                 hidden_units=100,
                 stacks=['trend', 'seasonality'],
                 num_blocks_per_stack=3,
                 share_weights=True,
                 share_coefficients=True):

        '''
        Implementation of univariate time series forecasting model introduced in Oreshkin, B. N., Carpov, D.,
        Chapados, N., & Bengio, Y. (2019). N-BEATS: Neural basis expansion analysis for interpretable time
        series forecasting. arXiv preprint arXiv:1905.10437.

        Parameters:
        __________________________________
        target: np.array, pd.Series, list.
            Time series.

        forecast_period: int.
            Length of forecast period.

        lookback_period: int.
            Length of lookback period.

        num_trend_coefficients: int.
            Number of basis expansion coefficients of the trend block. This is the number of polynomial terms
            used for modelling the trend. The default is 3.

        num_seasonal_coefficients: int.
            Number of basis expansion coefficients of the seasonality block. This is the number of Fourier terms
            used for modelling the seasonality. The default is 5.

        num_generic_coefficients: int.
            Number of basis expansion coefficients of the generic block. The default is 7.

        hidden_units: int.
            Number of hidden units of each of the 4 layers of the fully connected stack. The default is 100.

        stacks: list of strings.
            The length of the list is the number of stacks, the items in the list are strings identifying the
            stack types (either 'trend', 'seasonality' or 'generic'). The default is ['trend', 'seasonality'].

        num_blocks_per_stack: int.
            The number of blocks in each stack. The default is 3.

        share_weights: bool.
            True if the weights of the 4 layers of the fully connected stack should be shared by the different
            blocks inside the same stack, False otherwise. The default is True.

        share_coefficients: bool.
            True if the forecasts and backcasts of each block should share the same basis expansion coefficients,
            False otherwise. The default is True.
        '''

        # Cast the data to numpy array.
        y = cast_target_to_array(target)

        # Scale the data.
        self.y_min, self.y_max = np.min(y), np.max(y)
        self.y = (y - self.y_min) / (self.y_max - self.y_min)

        # Extract the input and output sequences.
        self.X, self.Y = get_training_sequences(self.y, lookback_period, forecast_period)

        # Save the lengths of the input and output sequences.
        self.lookback_period = lookback_period
        self.forecast_period = forecast_period

        # Extract the time indices of the input and output sequences.
        backcast_time_idx, forecast_time_idx = get_time_indices(lookback_period, forecast_period)

        # Build the model graph
        self.model = build_model_graph(
            backcast_time_idx,
            forecast_time_idx,
            num_trend_coefficients,
            num_seasonal_coefficients,
            num_generic_coefficients,
            hidden_units,
            stacks,
            num_blocks_per_stack,
            share_weights,
            share_coefficients)

    def fit(self,
            loss='mse',
            learning_rate=0.01,
            batch_size=128,
            epochs=100,
            validation_split=0.2,
            backcast_loss_weight=0.5):

        '''
        Train the model.

        Parameters:
        __________________________________
        loss: str.
            Loss function, see https://www.tensorflow.org/api_docs/python/tf/keras/losses.
            The default is 'mse'.

        learning_rate: float.
            Learning rate, the default is 0.01.

        batch_size: int.
            Batch size, the default is 128.

        epochs: int.
            Number of epochs, the default is 100.

        validation_split: float.
            Fraction of the training data to be used as validation data, must be between 0 and 1. The default is 0.2.

        backcast_loss_weight: float.
            Weight of backcast in comparison to forecast when calculating the loss, must be between 0 and 1.
            A weight of 0.5 means that forecast and backcast loss is weighted the same. The default is 0.5.
        '''

        if backcast_loss_weight < 0 or backcast_loss_weight > 1:
            raise ValueError('The backcast loss weight must be between zero and one.')

        # Compile the model.
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=[loss, loss],
            loss_weights=[backcast_loss_weight, 1 - backcast_loss_weight]
        )

        # Fit the model.
        self.history = self.model.fit(
            x=self.X,
            y=[self.X, self.Y],
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
            callbacks=[callback()]
        )

    def predict(self, index, return_backcast=False):

        '''
        Extract the in-sample predictions.

        Parameters:
        __________________________________
        index: int.
            The start index of the sequence to predict.

        return_backcast: bool.
            True if the output should include the backcast, False otherwise.

        Returns:
        __________________________________
        predictions: pd.DataFrame
            Data frame including the actual and predicted values of the time series.
        '''

        if index < self.lookback_period:
            raise ValueError('The index must be greater than {}.'.format(self.lookback_period))

        elif index > len(self.y) - self.forecast_period:
            raise ValueError('The index must be less than {}.'.format(len(self.y) - self.forecast_period))

        # Extract the predictions for the selected sequence
        backcast, forecast = self.model.predict(self.X)
        backcast = self.y_min + (self.y_max - self.y_min) * backcast[index - self.lookback_period, :].flatten()
        forecast = self.y_min + (self.y_max - self.y_min) * forecast[index - self.lookback_period, :].flatten()

        # Organize the predictions in a data frame
        predictions = pd.DataFrame(columns=['time_idx', 'actual', 'forecast'])
        predictions['time_idx'] = np.arange(len(self.y))
        predictions['actual'] = self.y_min + (self.y_max - self.y_min) * self.y
        predictions['forecast'].iloc[index: (index + self.forecast_period)] = forecast

        if return_backcast:
            predictions['backcast'] = np.nan
            predictions['backcast'].iloc[(index - self.lookback_period): index] = backcast

        predictions = predictions.astype(float)

        # Save the data frame
        self.predictions = predictions

        # Return the data frame
        return predictions

    def forecast(self, return_backcast=False):

        '''
        Generate the out-of-sample forecasts.

        Parameters:
        __________________________________
        return_backcast: bool.
            True if the output should include the backcast, False otherwise.

        Returns:
        __________________________________
        forecasts: pd.DataFrame
            Data frame including the actual and predicted values of the time series.
        '''

        # Generate the forecasts
        backcast, forecast = self.model.predict(self.y[- self.lookback_period:].reshape(1, - 1))
        backcast = self.y_min + (self.y_max - self.y_min) * backcast[- 1, :].flatten()
        forecast = self.y_min + (self.y_max - self.y_min) * forecast[- 1, :].flatten()

        # Organize the forecasts in a data frame
        forecasts = pd.DataFrame(columns=['time_idx', 'actual', 'forecast'])
        forecasts['time_idx'] = np.arange(len(self.y) + self.forecast_period)
        forecasts['actual'].iloc[: - self.forecast_period] = self.y_min + (self.y_max - self.y_min) * self.y
        forecasts['forecast'].iloc[- self.forecast_period:] = forecast

        if return_backcast:
            forecasts['backcast'] = np.nan
            forecasts['backcast'].iloc[- (self.lookback_period + self.forecast_period): - self.forecast_period] = backcast

        forecasts = forecasts.astype(float)

        # Save the data frame
        self.forecasts = forecasts

        # Return the data frame
        return forecasts


def get_block_output(stack_type,
                     dense_layers_output,
                     backcast_time_idx,
                     forecast_time_idx,
                     num_trend_coefficients,
                     num_seasonal_coefficients,
                     num_generic_coefficients,
                     share_coefficients):

    '''
    Get a given block's backcast and forecast.

    Parameters:
    __________________________________
    stack_type: str.
        The stack type, either 'trend', 'seasonality' or 'generic'.

    dense_layers_output: tf.Tensor.
        Output of 4-layer fully connected stack, 2-dimensional tensor with shape (N, k) where
        N is the batch size and k is the number of hidden units of each fully connected layer.
        Note that all fully connected layers have the same number of units.

    backcast_time_idx: tf.Tensor
        Input time index, 1-dimensional tensor with shape t used for generating the backcasts.

    forecast_time_idx: tf.Tensor
        Output time index, 1-dimensional tensor with shape H used for generating the forecasts.

    num_trend_coefficients: int.
        Number of basis expansion coefficients of the trend block. This is the number of polynomial terms
        used for modelling the trend.

    num_seasonal_coefficients: int.
        Number of basis expansion coefficients of the seasonality block. This is the number of Fourier terms
        used for modelling the seasonality.

    num_generic_coefficients: int.
        Number of basis expansion coefficients of the generic block.

    share_coefficients: bool.
        True if the forecasts and backcasts of each block should share the same basis expansion coefficients,
        False otherwise.
    '''

    if stack_type == 'trend':

        return trend_block(
            h=dense_layers_output,
            p=num_trend_coefficients,
            t_b=backcast_time_idx,
            t_f=forecast_time_idx,
            share_theta=share_coefficients)

    elif stack_type == 'seasonality':

        return seasonality_block(
            h=dense_layers_output,
            p=num_seasonal_coefficients,
            t_b=backcast_time_idx,
            t_f=forecast_time_idx,
            share_theta=share_coefficients)

    else:

        return generic_block(
            h=dense_layers_output,
            p=num_generic_coefficients,
            t_b=backcast_time_idx,
            t_f=forecast_time_idx,
            share_theta=share_coefficients)


def build_model_graph(backcast_time_idx,
                      forecast_time_idx,
                      num_trend_coefficients,
                      num_seasonal_coefficients,
                      num_generic_coefficients,
                      hidden_units,
                      stacks,
                      num_blocks_per_stack,
                      share_weights,
                      share_coefficients):

    '''
    Build the model's graph.

    Parameters:
    __________________________________
    backcast_time_idx: tf.Tensor
        Input time index, 1-dimensional tensor with shape t used for generating the backcasts.

    forecast_time_idx: tf.Tensor
        Output time index, 1-dimensional tensor with shape H used for generating the forecasts.

    num_trend_coefficients: int.
        Number of basis expansion coefficients of the trend block, corresponds to the number of polynomial terms.

    num_seasonal_coefficients: int.
        Number of basis expansion coefficients of the seasonality block, corresponds to the number of Fourier terms.

    num_generic_coefficients: int.
        Number of basis expansion coefficients of the generic block, corresponds to the number of linear terms.

    hidden_units: int.
        Number of hidden units of each of the 4 layers of the fully connected stack.

    stacks: list of strings.
        The length of the list is the number of stacks, the items in the list are strings identifying the
        stack types (either 'trend', 'seasonality' or 'generic').

    num_blocks_per_stack: int.
        The number of blocks in each stack.

    share_weights: bool.
        True if the weights of the 4 layers of the fully connected stack should be shared by the different
        blocks inside the same stack, False otherwise.

    share_coefficients: bool.
        True if the forecasts and backcasts of each block should share the same basis expansion coefficients,
        False otherwise.
    '''

    x = Input(shape=len(backcast_time_idx))

    for s in range(len(stacks)):

        if share_weights:

            d1 = Dense(units=hidden_units, activation='relu')
            d2 = Dense(units=hidden_units, activation='relu')
            d3 = Dense(units=hidden_units, activation='relu')
            d4 = Dense(units=hidden_units, activation='relu')

        for b in range(num_blocks_per_stack):

            if s == 0 and b == 0:

                if share_weights:

                    h = d1(x)
                    h = d2(h)
                    h = d3(h)
                    h = d4(h)

                else:

                    h = Dense(units=hidden_units, activation='relu')(x)
                    h = Dense(units=hidden_units, activation='relu')(h)
                    h = Dense(units=hidden_units, activation='relu')(h)
                    h = Dense(units=hidden_units, activation='relu')(h)

                backcast_block, forecast_block = get_block_output(
                    stack_type=stacks[s],
                    dense_layers_output=h,
                    backcast_time_idx=backcast_time_idx,
                    forecast_time_idx=forecast_time_idx,
                    num_trend_coefficients=num_trend_coefficients,
                    num_seasonal_coefficients=num_seasonal_coefficients,
                    num_generic_coefficients=num_generic_coefficients,
                    share_coefficients=share_coefficients)

                backcast = Subtract()([x, backcast_block])
                forecast = forecast_block

            else:

                if share_weights:

                    h = d1(backcast)
                    h = d2(h)
                    h = d3(h)
                    h = d4(h)

                else:

                    h = Dense(units=hidden_units, activation='relu')(backcast)
                    h = Dense(units=hidden_units, activation='relu')(h)
                    h = Dense(units=hidden_units, activation='relu')(h)
                    h = Dense(units=hidden_units, activation='relu')(h)

                backcast_block, forecast_block = get_block_output(
                    stack_type=stacks[s],
                    dense_layers_output=h,
                    backcast_time_idx=backcast_time_idx,
                    forecast_time_idx=forecast_time_idx,
                    num_trend_coefficients=num_trend_coefficients,
                    num_seasonal_coefficients=num_seasonal_coefficients,
                    num_generic_coefficients=num_generic_coefficients,
                    share_coefficients=share_coefficients)

                backcast = Subtract()([backcast, backcast_block])
                forecast = Add()([forecast, forecast_block])

    return Model(x, [backcast, forecast])


class callback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('epoch: {}, loss: {:,.4f}, val_loss: {:,.4f}'.format(1 + epoch, logs['loss'], logs['val_loss']))
