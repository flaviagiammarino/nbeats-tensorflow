import pandas as pd
import numpy as np
import tensorflow as tf
pd.options.mode.chained_assignment = None

from nbeats_tensorflow.utils import cast_target_to_array, get_training_sequences, get_time_indices
from nbeats_tensorflow.blocks import trend_block, seasonality_block, generic_block

class NBeats():

    def __init__(self,
                 y,
                 forecast_period,
                 lookback_period,
                 num_trend_coefficients=3,
                 num_seasonal_coefficients=5,
                 num_generic_coefficients=7,
                 units=100,
                 stacks=['trend', 'seasonality'],
                 num_blocks_per_stack=3,
                 share_weights=True,
                 share_coefficients=True):

        '''
        Implementation of univariate time series forecasting model introduced in Oreshkin, B. N., Carpov, D.,
        Chapados, N. and Bengio, Y., 2019. N-BEATS: Neural basis expansion analysis for interpretable time
        series forecasting. In International Conference on Learning Representations.

        Parameters:
        __________________________________
        y: np.array, pd.Series, list.
            Time series.

        forecast_period: int.
            Length of forecast period.

        lookback_period: int.
            Length of lookback period.

        num_trend_coefficients: int.
            Number of basis expansion coefficients of the trend block. This is the number of polynomial terms
            used for modelling the trend component. Only used when the model includes a trend stack.

        num_seasonal_coefficients: int.
            Number of basis expansion coefficients of the seasonality block. This is the number of Fourier terms
            used for modelling the seasonal component. Only used when the model includes a seasonality stack.

        num_generic_coefficients: int.
            Number of basis expansion coefficients of the generic block. This is the number of linear terms used
            for modelling the generic component. Only used when the model includes a generic stack.

        units: int.
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
            True if the forecast and backcast of each block should share the same basis expansion coefficients,
            False otherwise.
        '''

        # Cast the data to numpy array.
        y = cast_target_to_array(y)

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

        # Build the model.
        self.model = build_fn(
            backcast_time_idx,
            forecast_time_idx,
            num_trend_coefficients,
            num_seasonal_coefficients,
            num_generic_coefficients,
            units,
            stacks,
            num_blocks_per_stack,
            share_weights,
            share_coefficients)

    def fit(self,
            loss='mse',
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            validation_split=0.2,
            backcast_loss_weight=0.5,
            verbose=True):

        '''
        Train the model.

        Parameters:
        __________________________________
        loss: str, function.
            Loss function, see https://www.tensorflow.org/api_docs/python/tf/keras/losses.

        learning_rate: float.
            Learning rate.

        batch_size: int.
            Batch size.

        epochs: int.
            Number of epochs.

        validation_split: float.
            Fraction of the training data to be used as validation data, must be between 0 and 1.

        backcast_loss_weight: float.
            Weight of backcast in comparison to forecast when calculating the loss, must be between 0 and 1.
            A weight of 0.5 means that forecast and backcast loss is weighted the same.

        verbose: bool.
            True if the training history should be printed in the console, False otherwise.
        '''

        if backcast_loss_weight < 0 or backcast_loss_weight > 1:
            raise ValueError('The backcast loss weight must be between zero and one.')

        # Compile the model.
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=[loss, loss],
            loss_weights=[backcast_loss_weight, 1 - backcast_loss_weight])

        # Fit the model.
        self.history = self.model.fit(
            x=self.X,
            y=[self.X, self.Y],
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
            callbacks=[callback()] if verbose else None)

    def forecast(self, y, return_backcast=False):

        '''
        Generate the forecasts and backcasts.

        Parameters:
        __________________________________
        y: np.array, pd.Series, list.
            Past values of the time series.
            
        return_backcast: bool.
            True if the output should include the backcasts, False otherwise.

        Returns:
        __________________________________
        df: pd.DataFrame.
            Data frame with the actual values of the time series, forecasts and backcasts.
        '''

        # Cast the data to numpy array.
        y = cast_target_to_array(y)

        # Scale the data.
        y = (y - self.y_min) / (self.y_max - self.y_min)
        
        # Generate the forecasts and backcasts.
        backcast, forecast = self.model(y[- self.lookback_period:].reshape(1, - 1))
        backcast = self.y_min + (self.y_max - self.y_min) * backcast[- 1, :].numpy().flatten()
        forecast = self.y_min + (self.y_max - self.y_min) * forecast[- 1, :].numpy().flatten()

        # Organize the forecasts and backcasts in a data frame.
        df = pd.DataFrame(columns=['time_idx', 'actual', 'forecast'])
        df['time_idx'] = np.arange(len(self.y) + self.forecast_period)
        df['actual'].iloc[: - self.forecast_period] = self.y_min + (self.y_max - self.y_min) * self.y
        df['forecast'].iloc[- self.forecast_period:] = forecast

        if return_backcast:
            df['backcast'] = np.nan
            df['backcast'].iloc[- (self.lookback_period + self.forecast_period): - self.forecast_period] = backcast

        # Return the data frame.
        return df.astype(float)


def get_block_output(stack_type,
                     dense_layers_output,
                     backcast_time_idx,
                     forecast_time_idx,
                     num_trend_coefficients,
                     num_seasonal_coefficients,
                     num_generic_coefficients,
                     share_coefficients):

    '''
    Generate the block backcast and forecast.

    Parameters:
    __________________________________
    stack_type: str.
        The stack type, either 'trend', 'seasonality' or 'generic'.

    dense_layers_output: tf.Tensor.
        Output of 4-layer fully connected stack, 2-dimensional tensor with shape (N, k) where
        N is the batch size and k is the number of hidden units of each fully connected layer.
        Note that all fully connected layers have the same number of units.

    backcast_time_idx: tf.Tensor.
        Input time index, 1-dimensional tensor with length t used for generating the backcast.

    forecast_time_idx: tf.Tensor.
        Output time index, 1-dimensional tensor with length H used for generating the forecast.

    num_trend_coefficients: int.
        Number of basis expansion coefficients of the trend block. This is the number of polynomial terms used for
        modelling the trend component.

    num_seasonal_coefficients: int.
        Number of basis expansion coefficients of the seasonality block. This is the number of Fourier terms used
        for modelling the seasonal component.

    num_generic_coefficients: int.
        Number of basis expansion coefficients of the generic block. This is the number of linear terms used for
        modelling the generic component.

    share_coefficients: bool.
        True if the block forecast and backcast should share the same basis expansion coefficients, False otherwise.
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


def build_fn(backcast_time_idx,
             forecast_time_idx,
             num_trend_coefficients,
             num_seasonal_coefficients,
             num_generic_coefficients,
             units,
             stacks,
             num_blocks_per_stack,
             share_weights,
             share_coefficients):

    '''
    Build the model.

    Parameters:
    __________________________________
    backcast_time_idx: tf.Tensor.
        Input time index, 1-dimensional tensor with length t used for generating the backcast.

    forecast_time_idx: tf.Tensor.
        Output time index, 1-dimensional tensor with length H used for generating the forecast.

    num_trend_coefficients: int.
        Number of basis expansion coefficients of the trend block, corresponds to the number of polynomial terms.

    num_seasonal_coefficients: int.
        Number of basis expansion coefficients of the seasonality block, corresponds to the number of Fourier terms.

    num_generic_coefficients: int.
        Number of basis expansion coefficients of the generic block, corresponds to the number of linear terms.

    units: int.
        Number of hidden units of each of the 4 layers of the fully connected stack. Note that all fully connected
        layers have the same number of units.

    stacks: list of strings.
        The length of the list is the number of stacks, the items in the list are strings identifying the stack
        types (either 'trend', 'seasonality' or 'generic').

    num_blocks_per_stack: int.
        The number of blocks in each stack.

    share_weights: bool.
        True if the weights of the 4 layers of the fully connected stack should be shared by the different blocks
        inside the same stack, False otherwise.

    share_coefficients: bool.
        True if the block forecast and backcast should share the same basis expansion coefficients, False otherwise.
    '''

    # Define the model input, the input shape is
    # equal to the length of the lookback period.
    x = tf.keras.layers.Input(shape=len(backcast_time_idx))

    # Loop across the different stacks.
    for s in range(len(stacks)):

        # If share_weights is true, use the same 4 fully connected
        # layers across all blocks in the stack.
        if share_weights:

            d1 = tf.keras.layers.Dense(units=units, activation='relu')
            d2 = tf.keras.layers.Dense(units=units, activation='relu')
            d3 = tf.keras.layers.Dense(units=units, activation='relu')
            d4 = tf.keras.layers.Dense(units=units, activation='relu')

        # Loop across the different blocks in the stack.
        for b in range(num_blocks_per_stack):

            if s == 0 and b == 0:

                # For the first block of the first stack, forward pass the
                # input directly through the 4 fully connected layers.
                if share_weights:

                    h = d1(x)
                    h = d2(h)
                    h = d3(h)
                    h = d4(h)

                else:

                    h = tf.keras.layers.Dense(units=units, activation='relu')(x)
                    h = tf.keras.layers.Dense(units=units, activation='relu')(h)
                    h = tf.keras.layers.Dense(units=units, activation='relu')(h)
                    h = tf.keras.layers.Dense(units=units, activation='relu')(h)

                # Generate the block backcast and forecast.
                backcast_block, forecast_block = get_block_output(
                    stack_type=stacks[s],
                    dense_layers_output=h,
                    backcast_time_idx=backcast_time_idx,
                    forecast_time_idx=forecast_time_idx,
                    num_trend_coefficients=num_trend_coefficients,
                    num_seasonal_coefficients=num_seasonal_coefficients,
                    num_generic_coefficients=num_generic_coefficients,
                    share_coefficients=share_coefficients)

                # Calculate the backcast residual by subtracting the block backcast from the input.
                # See Section 3.2 in the N-BEATS paper.
                backcast = tf.keras.layers.Subtract()([x, backcast_block])

                # For the first block of the first stack, no adjustment is applied to the forecast.
                forecast = forecast_block

            else:

                # For the subsequent blocks and stacks, forward pass the
                # backcast residual through the 4 fully connected layers.
                if share_weights:

                    h = d1(backcast)
                    h = d2(h)
                    h = d3(h)
                    h = d4(h)

                else:

                    h = tf.keras.layers.Dense(units=units, activation='relu')(backcast)
                    h = tf.keras.layers.Dense(units=units, activation='relu')(h)
                    h = tf.keras.layers.Dense(units=units, activation='relu')(h)
                    h = tf.keras.layers.Dense(units=units, activation='relu')(h)

                # Generate the block backcast and forecast.
                backcast_block, forecast_block = get_block_output(
                    stack_type=stacks[s],
                    dense_layers_output=h,
                    backcast_time_idx=backcast_time_idx,
                    forecast_time_idx=forecast_time_idx,
                    num_trend_coefficients=num_trend_coefficients,
                    num_seasonal_coefficients=num_seasonal_coefficients,
                    num_generic_coefficients=num_generic_coefficients,
                    share_coefficients=share_coefficients)

                # Substract the current block backcast from the previous block backcast.
                # See Section 3.2 in the N-BEATS paper.
                backcast = tf.keras.layers.Subtract()([backcast, backcast_block])

                # Add the current block forecast to the previous block forecast.
                # See Section 3.2 in the N-BEATS paper.
                forecast = tf.keras.layers.Add()([forecast, forecast_block])

    return tf.keras.models.Model(x, [backcast, forecast])


class callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if 'val_loss' in logs.keys():
            print('epoch: {}, loss: {:,.6f}, val_loss: {:,.6f}'.format(1 + epoch, logs['loss'], logs['val_loss']))
        else:
            print('epoch: {}, loss: {:,.6f}'.format(1 + epoch, logs['loss']))
