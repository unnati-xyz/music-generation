from keras.models import Sequential
from keras.layers.core import TimeDistributedDense
# In TimeDistributedDense we apply the same dense layer for each time dimension. It's used when you want the entire O/P sequence
from keras.layers.recurrent import LSTM


def create_lstm_network(num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=1):
    model = Sequential()  # Sequential is a linear stack of layers
    # This layer converts frequency space to hidden space
    model.add(TimeDistributedDense(input_dim=num_frequency_dimensions, output_dim=num_hidden_dimensions))
    for cur_unit in range(num_recurrent_units):
        # return_sequences=True implies return the entire output sequence & not just the last output
        model.add(LSTM(input_dim=num_hidden_dimensions, output_dim=num_hidden_dimensions, return_sequences=True))
    # This layer converts hidden space back to frequency space
    model.add(TimeDistributedDense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions))

    # Done building the model.Now, configure it for the learning process
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model
