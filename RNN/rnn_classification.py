from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Embedding, Reshape
from keras.layers import Dense, LSTM, Embedding, Reshape

class DeepANN:
    def rnn_model(self, input_shape):
        model = Sequential()
       # model.add(Reshape((input_shape[0] * input_shape[1], input_shape[2])))
        model.add(Reshape((input_shape[0] * input_shape[1], input_shape[2]), input_shape=input_shape))
        model.add(SimpleRNN(units=64, activation='relu'))
        model.add(Dense(units=7, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model