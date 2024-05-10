import tensorflow as tf
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM, SimpleRNN, Dropout
from keras.callbacks import LambdaCallback

import wandb

wandb.init()
config = wandb.config

# If repeated prediction is True, the green line in the wandb plot will correspond to
# using the past prediction as input to the next prediction (hard case).
# If repeated prediction is False, the green line in the wandb plot will correspond
# to make in a prediction off of ground truth data every time.
config.repeated_predictions = False
config.look_back = 20

df = pd.read_csv('daily-min-temperatures.csv')
data = df.Temp.astype('float32').values

print(data)

# convert an array of values into a dataset matrix
def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset)-config.look_back-1):
        a = dataset[i:(i+config.look_back)]
        dataX.append(a)
        dataY.append(dataset[i + config.look_back])
    return np.array(dataX), np.array(dataY)


split = int(len(data) * 0.70)
train = data[:split]
test = data[split:]

trainX, trainY = create_dataset(train)
testX, testY = create_dataset(test)


trainX = trainX[:, :, np.newaxis]
testX = testX[:, :, np.newaxis]


model = Sequential()
model.add(SimpleRNN(1, input_shape=(config.look_back,1 )))

model.compile(loss='mae', optimizer='adam', metrics=['mae'])
model.fit(trainX, trainY, epochs=3, batch_size=1, validation_data=(testX, testY))

