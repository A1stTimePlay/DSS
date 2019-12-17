# Binary Classification with Sonar Dataset: Baseline
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import talib
import numpy as np

# load dataset
df_data = read_csv("VTR_data.csv", header=None)
np_data = df_data.values


df_result = read_csv("VTR.result.csv", header=None)
np_result = df_result.values

openp = np_data[:, 0]
highp = np_data[:, 1]
lowp = np_data[:, 2]
closep = np_data[:, 3]
volume = np_data[:, 4]

WMA = talib.WMA(closep, timeperiod=20)
SMA = talib.SMA(closep, timeperiod=20)
EMA = talib.SMA(closep, timeperiod=20)
OBV = talib.OBV(closep, volume)
RSI = talib.RSI(closep, timeperiod=20)

# np_data = np.concatenate((np_data, WMA[:, None]), axis=1)
# np_data = np.concatenate((np_data, SMA[:, None]), axis=1)
# np_data = np.concatenate((np_data, EMA[:, None]), axis=1)
np_data = np.concatenate((np_data, OBV[:, None]), axis=1)
# np_data = np.concatenate((np_data, RSI[:, None]), axis=1)

# split into input (X) and output (Y) variables
X = np_data[:, 0:6].astype(float)
Y = np_result[:, 0]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

X = X.reshape(1259, 1, 6)

# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(LSTM(200, input_shape=(1, 6), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=2, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))