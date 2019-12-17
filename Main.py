import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib
import sklearn.metrics
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

data = pd.read_csv('PCLN.csv', parse_dates=True)

openp = data['Open'].to_numpy()
highp = data['High'].to_numpy()
lowp = data['Low'].to_numpy()
closep = data['Close'].to_numpy()
volume = data['Volume'].to_numpy()

data_training = data[data['Date'] < '2018-01-01'].copy()
data_test = data[data['Date'] >= '2018-01-01'].copy()

training_data = data_training.drop(['Date'], axis=1)

scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)

X_train = []
Y_train = []
for i in range(60, training_data.shape[0]):  # khoảng từ 60 đến số dòng của data
    X_train.append(training_data[i - 60:i,0:5])  # 1 phần tử của mảng X_train gồm dữ liệu của 60 dòng, lấy cột từ 0 tới 4
    Y_train.append(training_data[i, 5])  # 1 phần tử của mảng Y_train gồm dữ liệu của 1 dòng ở cột thứ 5 (Result)

X_train, Y_train = np.array(X_train), np.array(Y_train)

past_60_days = data_training.tail(60)
df = past_60_days.append(data_test, ignore_index=True)
df = df.drop(['Date'], axis=1)

inputs = scaler.transform(df)
X_test = []
Y_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i - 60:i,0:5])
    Y_test.append(inputs[i, 5])

X_test, Y_test = np.array(X_test), np.array(Y_test)

regressior = Sequential()
regressior.add(LSTM(units=150, activation='tanh', return_sequences=True))
regressior.add(Dropout(0.2))

regressior.add(LSTM(units=150, activation='tanh'))
regressior.add(Dropout(0.2))

regressior.add(Dense(1, activation='sigmoid'))

regressior.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
regressior.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size=64)

scores = regressior.evaluate(X_test, Y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))

# Y_pred = regressior.predict(X_test)
