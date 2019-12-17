import numpy as np
import talib
import matplotlib.pyplot as plt

data = np.genfromtxt('sp500.csv', dtype=None, delimiter=',', names=True, encoding=None)

date = data['Date']
openp = data['Open']
highp = data['High']
lowp = data['Low']
closep = data['Close']
volume = data['Volume']


WMA = talib.WMA(closep, timeperiod=20)
SMA = talib.SMA(closep, timeperiod=20)
EMA = talib.SMA(closep, timeperiod=20)
OBV = talib.OBV(closep, np.asarray(volume, dtype='float'))
RSI = talib.RSI(closep, timeperiod=20)


# plt.plot(WMA)
# plt.plot(SMA)
# plt.plot(EMA)
# plt.plot(OBV)
# plt.plot(RSI)

# plt.plot(closep)
# plt.show()

