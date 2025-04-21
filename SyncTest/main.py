import numpy as np
import matplotlib.pyplot as plt


fs = 8000
ts = 1 / fs

sinPeriodsCount = 3
fSin = 100

samplesCountOneSin = round(fs / fSin)
samplesCountSins = samplesCountOneSin * sinPeriodsCount
durationSins = (samplesCountSins) / fs

silentSamplesCount = 1000
repeatCounts = 3

silentValue = np.zeros(silentSamplesCount)

sinTime = np.linspace(0, durationSins, samplesCountSins)
ySin = np.sin(2 * fSin * np.pi * sinTime)

concatValue = np.concat((ySin, silentValue))
y = np.tile(concatValue, repeatCounts)
samplesCount = y.size
duration = samplesCount / fs
time = np.linspace(0, duration, samplesCount)

noise = np.random.normal(0, 0.01, len(y))
yWithNoise = y + noise


correlate = np.correlate(yWithNoise, ySin)
correlate = correlate / correlate.max()

correlateTime = np.linspace(0, correlate.size / fs, correlate.size)

plt.grid(True)
plt.plot(time, yWithNoise, label="withNoise")
plt.plot(time, y, label="originalSignal")
plt.plot(correlateTime, correlate, label="correlate")
plt.legend()
plt.show()