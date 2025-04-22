import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


fs = 8000
ts = 1 / fs

sinPeriodsCount = 3
fSin = 100

samplesCountOneSin = round(fs / fSin)

silentSamplesCount = 1000
repeatCounts = 3

silentValue = np.zeros(silentSamplesCount)

sinTime = np.linspace(0, (samplesCountOneSin * sinPeriodsCount) / fs, samplesCountOneSin * sinPeriodsCount)
ySin = np.sin(2 * fSin * np.pi * sinTime)

startSilentSamplesCount = 1000

concatValue = np.concat((ySin, silentValue))
y = np.tile(concatValue, repeatCounts)

y = np.concat((np.zeros(silentSamplesCount), y))

samplesCount = y.size
duration = samplesCount / fs
time = np.linspace(0, duration, samplesCount)

noise = np.random.normal(0, 0.1, len(y))
yWithNoise = y + noise

correlate = signal.convolve(yWithNoise, np.flip(ySin))
# correlate = np.correlate(yWithNoise, ySin)
correlate = correlate / correlate.max()

correlateTime = np.linspace(0, correlate.size / fs, correlate.size)

peaks, _ = signal.find_peaks(correlate, width=1)
# peaks = signal.find_peaks_cwt(correlate, widths=np.arange(1, ySin.size))
# peaks = signal.argrelextrema(
#     np.array(correlate),
#     comparator=np.greater,order=2
# )
# peaks, _ = signal.find_peaks(correlate, height=1, distance=1)

estimatedSignals = np.zeros(y.size)

filteredPeaks = peaks[correlate[peaks] > correlate[peaks].max() * 0.7]


for peak in filteredPeaks:
    # if ((peak < ySin.size) | (peak + ySin.size > estimatedSignals.size)): continue
    startIndex = np.clip(peak - ySin.size + 1, 0, estimatedSignals.size)
    stopIndex = np.clip(peak, 0, estimatedSignals.size)
    estimatedSignals[startIndex:stopIndex] = ySin[:stopIndex-startIndex]


plt.grid(True)
plt.plot(time, yWithNoise, label="withNoise")
plt.plot(time, y, label="originalSignal")
plt.plot(time, estimatedSignals, label="estimatedSignal")
plt.plot(correlateTime, correlate, label="correlate")
plt.plot(correlateTime[filteredPeaks], correlate[filteredPeaks], "x", label="filteredPeaks")
plt.plot(correlateTime[peaks], correlate[peaks], "x", label="peaks")
plt.plot(time, y - estimatedSignals, label="original-estimated")
plt.legend()
plt.show()