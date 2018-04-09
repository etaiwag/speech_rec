
# Math
import numpy as np
from scipy import signal
from scipy.io import wavfile
import plotly.offline as py
py.init_notebook_mode(connected=True)


fs, data = wavfile.read('C:\\Users\\etaiwag\\Downloads\\train\\train\\audio\\five\\2aa787cf_nohash_1.wav')
print(fs)
print(data)
X = 2
print(X)


def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)


freqs, times, spectrogram = log_specgram(data, fs)
