import numpy as np
from config import FFT_SIZE, SAMPLE_RATE, NUM_BANDS, WINDOW

if WINDOW == 'hann':
    window = np.hanning(FFT_SIZE)

def stft_band_energy(frame):
    spectrum = np.fft.rfft(frame * window)
    mag = np.abs(spectrum)
    freqs = np.fft.rfftfreq(FFT_SIZE, 1/SAMPLE_RATE)
    band_edges = np.geomspace(20, SAMPLE_RATE/2, NUM_BANDS+1)

    band_env = []
    for i in range(NUM_BANDS):
        mask = (freqs >= band_edges[i]) & (freqs < band_edges[i+1])
        if np.any(mask):
            band_env.append(np.mean(mag[mask]))
        else:
            band_env.append(0.0)

    band_env = np.array(band_env)
    if np.max(band_env) > 0:
        band_env /= np.max(band_env)
    return band_env

def compute_dynamic_gain(band_env, long_term_avg, alpha=0.01, min_gain=0.5, max_gain=3.0):
    long_term_avg = (1 - alpha) * long_term_avg + alpha * band_env
    band_gain = 1 / (long_term_avg + 1e-6)
    band_gain = np.clip(band_gain, min_gain, max_gain)
    return band_gain, long_term_avg
