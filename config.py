# DSP Settings
SAMPLE_RATE = 48000
FFT_SIZE = 1024
HOP_SIZE = FFT_SIZE // 2
NUM_BANDS = 32
WINDOW = 'hann'  # or np.hanning(FFT_SIZE)

# Gain smoothing
SMOOTH_ALPHA = 0.01
GAIN_MIN = 0.5
GAIN_MAX = 3.0

# UDP Settings
UDP_IP = "127.0.0.1"
UDP_PORT = 5005