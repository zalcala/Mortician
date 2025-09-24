"""
python_sender_audio.py

Real-time Audio Spectrum Analyzer (Python → UDP)
------------------------------------------------

This script:
1. Captures live audio from the system's output (loopback device required).
2. Performs Short-Time Fourier Transform (STFT) on overlapping frames.
3. Splits the spectrum into perceptually spaced frequency bands.
4. Normalizes features and assembles them into a binary frame.
5. Streams frames via UDP for external visualization (e.g., Processing).

Dependencies:
    pip install sounddevice numpy
"""

import socket, struct, time
import sounddevice as sd
import numpy as np

# ---------------------------
# Networking Setup (UDP Push)
# ---------------------------
UDP_IP = "127.0.0.1"   # Destination (localhost for testing)
UDP_PORT = 5005        # Port to send data to
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ---------------------------
# Audio Analysis Parameters
# ---------------------------
SAMPLE_RATE = 48000    # Matches project spec (Layer 1 choice)
FFT_SIZE    = 1024     # Window size (N)
HOP_SIZE    = FFT_SIZE // 2  # 50% overlap
NUM_BANDS   = 32       # Perceptual/log-spaced bands for visualization

# Precompute Hann window (applied before FFT to reduce spectral leakage)
window = np.hanning(FFT_SIZE)

# ---------------------------
# Helper: Band Energy Extractor
# ---------------------------
def stft_band_energy(frame):
    """
    Compute band energies from a single audio frame.
    
    Args:
        frame (np.ndarray): 1D mono audio buffer of length FFT_SIZE
    
    Returns:
        band_env (list): Normalized energy (0–1) per frequency band
    """
    # Apply Hann window
    frame_windowed = frame * window

    # FFT → complex spectrum
    spectrum = np.fft.rfft(frame_windowed)

    # Magnitude spectrum (real amplitudes)
    mag = np.abs(spectrum)

    # Frequency bins corresponding to FFT result
    freqs = np.fft.rfftfreq(FFT_SIZE, 1/SAMPLE_RATE)

    # Define perceptual/logarithmic band edges
    band_edges = np.geomspace(20, SAMPLE_RATE/2, NUM_BANDS+1)

    # Compute mean energy per band
    band_env = []
    for i in range(NUM_BANDS):
        mask = (freqs >= band_edges[i]) & (freqs < band_edges[i+1])
        if np.any(mask):
            band_env.append(np.mean(mag[mask]))
        else:
            band_env.append(0.0)

    # Normalize to [0,1] range for consistent visualization
    band_env = np.array(band_env)
    if np.max(band_env) > 0:
        band_env /= np.max(band_env)

    return band_env.tolist()

# ---------------------------
# Audio Callback
# ---------------------------
# Global buffer for overlapping frames
buffer = np.zeros(FFT_SIZE)
hop_size = FFT_SIZE // 2

def audio_callback(indata, frames, time_info, status):
    """
    Audio callback with rolling buffer to ensure FFT_SIZE samples
    and 50% overlap for STFT.
    """
    global buffer

    # Convert stereo → mono
    mono = np.mean(indata, axis=1)

    # Ensure mono is not longer than hop_size
    if len(mono) > hop_size:
        mono = mono[:hop_size]

    # Shift old samples left by hop_size
    buffer[:FFT_SIZE - hop_size] = buffer[hop_size:]

    # Append new samples at the end
    buffer[FFT_SIZE - hop_size:] = mono

    # Full frame ready for STFT
    frame = buffer.copy()

    # ----------------------
    # DSP: Band Energy
    # ----------------------
    band_env = stft_band_energy(frame)

    # Placeholder per-band flux (frame-to-frame difference)
    # For real flux, you can maintain a previous band_env global variable
    band_flux = band_env.copy()

    # Placeholder per-band gain offsets
    band_gain = [1.0] * NUM_BANDS

    # Global features
    centroid  = float(np.mean(band_env))
    flux      = float(np.mean(np.diff(band_env))) if len(band_env) > 1 else 0
    loudness  = float(np.mean(np.square(frame)))
    transient = float(np.max(band_env) > 0.8)

    # Timestamp
    timestamp = time.time()

    # Construct frame according to schema
    frame_data = (
        [timestamp, float(NUM_BANDS)]
        + band_env
        + band_flux
        + band_gain
        + [centroid, flux, loudness, transient]
    )

    # Pack floats to binary and send via UDP
    frame_bytes = struct.pack(f'{len(frame_data)}f', *frame_data)
    sock.sendto(frame_bytes, (UDP_IP, UDP_PORT))


# ---------------------------
# Main Audio Stream
# ---------------------------
stream = sd.InputStream(
    device=0,         # 🔑 Must set this to your loopback device index
    channels=2,          # Capture stereo, mix down to mono
    samplerate=SAMPLE_RATE,
    blocksize=HOP_SIZE,  # Controls callback rate (~60 FPS with 50% overlap)
    callback=audio_callback
)

# Run forever (blocking loop)
print("🎵 Streaming live analyzer data over UDP...")
print("   Listening on device (loopback required).")
print("   Sending to {}:{}".format(UDP_IP, UDP_PORT))

with stream:
    while True:
        sd.sleep(1000)
