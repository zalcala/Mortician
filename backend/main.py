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
    pip install sounddevice numpy python-dotenv
"""

import socket
import struct
import time
import sounddevice as sd
import numpy as np
import os
from config import SAMPLE_RATE, FFT_SIZE, NUM_BANDS, DEVICE_INDEX
from dsp import EnvelopeSmoother, stft_band_energy, compute_global_features

# ---------------------------
# Networking Setup (UDP Push)
# ---------------------------
from dotenv import load_dotenv
load_dotenv()
UDP_IP = os.getenv('UDP_IP', '127.0.0.1')   # Destination (localhost for testing)
UDP_PORT = int(os.getenv('UDP_PORT', 5005))        # Port to send data to
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Send config packet with number of bands
config_msg = f"CONFIG:{NUM_BANDS}".encode()
sock.sendto(config_msg, (UDP_IP, UDP_PORT))
time.sleep(0.1)  # 100 ms delay to ensure CONFIG is received first

# ---------------------------
# Audio Analysis Parameters
# ---------------------------
HOP_SIZE = FFT_SIZE // 2

# Gain and smoothing parameters from .env
GAIN_ALPHA = float(os.getenv('GAIN_ALPHA', '0.01'))
GAIN_MIN = float(os.getenv('GAIN_MIN', '0.5'))
GAIN_MAX = float(os.getenv('GAIN_MAX', '3.0'))
SMA_WINDOW = int(os.getenv('SMA_WINDOW', '50'))  # Number of frames for the moving average

if 'band_buffers' not in globals():
    band_buffers = [ [0.0]*SMA_WINDOW for _ in range(NUM_BANDS) ]
    band_buffer_idx = 0

# ---------------------------
# Audio Callback
# ---------------------------
buffer = np.zeros(FFT_SIZE)
hop_size = HOP_SIZE
smoother = EnvelopeSmoother(num_bands=NUM_BANDS)
prev_band_env = np.zeros(NUM_BANDS)
long_term_avg = np.ones(NUM_BANDS) * 1e-6  # For dynamic gain


def audio_callback(indata, frames, time_info, status):
    """
    Audio callback with rolling buffer to ensure FFT_SIZE samples
    and 50% overlap for STFT.
    """
    global buffer, prev_band_env, band_buffers, band_buffer_idx

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

    # DSP: Band Energy
    raw_band_env = stft_band_energy(frame)
    band_env = smoother.process(raw_band_env)

    # Per-band flux (frame-to-frame difference)
    band_flux = np.abs(band_env - prev_band_env)
    prev_band_env = band_env.copy()

    # --- Simple Moving Average for Band Gain ---
    for i in range(NUM_BANDS):
        band_buffers[i][band_buffer_idx % SMA_WINDOW] = band_env[i]
    band_buffer_idx += 1
    long_term_avg = np.array([np.mean(band_buffers[i]) for i in range(NUM_BANDS)])

    band_gain = np.log1p(1.0 / (long_term_avg + 1e-6))
    band_gain = np.clip(band_gain, GAIN_MIN, GAIN_MAX)

    # Apply gain compensation to band energies
    compensated_band_env = band_env * band_gain

    # Global features
    centroid, flux, loudness, transient = compute_global_features(compensated_band_env, prev_band_env)

    # Timestamp
    timestamp = time.time()

    # Construct frame according to schema
    frame_data = (
        [timestamp, float(NUM_BANDS)]
        + compensated_band_env.tolist()
        + band_flux.tolist()
        + band_gain.tolist()
        + [centroid, flux, loudness, transient]
    )

    # Pack floats to binary and send via UDP
    frame_bytes = struct.pack(f'{len(frame_data)}f', *frame_data)
    sock.sendto(frame_bytes, (UDP_IP, UDP_PORT))

# ---------------------------
# Main Audio Stream
# ---------------------------
stream = sd.InputStream(
    device=DEVICE_INDEX,         # Set from config.py/.env
    channels=2,          # Capture stereo, mix down to mono
    samplerate=SAMPLE_RATE,
    blocksize=HOP_SIZE,  # Controls callback rate (~60 FPS with 50% overlap)
    callback=audio_callback
)

# Run forever (blocking loop)
print("🎵 Streaming live analyzer data over UDP...")
print("   Listening on device (loopback required).")
print(f"   Sending to {UDP_IP}:{UDP_PORT}")

with stream:
    while True:
        sd.sleep(1000)
