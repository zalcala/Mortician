# Mortician

**Mortician** is a real-time audio spectrum analyzer and UDP streaming tool written in Python. It captures live audio from your system (using a loopback device), analyzes the frequency spectrum, and streams the results over UDP for external visualization or debugging.

---

## Features

- **Real-time audio capture** from system output (loopback device required)
- **Short-Time Fourier Transform (STFT)** on overlapping frames
- **Perceptually/logarithmically spaced frequency bands** (configurable)
- **Feature normalization** for consistent visualization
- **UDP streaming** of binary-packed feature frames
- **Highly modular DSP package** for easy extension and experimentation
- **Debug UDP listener** for quick verification of output

---

## Project Structure

- `main.py` — Main script: audio capture, analysis, and UDP streaming
- `config.py` — Central configuration for DSP and network parameters (reads from `.env`)
- `dsp/` — Modular DSP package:
  - `windowing.py`: Window function utilities
  - `bands.py`: Band edge calculation and mapping
  - `envelope.py`: Envelope smoothing and dynamic gain
  - `features.py`: Feature extraction (STFT, global features)
  - `utils.py`: Shared helpers (placeholder)
  - `__init__.py`: Main DSP API
- `network.py` — UDP socket setup and frame sending
- `utils/` — Utility scripts:
  - `list_devices.py`: List available audio devices
  - `udp_debug_listener.py`: Simple UDP server for debugging/visualization

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd Mortician
   ```
2. **Set up a virtual environment (recommended):**
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

---

## Usage

### 0. Find Your Loopback Device Index

Before running the analyzer, list available audio devices to find your system's loopback device index:

```sh
python utils/list_devices.py
```

Look for the device with the appropriate name and note its index. Use this index in `main.py`.

### 1. Start the Analyzer

Edit `main.py` to set the correct audio device index for your system's loopback device.

Run the analyzer:
```sh
python main.py
```

### 2. Listen for UDP Frames (Debug)

In a separate terminal, run the UDP debug listener:
```sh
python utils/udp_debug_listener.py
```
You should see incoming frames printed to the console.

---

## Configuration

All key parameters (sample rate, FFT size, number of bands, UDP IP/port, etc.) are set in `config.py`, which loads values from a `.env` file in the project root. Edit `.env` to customize your setup.

### Gain and Smoothing Parameters in `.env`

- `GAIN_ALPHA`: Smoothing factor for the moving average of each band's energy. Lower = slower, more stable gain adaptation. Higher = more responsive, less stable.
- `GAIN_MIN`: Minimum allowed gain for any band. Prevents excessive boosting of quiet bands (can reduce noise).
- `GAIN_MAX`: Maximum allowed gain for any band. Prevents excessive attenuation/boosting of loud/quiet bands.
- `SMA_WINDOW`: Number of frames used for the simple moving average (SMA) of each band's energy. Higher = smoother, more historical average. Lower = more responsive to recent changes.

Example `.env` section:
```
GAIN_ALPHA=0.01
GAIN_MIN=0.5
GAIN_MAX=3.0
SMA_WINDOW=50
```

---

## Notes
- **Loopback device required:** You must have a system audio loopback device enabled to capture output audio.
- **Visualization:** The UDP output is suitable for use with external visualization tools (e.g., Processing, custom apps).
- **Extensible:** Modular code structure makes it easy to add new DSP features or change the network protocol.

---

## License

This project is licensed under the GNU General Public License (GPL). See the LICENSE file for details.