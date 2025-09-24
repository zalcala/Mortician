# Mortician

**Mortician** is a real-time audio spectrum analyzer and UDP streaming tool written in Python. It captures live audio from your system (using a loopback device), analyzes the frequency spectrum, and streams the results over UDP for external visualization or debugging.

---

## Features

- **Real-time audio capture** from system output (loopback device required)
- **Short-Time Fourier Transform (STFT)** on overlapping frames
- **Perceptually/logarithmically spaced frequency bands** (configurable)
- **Feature normalization** for consistent visualization
- **UDP streaming** of binary-packed feature frames
- **Modular DSP and network code** for easy extension
- **Debug UDP listener** for quick verification of output

---

## Project Structure

- `main.py` — Main script: audio capture, analysis, and UDP streaming
- `config.py` — Central configuration for DSP and network parameters
- `dsp.py` — DSP utilities: band energy extraction, gain smoothing
- `network.py` — UDP socket setup and frame sending
- `audio_stream.py` — Entry point for starting the audio stream
- `udp_debug_listener.py` — Simple UDP server for debugging/visualization

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd Mortician
   ```
2. **Set up a virtual environment (optional but recommended):**
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```sh
   pip install sounddevice numpy
   ```

---

## Usage

### 1. Start the Analyzer

Edit `main.py` or `audio_stream.py` to set the correct audio device index for your system's loopback device.

Run the analyzer:
```sh
python main.py
```

### 2. Listen for UDP Frames (Debug)

In a separate terminal, run the UDP debug listener:
```sh
python udp_debug_listener.py
```
You should see incoming frames printed to the console.

---

## Configuration

All key parameters (sample rate, FFT size, number of bands, UDP IP/port, etc.) are set in `config.py`.

---

## Notes
- **Loopback device required:** You must have a system audio loopback device enabled to capture output audio.
- **Visualization:** The UDP output is suitable for use with external visualization tools (e.g., Processing, custom apps).
- **Extensible:** Modular code structure makes it easy to add new DSP features or change the network protocol.

---

## License

This project is licensed under the GNU General Public License (GPL). See the LICENSE file for details.
