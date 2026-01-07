# SWAG-DDS (Shockwave Acoustic Gunfire - Direction Detection System)

SWAG-DDS is a hybrid C++/Python acoustic surveillance system designed to detect gunshot signatures and estimate Direction of Arrival (DoA) using a 3-microphone array.

It utilizes a **C++ Acceleration Layer** for heavy signal processing (GCC-PHAT, FFT) and a **Python Intelligence Layer** for spectral classification and User Interface.

## Project Structure

* `core/`: C++ DSP Engine (Buffer management, FFT, GCC-PHAT).
* `ai/`: Python Classifier (Energy, ZCR, Spectral Centroid logic).
* `ui/`: Dashboard (Curses) and Simulator (PyQt5).
* `simulation/`: Tools to generate synthetic 3-mic wav scenarios.

## Installation

### Prerequisites
* Python 3.8+
* C++ Compiler (GCC, Clang, or MSVC)

### Build
1.  **Clone the repository**
2.  **Install dependencies and compile the C++ core:**
    ```bash
    pip install -r requirements.txt
    pip install .
    ```
    *Note: The `pip install .` command runs `setup.py`, which compiles the `core` C++ module and binds it to Python.*

## Usage

### 1. Generate Simulation Data
First, create the synthetic acoustic scenario (Gunshots at specific angles):
```bash
python simulation/generate_scenario.py