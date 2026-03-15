# SWAG-DDS (Shockwave Acoustic Gunfire - Direction Detection System)

SWAG-DDS is a hybrid C++/Python acoustic surveillance system designed to detect gunshot signatures and estimate Direction of Arrival (DoA) using a 3-microphone array.

It utilizes a **C++ Acceleration Layer** for heavy signal processing (GCC-PHAT, FFT) and a **Python Intelligence Layer** for spectral classification and User Interface.
     
## Processing Pipeline

* `Audio Input/`: Multi-channel stream capture.
* `Signal Buffering/`: Real-time windowing and Hann window application.
* `Feature Extraction/`: Calculation of temporal and spectral descriptors.
* `Gunshot Classification/`: ML-based binary discrimination.
* `GCC-PHAT Localization/`: Time Difference of Arrival (TDoA) estimation.
* `DoA Estimation/`: Mapping TDoA to angular coordinates.

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
```


### Experimental Results
Gunshot Detection Performance
* Accuracy - 1.00
* Precision - 1.00
* Recall - 1.00
* F1 Score - 1.00

### Localization & Real-Time Metrics
* Mean Direction Error - 2.105°
* Mean Latency - 3.97 ms
* Max Latency - 7.05 ms
* Min Latency - 3.67 ms

### 🔊 Robustness & Mixed Signal Analysis
The system was evaluated against composite acoustic environments to simulate urban clutter.

| Scenario | Signal Composition | Detection Accuracy |
| :--- | :--- | :--- |
| **Clean** | Gunshot only | 1.00 |
| **Urban Ambient** | Gunshot + Gaussian White Noise (0dB SNR) | 0.94 |
| **Impulsive Interference** | Gunshot + Door Slams/Construction | 0.61 |

*Note: Performance on impulsive noise is currently being optimized through Spectral Subtraction and Median Filtering in the DSP layer.*

🧪 Reproducibility
To reproduce the experimental results published in the paper, execute the research scripts in the following order:

```bash
python experiments/generate_dataset.py
python experiments/extract_features.py
python experiments/train_classifier.py
python experiments/compute_metrics.py
python experiments/exp_localization_accuracy.py
python experiments/exp_latency.py
```
### License and Author
Author: Shreya Sable
License: MIT
Acoustic Event Detection Research Project.
