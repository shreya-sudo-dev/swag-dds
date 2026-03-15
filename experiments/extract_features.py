import numpy as np
import os
import csv

FS = 44100

def extract_features(audio):

    FS = 44100

    max_val = np.max(np.abs(audio)) + 1e-6
    audio = audio.astype(np.float32) / max_val

    energy = np.sum(audio**2)
    zcr = ((audio[:-1] * audio[1:]) < 0).sum()
    peak = np.max(np.abs(audio))

    spectrum = np.abs(np.fft.rfft(audio))
    frequencies = np.linspace(0, FS/2, len(spectrum))

    sum_mag = np.sum(spectrum) + 1e-6
    centroid = np.sum(frequencies * spectrum) / sum_mag
    centroid_norm = centroid / (FS/2)

    # NEW FEATURE
    bandwidth = np.sqrt(
        np.sum(((frequencies - centroid)**2) * spectrum) / sum_mag
    )
    bandwidth_norm = bandwidth / (FS/2)

    # temporal decay feature
    center = len(audio) // 2
    tail = audio[center:]

    decay_energy = np.sum(tail**2)
    head_energy = np.sum(audio[:center]**2) + 1e-6

    decay_ratio = decay_energy / head_energy

    return [energy, zcr, peak, centroid_norm, bandwidth_norm, decay_ratio]

rows = []

for file in os.listdir("dataset/gunshot"):
    sig = np.load("dataset/gunshot/" + file)
    feat = extract_features(sig)
    rows.append(feat + [1])

for file in os.listdir("dataset/noise"):
    sig = np.load("dataset/noise/" + file)
    feat = extract_features(sig)
    rows.append(feat + [0])

with open("results/features.csv","w") as f:
    writer = csv.writer(f)
    writer.writerow(["energy","zcr","peak","centroid","bandwidth","decay","label"])
    writer.writerows(rows)

print("Feature dataset created")