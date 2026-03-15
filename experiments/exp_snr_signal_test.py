import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pandas as pd

FS = 44100
FRAME = 4096

def extract_features(audio):

    max_val = np.max(np.abs(audio)) + 1e-6
    audio = audio / max_val

    energy = np.sum(audio**2)
    zcr = ((audio[:-1] * audio[1:]) < 0).sum()
    peak = np.max(np.abs(audio))

    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.linspace(0,FS/2,len(spectrum))

    centroid = np.sum(freqs*spectrum)/(np.sum(spectrum)+1e-6)
    centroid_norm = centroid/(FS/2)

    return [energy,zcr,peak,centroid_norm]


# load training data
df = pd.read_csv("results/features.csv")

X = df[["energy","zcr","peak","centroid"]]
y = df["label"]

model = LogisticRegression()
model.fit(X,y)


def generate_gunshot():
    x = np.zeros(FRAME)
    x[FRAME//2] = 1
    x += 0.02*np.random.randn(FRAME)
    return x


snr_levels = [20,15,10,5,0,-5]

accs = []

for snr in snr_levels:

    correct = 0
    total = 200

    for _ in range(total):

        sig = generate_gunshot()

        signal_power = np.mean(sig**2)
        noise_power = signal_power/(10**(snr/10))

        noise = np.random.randn(FRAME)*np.sqrt(noise_power)

        noisy = sig + noise

        feat = extract_features(noisy)

        pred = model.predict([feat])[0]

        if pred == 1:
            correct += 1

    accs.append(correct/total)


plt.plot(snr_levels,accs,marker='o')
plt.xlabel("SNR (dB)")
plt.ylabel("Detection Accuracy")
plt.title("Gunshot Detection vs SNR")

plt.gca().invert_xaxis()

plt.grid(True)
plt.show()