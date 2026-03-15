import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("results/features.csv")

X = df[["energy","zcr","peak","centroid"]]
y = df["label"]

model = LogisticRegression()
model.fit(X,y)

snr_levels = [20,15,10,5,0,-5]

accs = []

for snr in snr_levels:

    noise = np.random.randn(len(X),4)

    scale = 10**(-snr/20)

    X_noisy = X + scale*noise

    pred = model.predict(X_noisy)

    acc = accuracy_score(y,pred)

    accs.append(acc)

plt.plot(snr_levels,accs,marker='o')
plt.xlabel("SNR (dB)")
plt.ylabel("Detection Accuracy")
plt.title("Gunshot Detection vs SNR")

plt.gca().invert_xaxis()

plt.show()