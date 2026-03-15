import numpy as np
import csv
from ai.classifier import GunshotClassifier

clf = GunshotClassifier()

N = 200
results = []

def generate_gunshot(n=4096):
    x = np.zeros(n)
    x[n//2] = 1
    x += 0.05*np.random.randn(n)
    return x.astype(np.float32)

def generate_noise(n=4096):
    return np.random.randn(n).astype(np.float32)

for i in range(N):

    if i < N//2:
        signal = generate_gunshot()
        label = 1
    else:
        signal = generate_noise()
        label = 0

    pred = clf.is_gunshot(signal)

    results.append([label, int(pred)])

with open("results/detection_results.csv","w") as f:
    writer = csv.writer(f)
    writer.writerow(["ground_truth","prediction"])
    writer.writerows(results)

print("Detection experiment completed")