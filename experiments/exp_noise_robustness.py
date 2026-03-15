import numpy as np
import csv
from ai.classifier import GunshotClassifier

clf = GunshotClassifier()

noise_levels = [0.01,0.05,0.1,0.2,0.3]

results = []

def generate_gunshot(n=4096):
    x = np.zeros(n)
    x[n//2] = 1
    return x.astype(np.float32)

for noise in noise_levels:

    correct = 0
    total = 100

    for i in range(total):

        signal = generate_gunshot()

        signal += noise*np.random.randn(len(signal))

        pred = clf.is_gunshot(signal)

        if pred:
            correct += 1

    accuracy = correct/total

    results.append([noise,accuracy])

with open("results/noise_results.csv","w") as f:
    writer = csv.writer(f)
    writer.writerow(["noise_level","accuracy"])
    writer.writerows(results)

print("Noise robustness experiment completed")