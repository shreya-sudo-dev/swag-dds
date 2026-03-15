import numpy as np
import os
import csv
from ai.classifier import GunshotClassifier

clf = GunshotClassifier()

results = []

gunshot_dir = "dataset/gunshot"
noise_dir = "dataset/noise"

for file in os.listdir(gunshot_dir):

    sig = np.load(os.path.join(gunshot_dir,file))
    pred = clf.is_gunshot(sig)

    results.append([1,int(pred)])

for file in os.listdir(noise_dir):

    sig = np.load(os.path.join(noise_dir,file))
    pred = clf.is_gunshot(sig)

    results.append([0,int(pred)])

with open("results/large_scale_results.csv","w") as f:

    writer = csv.writer(f)
    writer.writerow(["ground_truth","prediction"])
    writer.writerows(results)

print("Large scale evaluation complete")