import numpy as np
import os
from ai.classifier import GunshotClassifier

clf = GunshotClassifier()

scores_gun = []
scores_noise = []

gun_dir = "dataset/gunshot"
noise_dir = "dataset/noise"

for file in os.listdir(gun_dir):
    sig = np.load(os.path.join(gun_dir,file))
    score = clf.compute_score(sig)
    scores_gun.append(score)

for file in os.listdir(noise_dir):
    sig = np.load(os.path.join(noise_dir,file))
    score = clf.compute_score(sig)
    scores_noise.append(score)

print("Gunshot scores:")
print(min(scores_gun), max(scores_gun))

print("Noise scores:")
print(min(scores_noise), max(scores_noise))