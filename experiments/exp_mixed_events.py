import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
from sklearn.linear_model import LogisticRegression

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from experiments.extract_features import extract_features

FRAME = 4096

df = pd.read_csv("results/features.csv")

X = df[["energy","zcr","peak","centroid","bandwidth","decay"]]
y = df["label"]

model = LogisticRegression()
model.fit(X,y)


def generate_impulse_noise():
    x = np.zeros(FRAME)
    idx = np.random.randint(0,FRAME)
    x[idx] = np.random.uniform(0.5,1.0)
    x += 0.05*np.random.randn(FRAME)
    return x


def generate_gunshot():
    x = np.zeros(FRAME)
    x[FRAME//2] = 1
    x += 0.02*np.random.randn(FRAME)
    return x


correct = 0
total = 500

for _ in range(total):

    r = np.random.rand()

    if r < 0.4:
        sig = generate_gunshot()
        label = 1
    else:
        sig = generate_impulse_noise()
        label = 0

    feat = extract_features(sig)

    feat_df = pd.DataFrame([feat], columns=["energy","zcr","peak","centroid","bandwidth","decay"])
    pred = model.predict(feat_df)[0]

    if pred == label:
        correct += 1

print("Mixed Event Accuracy:", correct/total)