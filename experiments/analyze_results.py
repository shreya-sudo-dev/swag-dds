import pandas as pd
import matplotlib.pyplot as plt

# Detection accuracy
df = pd.read_csv("results/detection_results.csv")

acc = (df.ground_truth == df.prediction).mean()

print("Detection Accuracy:", acc)

# Localization error
loc = pd.read_csv("results/localization_results.csv")

plt.figure()
plt.plot(loc.true_angle, loc.error)
plt.title("Localization Error vs Angle")
plt.xlabel("True Angle")
plt.ylabel("Error")
plt.show()

# Noise robustness
noise = pd.read_csv("results/noise_results.csv")

plt.figure()
plt.plot(noise.noise_level, noise.accuracy)
plt.title("Accuracy vs Noise")
plt.xlabel("Noise Level")
plt.ylabel("Accuracy")
plt.show()