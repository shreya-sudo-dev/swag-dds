import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/noise_results.csv")

plt.figure()
plt.plot(df["noise_level"], df["accuracy"], marker="o")

plt.xlabel("Noise Level")
plt.ylabel("Detection Accuracy")
plt.title("Gunshot Detection Accuracy vs Noise Level")

plt.grid(True)

plt.show()