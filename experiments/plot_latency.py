import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/latency_results.csv")

plt.hist(df["latency"]*1000, bins=20)

plt.xlabel("Latency (ms)")
plt.ylabel("Frequency")
plt.title("DSP Processing Latency Distribution")

plt.show()