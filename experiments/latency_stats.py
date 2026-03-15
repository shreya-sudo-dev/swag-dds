import pandas as pd

df = pd.read_csv("results/latency_results.csv")

mean_latency = df["latency"].mean()
max_latency = df["latency"].max()
min_latency = df["latency"].min()

print("Latency Statistics")
print("------------------")

print("Mean latency :", mean_latency, "seconds")
print("Max latency  :", max_latency, "seconds")
print("Min latency  :", min_latency, "seconds")

print("Mean latency (ms):", mean_latency * 1000)