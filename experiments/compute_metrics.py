import pandas as pd

df = pd.read_csv("results/localization_results.csv")

mae = df["error"].mean()

print("Mean Absolute Error:", mae)