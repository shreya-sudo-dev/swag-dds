import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("results/large_scale_results.csv")

y_true = df["ground_truth"]
y_pred = df["prediction"]

print("Large Scale Evaluation")
print("----------------------")

print("Accuracy :", accuracy_score(y_true,y_pred))
print("Precision:", precision_score(y_true,y_pred))
print("Recall   :", recall_score(y_true,y_pred))
print("F1 Score :", f1_score(y_true,y_pred))