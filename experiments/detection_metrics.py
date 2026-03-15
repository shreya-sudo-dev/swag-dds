import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

df = pd.read_csv("results/detection_results.csv")

y_true = df["ground_truth"]
y_pred = df["prediction"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Detection Metrics")
print("------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)