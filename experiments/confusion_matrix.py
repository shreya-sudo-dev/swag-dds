import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("results/detection_results.csv")

y_true = df["ground_truth"]
y_pred = df["prediction"]

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(cm, display_labels=["Noise","Gunshot"])
disp.plot()

plt.title("Gunshot Detection Confusion Matrix")
plt.show()