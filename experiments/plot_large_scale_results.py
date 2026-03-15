import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("results/large_scale_results.csv")

cm = confusion_matrix(df["ground_truth"],df["prediction"])

disp = ConfusionMatrixDisplay(cm, display_labels=["Noise","Gunshot"])
disp.plot()

plt.title("Large Scale Gunshot Detection")
plt.show()