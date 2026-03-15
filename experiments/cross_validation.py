import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("results/features.csv")

X = df[["energy","zcr","peak","centroid"]]
y = df["label"]

model = LogisticRegression()

scores = cross_val_score(model,X,y,cv=10)

print("Cross Validation Scores:",scores)
print("Mean Accuracy:",scores.mean())