import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("results/features.csv")

X = df[["energy","zcr","peak","centroid"]]
y = df["label"]

model = LogisticRegression()
model.fit(X,y)

for name,val in zip(X.columns,model.coef_[0]):
    print(name,":",val)