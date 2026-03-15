import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("results/features.csv")

X = df[["energy","zcr","peak","centroid","bandwidth","decay"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

model = RandomForestClassifier(n_estimators=2000)
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test,pred))
print("Precision:", precision_score(y_test,pred))
print("Recall   :", recall_score(y_test,pred))
print("F1 Score :", f1_score(y_test,pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test,pred))