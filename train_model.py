import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ---------- LOAD DATA ----------
df = pd.read_csv("placementdata.csv")

# rename column
df.rename(columns={"Workshops/Certifications": "Workshops_Certifications"}, inplace=True)

# clean target
df["PlacementStatus"] = df["PlacementStatus"].astype(str).str.strip().str.lower()
df["PlacementStatus"] = df["PlacementStatus"].replace({
    "placed": 1,
    "notplaced": 0,
    "not placed": 0
})

df["PlacementStatus"] = df["PlacementStatus"].astype(int)

# ---------- SPLIT ----------
X = df.drop("PlacementStatus", axis=1)
y = df["PlacementStatus"]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---------- MODEL ----------
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

# ---------- SAVE ----------
with open("catboost_model.pkl", "wb") as f:
    pickle.dump((model, X.columns), f)

print("Model saved successfully!")