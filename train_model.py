import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ---------- LOAD DATA ----------
df = pd.read_csv("placementdata.csv")

# ---------- CLEAN COLUMN NAMES ----------
df.rename(columns={
    "Workshops/Certifications": "Workshops_Certifications"
}, inplace=True)

# ---------- CLEAN TARGET ----------
df["PlacementStatus"] = df["PlacementStatus"].astype(str).str.strip().str.lower()

df["PlacementStatus"] = df["PlacementStatus"].replace({
    "placed": 1,
    "notplaced": 0,
    "not placed": 0
})

df["PlacementStatus"] = df["PlacementStatus"].astype(int)

# ---------- FEATURES ----------
X = df.drop("PlacementStatus", axis=1)
y = df["PlacementStatus"]

# One-hot encoding
X = pd.get_dummies(X)

# ⭐ IMPORTANT FIX (Render compatibility)
X = X.astype("object")

# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- MODEL ----------
model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

# ---------- SAVE MODEL ----------
with open("placement_model.pkl", "wb") as f:
    pickle.dump((model, X.columns.tolist()), f)

print("✅ Model trained and saved successfully!")