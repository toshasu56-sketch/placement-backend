import pickle
from flask import Flask, request, jsonify
import pandas as pd
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)

CORS(app)
@app.route('/')
def home_page():
    return "Flask is working!"
model, training_columns = pickle.load(open("catboost_model.pkl", "rb"))

# 3️⃣ Routes (PUT YOUR CODE HERE 👇)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Incoming:", data)

    # Step 1: Create DataFrame FIRST
    input_df = pd.DataFrame([data])

    # ✅ STEP 2: FIX COLUMN NAMES (PUT YOUR CODE HERE)
    input_df["AptitudeTestScore"] = input_df.get("AptitudeScore", 0)
    input_df["Workshops/Certifications"] = input_df.get("Workshops_Certifications", 0)
   
    input_df.drop(columns=["AptitudeScore", "Workshops_Certifications"], errors='ignore', inplace=True)

    # ✅ STEP 3: MATCH MODEL FEATURES
    input_df = input_df.reindex(columns=model.feature_names_, fill_value=0)

    print("Final columns:", input_df.columns)

    # ✅ STEP 4: PREDICT
    prediction = model.predict(input_df)
    print("Prediction:", prediction)
    # Convert to readable result
    result = "Congratulations! You are placed." if prediction[0] == 1 else "Sorry, you are not placed."
    

    return jsonify({'prediction': result})

# 🔹 Load dataset
df = pd.read_csv("placementdata.csv")
#  remane column name to avoid issues with special characters
df.rename(columns={
    "Workshops/Certifications": "Workshops_Certifications"
}, inplace=True)


print("Before cleaning:", df.shape)
print("Unique values in PlacementStatus:", df['PlacementStatus'].unique())
# 🔹 Clean target column
# Clean target column
df['PlacementStatus'] = df['PlacementStatus'].astype(str).str.strip().str.lower()

# ✅ Correct mapping for YOUR dataset
df['PlacementStatus'] = df['PlacementStatus'].replace({
    'placed': 1,
    'notplaced': 0,     # important (no space)
    'not placed': 0     # just in case
})

# Convert to numeric
df['PlacementStatus'] = pd.to_numeric(df['PlacementStatus'], errors='coerce')

# Remove invalid rows
df = df.dropna(subset=['PlacementStatus'])

# Convert to int
df['PlacementStatus'] = df['PlacementStatus'].astype(int)

# Debug check
print("Unique values after cleaning:", df['PlacementStatus'].unique())
print("After cleaning:", df.shape)
# Convert to numeric (VERY IMPORTANT)
df['PlacementStatus'] = pd.to_numeric(df['PlacementStatus'], errors='coerce')

# Remove invalid rows
df = df.dropna(subset=['PlacementStatus'])

# Convert to integer type
df['PlacementStatus'] = df['PlacementStatus'].astype(int)

# Debug check
print("Unique values after cleaning:", df['PlacementStatus'].unique())
print("Data type:", df['PlacementStatus'].dtype)

# Remove rows where target is still invalid
df = df[df['PlacementStatus'].isin([0, 1])]

print("After cleaning:", df.shape)
# 🔹 Remove missing values
df = df.dropna(subset=['PlacementStatus'])

# 🔹 Split features and target
X = df.drop('PlacementStatus', axis=1)
y = df['PlacementStatus']

# 🔹 Convert categorical data to numeric
X = pd.get_dummies(X)

# 🔹 Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Train model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)

# 🔹 Prediction
y_pred = model.predict(X_test)


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 🔹 Accuracy
from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))






import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns

# 🔹 1. Placement Count
sns.countplot(x='PlacementStatus', data=df)
plt.title("Placement Count (0 = Not Placed, 1 = Placed)")
plt.show()
input("Press Enter to exit...")

# 🔹 2. Distribution of Marks
df.hist(figsize=(10, 8))
plt.suptitle("Feature Distributions")
plt.show()
input("Press Enter to exit...")

# 🔹 3. Scatter Plot (Aptitude vs SSC)
sns.scatterplot(
    x='AptitudeTestScore',
    y='SSC_Marks',
    hue='PlacementStatus',
    data=df
)
plt.title("Aptitude vs SSC Marks")
plt.show()
input("Press Enter to exit...")

# 🔹 4. Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()
input("Press Enter to exit...")




from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "SGD": SGDClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0)
}

accuracy_results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")


import matplotlib.pyplot as plt

names = list(accuracy_results.keys())
values = list(accuracy_results.values())

plt.figure()
plt.bar(names, values)
plt.title("Model Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.show()





from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

results = {}

for model_name, metrics in results.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

for model_name, metrics in results.items():
    print(f"\n{model_name}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
        models_list = list(results.keys())
models_list = list(results.keys())

accuracy = [results[m]["Accuracy"] for m in models_list]

print(accuracy)
plt.figure()
plt.bar(models_list, accuracy)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.show()

input("Press Enter to close graph...")



####################

#   saving all done
#################
import pickle

with open("catboost_model.pkl", "wb") as f:
    pickle.dump((model, X.columns), f)

####Build a prediction system####
def predict_placement(input_data):
    import pickle
    import pandas as pd

    model = pickle.load(open("catboost_model.pkl", "rb"))

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)

    prediction = model.predict(input_df)

    return "Placed" if prediction[0] == 1 else "Not Placed"

#####Test my  model#####
sample = {
    "StudentID": 1,
    "CGPA": 7.5,
    "Internships": 1,
    "Projects": 1,
    "Workshops/Certifications": 1,
    "AptitudeTestScore": 65,
    "SoftSkillsRating": 4.4,
    "ExtracurricularActivities": "No",
    "PlacementTraining": "No",
    "SSC_Marks": 61,
    "HSC_Marks": 79
}


import matplotlib.pyplot as plt

models = list(results.keys())
accuracies = [results[m]['Accuracy'] for m in models]

plt.bar(models, accuracies)
plt.title("Model Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.show()

import matplotlib.pyplot as plt

feature_importance = model.get_feature_importance()
feature_names = X.columns

plt.barh(feature_names, feature_importance)
plt.title("Feature Importance")
plt.show()
input("Press Enter to close graph...")

##############################3
####     CLI      ####
def predict_placement(sample):
    import pickle
    import pandas as pd

    # Load model + columns
    model, training_columns = pickle.load(open("catboost_model.pkl", "rb"))

    input_df = pd.DataFrame([sample])

    # Apply SAME encoding as training
    input_df = pd.get_dummies(input_df)

    # Match columns exactly
    input_df = input_df.reindex(columns=training_columns, fill_value=0)

    prediction = model.predict(input_df)

    return "Placed" if prediction[0] == 1 else "Not Placed"
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

   