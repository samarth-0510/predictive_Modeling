import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

# Features and label
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Save model
joblib.dump(model, "model.pkl")
print("✅ model.pkl saved successfully.")
