"""Train/test a Random Forest gesture classifier from gestures.csv."""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CSV_PATH = "gestures.csv"
MODEL_PATH = "gesture_model.pkl"
if not os.path.isfile(CSV_PATH):
    raise SystemExit("gestures.csv missing. Run 07_collect.py first.")

df = pd.read_csv(CSV_PATH, header=None)
if df.shape[1] != 43:
    raise SystemExit(f"Expected 42 features plus label (43 columns); found {df.shape[1]}.")
X = df.iloc[:, :-1].astype(float)
y = df.iloc[:, -1].astype(str)
counts = y.value_counts()
print("Samples per gesture:\n", counts)
if len(counts) != 4 or counts.min() < 20:
    raise SystemExit("Need all four labels (fist, palm, peace, point), with at least 20 samples each.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"\nHeld-out accuracy: {accuracy_score(y_test,pred):.3f}")
print("\nClassification report:\n", classification_report(y_test,pred,zero_division=0))
labels = sorted(y.unique())
print("Confusion matrix label order:", labels)
print(confusion_matrix(y_test,pred,labels=labels))
joblib.dump(model, MODEL_PATH)
print(f"\nSaved {MODEL_PATH}")
print("Caution: adjacent frames can be very similar; test with a separate session/person for a better generalisation check.")
