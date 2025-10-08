# model/train_url_model_structured.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "urls.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__))
RANDOM_STATE = 42

def train():
    print(f"Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print("Columns:", df.columns.tolist())

    # find label column
    label_col = None
    for c in df.columns:
        if 'label' in c.lower() or 'phish' in c.lower() or 'target' in c.lower():
            label_col = c
            break

    if not label_col:
        raise ValueError("❌ Could not find label column (expected something like 'Label' or 'phishing')")

    # Separate features and target
    X = df.drop(columns=[label_col, 'id'], errors='ignore')
    y = df[label_col].astype(int)

    print(f"✅ Loaded {X.shape[0]} samples with {X.shape[1]} features")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Train RandomForest
    print("🌲 Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(model, os.path.join(OUT_DIR, "url_model.joblib"))
    print(f"\n💾 Saved model as {os.path.join(OUT_DIR, 'url_model.joblib')}")
    print("⚠️ Note: TF-IDF vectorizer not needed for structured dataset")

if __name__ == "__main__":
    train()
