# model/train_doc_model.py
import os, re, joblib, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "documents.csv")
OUT_DIR = os.path.dirname(__file__)
RANDOM_STATE = 42

def load_data(path):
    df = pd.read_csv(path, encoding='latin-1')
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    print("📋 Columns detected:", df.columns.tolist())

    # find text column
    text_col = None
    for c in df.columns:
        if 'text' in c or 'email' in c or 'message' in c or 'v2' in c:
            text_col = c
            break
    if not text_col:
        raise ValueError("No text column found!")

    # find label column
    label_col = None
    for c in df.columns:
        if 'label' in c or 'class' in c or 'category' in c or 'v1' in c:
            label_col = c
            break
    if not label_col:
        raise ValueError("No label column found!")

    df = df[[text_col, label_col]].dropna()
    df.rename(columns={text_col: 'text', label_col: 'label'}, inplace=True)

    # Normalize labels (spam/phishing → 1, ham/safe → 0)
    df['label'] = df['label'].astype(str).str.lower().map(
        lambda x: 1 if 'spam' in x or 'phish' in x or 'scam' in x else 0
    )
    df = df.dropna(subset=['label'])
    return df

def preprocess_text(s):
    s = re.sub(r'http\\S+', ' ', s)          # remove links
    s = re.sub(r'[^a-zA-Z\\s]', ' ', s)      # keep only letters
    s = re.sub(r'\\s+', ' ', s).strip().lower()
    return s

def train():
    df = load_data(DATA_PATH)
    df['text'] = df['text'].astype(str).apply(preprocess_text)
    print(f"✅ Loaded {len(df)} samples")

    X = df['text'].values
    y = df['label'].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    tf = TfidfVectorizer(stop_words='english', max_features=30000)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')

    pipe = make_pipeline(tf, clf)
    params = {'logisticregression__C': [0.1, 1.0, 5.0]}
    gs = GridSearchCV(pipe, params, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    print("Best params:", gs.best_params_)

    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]

    print("📊 Classification report:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save vectorizer and model
    tfidf = best.named_steps['tfidfvectorizer']
    model = best.named_steps['logisticregression']
    joblib.dump(tfidf, os.path.join(OUT_DIR, "tfidf_doc.joblib"))
    joblib.dump(model, os.path.join(OUT_DIR, "doc_model.joblib"))
    print("💾 Saved tfidf_doc.joblib and doc_model.joblib")

if __name__ == "__main__":
    train()
