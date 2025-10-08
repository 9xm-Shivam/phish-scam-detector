from flask import Flask, render_template, request
import joblib
import os
import PyPDF2
import re
import tldextract

app = Flask(__name__)

# ==============================
# Load Models
# ==============================
def safe_load(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

url_model = safe_load("model/url_model.joblib")
tfidf_url = safe_load("model/tfidf_url.joblib")
doc_model = safe_load("model/doc_model.joblib")
tfidf_doc = safe_load("model/tfidf_doc.joblib")


# ==============================
# Helper for RandomForest numeric features
# ==============================
def extract_basic_url_features(url: str):
    """
    Create a small numeric feature vector for a URL if only RandomForest model exists.
    These should roughly correspond to the type of columns in your structured dataset.
    """
    features = [
        url.count("."),                       # NumDots
        url.count("-"),                       # NumDash
        len(url),                             # urlLength
        int("@" in url),                      # AtSymbol
        int("https" not in url.lower()),       # NoHttps
        int(re.search(r"\d", url) is not None) # HasDigit
    ]
    while len(features) < 48:  # pad to expected size
        features.append(0)
    return features


# ==============================
# Routes
# ==============================
@app.route("/")
def index():
    return render_template("index.html")


# ---------- URL SCAN ----------
@app.route("/scan_url", methods=["POST"])
def scan_url():
    url = request.form.get("url")
    if not url:
        return render_template(
            "result.html",
            kind="URL",
            target="(No URL Provided)",
            score=0,
            reasons=["No URL entered."],
            snippet=""
        )

    reasons = []
    score = 0.0

    try:
        # --- Case 1: TF-IDF + LogisticRegression model ---
        if url_model and tfidf_url and "LogisticRegression" in str(type(url_model)):
            X = tfidf_url.transform([url])
            score = float(url_model.predict_proba(X)[0, 1])
            reasons.append("TF-IDF + Logistic Regression model used for text-based URL detection.")

        # --- Case 2: RandomForest structured model ---
        elif url_model and "RandomForest" in str(type(url_model)):
            feats = extract_basic_url_features(url)
            score = float(url_model.predict_proba([feats])[0, 1])
            reasons.append("RandomForest model used on numeric URL features.")

        else:
            reasons.append("⚠️ No valid URL model loaded.")
    except Exception as e:
        reasons.append(f"Error while predicting: {e}")

    return render_template(
        "result.html",
        kind="URL",
        target=url,
        score=round(score, 3),
        reasons=reasons,
        snippet=""
    )


# ---------- DOCUMENT / PDF SCAN ----------
@app.route("/scan_doc", methods=["POST"])
def scan_doc():
    file = request.files.get("file")
    if not file:
        return render_template(
            "result.html",
            kind="Document",
            target="(No file provided)",
            score=0,
            reasons=["No file was uploaded."],
            snippet=""
        )

    filename = file.filename
    ext = filename.lower().split(".")[-1]
    text_content = ""

    # Extract text
    try:
        if ext == "pdf":
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text_content += page.extract_text() or ""
        else:
            text_content = file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return render_template(
            "result.html",
            kind="Document",
            target=filename,
            score=0,
            reasons=[f"Error reading file: {e}"],
            snippet=""
        )

    reasons = []
    score = 0.0

    try:
        if doc_model and tfidf_doc:
            X = tfidf_doc.transform([text_content])
            score = float(doc_model.predict_proba(X)[0, 1])
            reasons.append("TF-IDF document model used for text analysis.")
        else:
            reasons.append("⚠️ Document model not loaded — using default score 0.0.")
    except Exception as e:
        reasons.append(f"Error while predicting: {e}")

    snippet = text_content[:400].replace("\n", " ") + "..."
    return render_template(
        "result.html",
        kind="Document",
        target=filename,
        score=round(score, 3),
        reasons=reasons,
        snippet=snippet
    )


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(rule)
    app.run(debug=True)
