# AI-Powered Phishing & Scam Detection System

This is a starter Flask project that demonstrates:
- URL scanning (heuristics + demo TF-IDF model)
- Document (PDF / TXT / HTML) scanning for scam/phishing-like text (demo TF-IDF model)

## Quick start (local)

1. Clone / copy project to local folder:
```bash
git clone <your-repo-url> phish_scam_detector
cd phish_scam_detector
```

2. Create virtual environment and install dependencies:
```bash
python -m venv venv
# Linux / mac:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

pip install -r requirements.txt
```

3. Create demo models (this generates `model/*.joblib`):
```bash
python model/create_demo_models.py
```

4. Run the app:
```bash
python app.py
```
Open `http://127.0.0.1:5000/` in your browser.

## Replace demo models with real ones
- Train URL model: use a dataset of labeled URLs (phishing vs benign). Use TF-IDF on character n-grams (good for URLs). Save classifier as `model/url_model.joblib` and vectorizer as `model/tfidf_url.joblib`.
- Train Document model: gather scam/phish email datasets (Kaggle). Train TF-IDF (word level) + LogisticRegression. Save `model/doc_model.joblib` and `model/tfidf_doc.joblib`.

**Important:** Keep training scripts and list dataset sources in README. Don't commit large datasets to the GitHub repository; instead include download scripts.

## Security & Deployment notes
- Do **not** allow public access to server-side page fetchers without rate limits and stronger SSRF defense.
- Add HTTPS and environment variable management when deploying.
- Deploy backend (Flask) on Render, Railway or similar:
  - Connect GitHub repository.
  - Set start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
  - Add environment variables via host dashboard.

## Git setup example
```bash
git init
git add .
git commit -m "Initial phish_scam_detector starter"
# create repo on GitHub, then:
git remote add origin git@github.com:YOURUSER/phish_scam_detector.git
git branch -M main
git push -u origin main
```

## Next steps (recommended)
1. Train with real datasets and replace demo models.
2. Add PhishTank offline DB lookup for faster and private checks.
3. Add unit tests + CI (GitHub Actions).
4. Add rate limiting and authentication for a public API.
