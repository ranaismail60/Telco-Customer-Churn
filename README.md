# NexaTel Churn Prediction & Retention Intelligence System

🌐 **Live Deployed Web Application**: [https://telco-customer-churn-1.onrender.com](https://telco-customer-churn-1.onrender.com)  
*(Try out live customer churn risk scoring and SHAP explanations directly in your browser!)*

## What's in here

```
nexatel-churn/
├── data/                  # put telco_churn.csv here (you download it)
├── sql/
│   ├── schema.sql          # Phase 1 — normalized schema
│   └── queries.sql         # Phase 1 — 13 business queries
├── scripts/
│   ├── 01_load_data.py     # Phase 1 — CSV -> SQLite
│   ├── 02_eda.py           # Phase 2 — EDA, saves plots to reports/
│   ├── 03_feature_engineering.py  # Phase 3
│   ├── 04_preprocessing.py # Phase 4 — split, scale, SMOTE
│   ├── 05_train_models.py  # Phase 5 — train, evaluate, tune, save model
│   └── 06_explainability.py # Phase 6 — SHAP
├── backend/
│   ├── app.py               # Phase 7 — Flask API (/predict)
│   └── requirements.txt
├── frontend/
│   ├── index.html            # Phase 7 — retention agent tool
│   ├── style.css
│   └── script.js
├── models/                  # model.pkl, scaler.pkl saved here
├── reports/                  # figures + comparison tables saved here
└── requirements.txt
```

## How to run it (in order)

### 0. Get the data
Download from Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Save the CSV as `data/telco_churn.csv`.

### 1. Set up your environment
```bash
cd nexatel-churn
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the pipeline, in this exact order
```bash
python scripts/01_load_data.py            # builds data/nexatel.db
sqlite3 data/nexatel.db < sql/queries.sql  # optional: view query outputs in terminal
python scripts/02_eda.py                   # writes reports/figures/*.png
python scripts/03_feature_engineering.py   # writes data/engineered.csv
python scripts/04_preprocessing.py         # writes data/processed/*, models/scaler.pkl
python scripts/05_train_models.py          # writes models/model.pkl, reports/model_comparison.csv
python scripts/06_explainability.py        # writes reports/figures/shap_summary.png
```

### 3. Run the web app locally
```bash
# Terminal 1
cd backend
pip install -r requirements.txt
python app.py            # runs on http://localhost:5000

# Terminal 2 — just open the file, or serve it
cd frontend
python -m http.server 8080   # then open http://localhost:8080
```

### 4. Deploy (Phase 8)
- **Backend** → Render (free web service). Point it at `backend/`, build command
  `pip install -r requirements.txt`, start command `gunicorn app:app`.
- **Frontend** → Vercel or Netlify, pointed at `frontend/`. After deploying,
  update `API_URL` in `frontend/script.js` to your live Render backend URL,
  then redeploy the frontend.

## What you still need to do yourself

The code above gets every phase technically working end-to-end. What it does
**not** do — because it's the part that's actually graded and the part
you'll be asked to defend in interviews — is the reasoning:

- **Phase 0:** write your own 2–3 sentence problem statement and pick which
  error type (missing a churner vs. false-alarming) costs more, in your words.
- **Phase 1:** read through `queries.sql` and be ready to explain each JOIN
  and each business question it answers.
- **Phase 2:** open the plots in `reports/figures/` and write the one-page
  insights summary — what's the single most at-risk segment, and why.
- **Phase 3:** for each engineered feature, write 1–2 sentences on why you
  expected it to help and whether the EDA backs that up.
- **Phase 5:** look at `reports/model_comparison.csv` and write why you're
  optimizing for the metric you picked, in NexaTel's business terms.
- **Phase 9:** the README, case study, and resume bullets should describe
  *your* results (real numbers from your own run), not placeholder text.

Run each script, actually read its printed output and saved plots before
moving to the next phase — several scripts print a "NEXT" hint at the end
telling you exactly what to look at.
