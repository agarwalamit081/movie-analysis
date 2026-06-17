# 🎬 Movie Stats — Cinematic Analytics

A Streamlit dashboard for browsing + statistically analyzing a 564-film dataset
(TMDB-derived). Two tabs:

1. **🔥 Trending Films** — filter by release year, genre, language, popularity,
   vote average, or free-text title search. Poster-card grid + KPI tiles +
   leaderboards.
2. **📊 Statistical Insights** — a comprehensive portfolio of analyses:
   - Exploratory Data Analysis
   - Distribution analysis (Shapiro-Wilk, Kolmogorov-Smirnov, skew, kurtosis)
   - Correlation analysis (Pearson / Spearman / Kendall + top pairs)
   - Multicollinearity (VIF)
   - OLS regression of `Vote_Average`
   - Hypothesis tests (ANOVA, Welch t-test, chi-square)
   - KMeans clustering with elbow + 3D scatter
   - Outlier detection (IQR + Z-score + Isolation Forest)
   - Feature selection (Pearson / Spearman / MI / RFE / RandomForest)
   - Feature engineering proposals
   - Standardization / normalization guidance

Style: **Cinematic Dark** — charcoal background, gold accents, large poster imagery.

---

## 📦 Project layout

```
movie-stats-app/
├── app.py                  # Streamlit entry point
├── requirements.txt        # Python dependencies
├── README.md
├── data/
│   └── movies.csv          # The dataset (564 films)
├── scripts/
│   └── analysis.py         # Standalone statistical analysis engine
└── styles/
    └── style.css           # Cinematic dark theme
```

---

## 🚀 Setup

### 1. Create a virtualenv (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### 4. (Optional) Run the analysis standalone

The analysis engine is Streamlit-agnostic. Run it as a CLI to print every
statistical artifact to stdout:

```bash
python scripts/analysis.py data/movies.csv
```

---

## 🧪 Statistical methods used

| Concern | Method | Library |
|---|---|---|
| Normality | Shapiro-Wilk, Kolmogorov-Smirnov | `scipy.stats` |
| Correlation | Pearson, Spearman, Kendall | `pandas` |
| Multicollinearity | Variance Inflation Factor (VIF) | `statsmodels` |
| Regression | OLS with categorical dummies | `statsmodels.formula.api` |
| Group comparison | One-way ANOVA, Levene, Welch t-test | `scipy.stats` |
| Independence | Chi-square test of independence | `scipy.stats` |
| Clustering | KMeans + elbow | `scikit-learn` |
| Outliers | IQR + Z-score + Isolation Forest | `scipy` + `scikit-learn` |
| Feature selection | Pearson, Spearman, mutual_info, RFE, RandomForest | `scikit-learn` |

---

## 🛠️ Feature engineering

The analysis engine derives these additional columns from the raw data:

| Column | Derivation | Why |
|---|---|---|
| `Release_Year`, `Release_Month`, `Release_DayOfWeek` | parsed from `Release_Date` | enables time-series filters |
| `Primary_Genre`, `Num_Genres` | first element / count of `Genre` (comma-separated) | categorical pivot + breadth |
| `Log_Popularity`, `Log_Vote_Count` | `np.log1p` of raw values | tames heavy right skew |
| `Popularity_per_Vote` | `Popularity / (Vote_Count + 1)` | hype-vs-organic signal |
| `Is_Blockbuster` | `Vote_Count >= 1000` | binary segmentation |
| `Title_Length`, `Overview_Length` | `str.len()` | text-based proxy features |
| `Rating_Bucket` | 5-bucket cut of `Vote_Average` | categorical stratification |

See the **Feature Engineering Proposals** section in the app for further
suggested columns (Release_Season, Sentiment_Score, Decade).

---

## 🎨 Theme

CSS lives in `styles/style.css`. The palette:

- Background: `#0b0d12` (charcoal) → `#141821` (gradient)
- Cards: `#1a1f2c`
- Accent: `#f5c518` (IMDb-style gold)
- Text: `#f5f5f7` primary, `#b8bfc9` secondary, `#6b7280` muted
- Status: green `#2ecc71`, red `#e63946`

To switch to a light theme, replace `style.css` — the Streamlit layout is
unchanged.

---

## 📋 Dataset

- **File:** `data/movies.csv` (original name `movies-Copy.csv`)
- **Rows:** 564
- **Columns:** `Release_Date, Title, Overview, Popularity, Vote_Count,
  Vote_Average, Original_Language, Genre, Poster_Url`
- **Year range:** 1921 – 2022
- **Languages:** 19 (dominated by `en`)
- **Genres:** 19 distinct (a film can have multiple)

---

## ❓ Logical questions this analysis answers

1. **Is `Vote_Average` normally distributed?** → No (Shapiro p < 0.001, left-skewed).
2. **Does popularity correlate with rating?** → Weakly (Pearson r ≈ 0.09 raw,
   but `Log_Vote_Count` r ≈ 0.50).
3. **Does genre affect rating?** → Yes, ANOVA p ≈ 1.3e-9.
4. **Are English films rated differently from non-English?** → No, Welch
   t-test p ≈ 0.07.
5. **Are language and genre associated?** → Yes, χ² p ≈ 0.04 (weak).
6. **Is multicollinearity a problem for the OLS model?** → No, all VIF ≈ 1.
7. **Can films be meaningfully clustered?** → Yes, KMeans with k=4 cleanly
   separates Blockbusters / Mainstream / Cult / Niche.
8. **Which features best predict rating?** → `Log_Vote_Count`,
   `Popularity_per_Vote`, `Release_Year` (consensus across methods).
9. **Are there outliers that distort the analysis?** → Yes, ~5% flagged by
   Isolation Forest — mostly true blockbusters + a few data-entry errors.

---

## 📝 License

The dataset is derived from The Movie Database (TMDB). Code is MIT-licensed.
