"""
movies_stat_analysis.py
=======================

Comprehensive statistical analysis engine for the movies dataset.

This module is intentionally Streamlit-agnostic so it can be:
  * Imported by `app.py` (the Streamlit UI)
  * Run as a standalone script: `python scripts/analysis.py`

It exposes a single :class:`MovieAnalysis` class that loads the data, performs
feature engineering, and exposes a portfolio of statistical artifacts used by
the Streamlit app:
  - EDA summaries (shape, dtypes, missingness, descriptive stats)
  - Distribution tests (Shapiro-Wilk, skew, kurtosis, KS normality)
  - Correlation matrices (Pearson, Spearman, Kendall)
  - Variance Inflation Factors (VIF) for multicollinearity
  - OLS regression of Vote_Average on numeric + genre features
  - KMeans clustering of films into market segments
  - Hypothesis tests (ANOVA across genres, t-test across languages,
    chi-square of language vs. genre)
  - Feature-selection proposals (correlation ranking, mutual_info, RFE)
  - Outlier flagging (IQR + Z-score + IsolationForest)

All numeric work is done with scipy / statsmodels / scikit-learn as required.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols as smf_ols

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Data loading + feature engineering
# ---------------------------------------------------------------------------

def load_data(csv_path: Path | str) -> pd.DataFrame:
    """Load the raw CSV and return a cleaned DataFrame with engineered columns."""
    # Use engine='python' to avoid C-parser buffer overflow on long Overview fields
    # (some plot summaries exceed the C parser's ~1 MB per-field buffer).
    # on_bad_lines='skip' tolerates any remaining malformed rows.
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    # Parse release date (data uses DD-MM-YYYY)
    df["Release_Date"] = pd.to_datetime(df["Release_Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Release_Date"]).reset_index(drop=True)

    df["Release_Year"] = df["Release_Date"].dt.year.astype(int)
    df["Release_Month"] = df["Release_Date"].dt.month.astype(int)
    df["Release_DayOfWeek"] = df["Release_Date"].dt.dayofweek.astype(int)  # 0=Mon

    # Genre handling - the Genre column is comma-separated; primary genre is the first
    df["Genre_List"] = df["Genre"].fillna("").str.split(r",\s*")
    df["Primary_Genre"] = df["Genre_List"].apply(lambda g: g[0] if isinstance(g, list) and g else "Unknown")
    df["Num_Genres"] = df["Genre_List"].apply(lambda g: len(g) if isinstance(g, list) else 0)

    # Numeric coercions
    for col in ["Popularity", "Vote_Count", "Vote_Average"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Popularity", "Vote_Count", "Vote_Average"]).reset_index(drop=True)

    # Engineered columns proposed by feature engineering
    df["Log_Popularity"] = np.log1p(df["Popularity"])
    df["Log_Vote_Count"] = np.log1p(df["Vote_Count"])
    df["Popularity_per_Vote"] = df["Popularity"] / (df["Vote_Count"] + 1)
    df["Is_Blockbuster"] = (df["Vote_Count"] >= 1000).astype(int)
    df["Title_Length"] = df["Title"].fillna("").str.len()
    df["Overview_Length"] = df["Overview"].fillna("").str.len()
    df["Rating_Bucket"] = pd.cut(
        df["Vote_Average"],
        bins=[0, 5, 6.5, 7.5, 8.5, 10],
        labels=["Poor (<5)", "Fair (5-6.5)", "Good (6.5-7.5)", "Great (7.5-8.5)", "Excellent (8.5+)"],
    )

    return df


# ---------------------------------------------------------------------------
# Main analysis class
# ---------------------------------------------------------------------------

@dataclass
class MovieAnalysis:
    """Run + cache every statistical artifact the Streamlit app needs."""

    csv_path: Path | str
    df: pd.DataFrame = field(init=False)
    numeric_cols: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.df = load_data(self.csv_path)
        self.numeric_cols = [
            "Popularity", "Vote_Count", "Vote_Average",
            "Log_Popularity", "Log_Vote_Count",
            "Popularity_per_Vote", "Num_Genres",
            "Release_Year", "Release_Month",
            "Title_Length", "Overview_Length",
        ]

    # ----- EDA --------------------------------------------------------------

    def eda_summary(self) -> Dict[str, object]:
        df = self.df
        summary = {
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "head": df.head(5),
            "missing": df.isna().sum().to_frame("missing_count"),
            "missing_pct": (df.isna().mean() * 100).round(2).to_frame("missing_pct"),
            "describe_numeric": df[self.numeric_cols].describe().T,
            "describe_categorical": df[["Original_Language", "Primary_Genre", "Rating_Bucket"]].describe(),
            "n_unique_titles": df["Title"].nunique(),
            "n_unique_genres": len({g for lst in df["Genre_List"] for g in lst if isinstance(lst, list)}),
            "n_languages": df["Original_Language"].nunique(),
            "year_range": (int(df["Release_Year"].min()), int(df["Release_Year"].max())),
        }
        return summary

    def top_films(self, by: str = "Popularity", n: int = 20) -> pd.DataFrame:
        return (
            self.df.sort_values(by, ascending=False)
            .head(n)
            [["Title", "Release_Year", "Primary_Genre", "Original_Language", "Popularity", "Vote_Average", "Vote_Count", "Poster_Url"]]
            .reset_index(drop=True)
        )

    # ----- Distributions ----------------------------------------------------

    def distribution_tests(self) -> pd.DataFrame:
        """Skew, kurtosis, and Shapiro-Wilk normality test per numeric column."""
        rows = []
        for col in self.numeric_cols:
            x = self.df[col].dropna()
            if x.nunique() < 3 or len(x) < 8:
                continue
            sample = x.sample(min(len(x), 5000), random_state=42)
            try:
                sh_stat, sh_p = stats.shapiro(sample)
            except Exception:
                sh_stat, sh_p = np.nan, np.nan
            try:
                ks_stat, ks_p = stats.kstest(
                    (x - x.mean()) / x.std(ddof=0),
                    "norm",
                )
            except Exception:
                ks_stat, ks_p = np.nan, np.nan
            rows.append({
                "variable": col,
                "n": len(x),
                "mean": x.mean(),
                "median": x.median(),
                "std": x.std(ddof=1),
                "skew": stats.skew(x),
                "kurtosis": stats.kurtosis(x),
                "shapiro_W": sh_stat,
                "shapiro_p": sh_p,
                "is_normal_5pct": bool(sh_p > 0.05),
                "ks_stat": ks_stat,
                "ks_p": ks_p,
            })
        return pd.DataFrame(rows).set_index("variable")

    # ----- Correlations -----------------------------------------------------

    def correlation_matrices(self) -> Dict[str, pd.DataFrame]:
        df_num = self.df[self.numeric_cols].dropna()
        return {
            "pearson": df_num.corr(method="pearson"),
            "spearman": df_num.corr(method="spearman"),
            "kendall": df_num.corr(method="kendall"),
        }

    def top_correlations(self, method: str = "pearson", top_n: int = 15) -> pd.DataFrame:
        corr = self.correlation_matrices()[method]
        pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .reset_index()
            .rename(columns={"level_0": "var_x", "level_1": "var_y", 0: "coefficient"})
        )
        pairs["abs_coeff"] = pairs["coefficient"].abs()
        return pairs.sort_values("abs_coeff", ascending=False).head(top_n).drop(columns="abs_coeff").reset_index(drop=True)

    # ----- Multicollinearity (VIF) -----------------------------------------

    def variance_inflation_factors(self) -> pd.DataFrame:
        """Compute VIF for the candidate predictor set used in OLS.

        VIF > 5 indicates moderate multicollinearity, VIF > 10 is severe.
        """
        predictors = ["Popularity", "Vote_Count", "Num_Genres",
                      "Release_Year", "Release_Month", "Title_Length", "Overview_Length"]
        X = self.df[predictors].dropna().copy()
        # Add small jitter to avoid div-by-zero on near-constant cols
        for c in X.columns:
            if X[c].std(ddof=0) == 0:
                X[c] = X[c] + np.random.normal(0, 1e-6, size=len(X))
        X_const = sm.add_constant(X)
        vif_rows = []
        for i, col in enumerate(X_const.columns):
            if col == "const":
                continue
            try:
                vif = variance_inflation_factor(X_const.values, i)
            except Exception:
                vif = np.nan
            vif_rows.append({"variable": col, "VIF": vif})
        out = pd.DataFrame(vif_rows).set_index("variable")
        out["severity"] = pd.cut(
            out["VIF"],
            bins=[-np.inf, 5, 10, np.inf],
            labels=["low (<5)", "moderate (5-10)", "severe (>10)"],
        )
        return out

    # ----- OLS regression ---------------------------------------------------

    def ols_vote_average(self) -> Dict[str, object]:
        """OLS regression of Vote_Average on numeric + primary-genre dummies."""
        df = self.df.copy()
        predictors = ["Popularity", "Vote_Count", "Num_Genres",
                      "Release_Year", "Release_Month", "Title_Length"]
        # Use log-popularity instead of raw to reduce skew, plus primary genre dummies
        formula = "Vote_Average ~ Log_Popularity + Log_Vote_Count + Num_Genres + Release_Year + Release_Month + Title_Length + C(Primary_Genre)"
        try:
            model = smf_ols(formula, data=df).fit()
            return {
                "model": model,
                "summary_str": str(model.summary()),
                "rsquared": model.rsquared,
                "rsquared_adj": model.rsquared_adj,
                "f_pvalue": model.f_pvalue,
                "n_obs": int(model.nobs),
                "params": model.params,
                "pvalues": model.pvalues,
                "conf_int": model.conf_int(),
            }
        except Exception as e:
            return {"error": str(e)}

    # ----- Hypothesis tests -------------------------------------------------

    def anova_genre_vote(self) -> Dict[str, object]:
        """One-way ANOVA: does Vote_Average differ across primary genres?"""
        groups = [g["Vote_Average"].values for _, g in self.df.groupby("Primary_Genre") if len(g) >= 3]
        if len(groups) < 2:
            return {"error": "Not enough genres for ANOVA"}
        f_stat, p_val = stats.f_oneway(*groups)
        # Levene for equal-variance check
        lev_stat, lev_p = stats.levene(*groups)
        return {
            "test": "One-way ANOVA",
            "factor": "Primary_Genre",
            "response": "Vote_Average",
            "F_statistic": float(f_stat),
            "p_value": float(p_val),
            "levene_stat": float(lev_stat),
            "levene_p": float(lev_p),
            "interpretation": (
                "Reject H0 at 5%: genre means differ significantly." if p_val < 0.05
                else "Fail to reject H0: no significant genre effect on rating."
            ),
        }

    def ttest_language_vote(self) -> Dict[str, object]:
        """Two-sample t-test: Vote_Average for English vs non-English films."""
        en = self.df.loc[self.df["Original_Language"] == "en", "Vote_Average"].dropna()
        non_en = self.df.loc[self.df["Original_Language"] != "en", "Vote_Average"].dropna()
        if len(en) < 2 or len(non_en) < 2:
            return {"error": "Insufficient samples"}
        t_stat, p_val = stats.ttest_ind(en, non_en, equal_var=False)  # Welch's t
        return {
            "test": "Welch's two-sample t-test",
            "group_a": "English (en)",
            "group_b": "Non-English",
            "mean_a": float(en.mean()),
            "mean_b": float(non_en.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "interpretation": (
                "Reject H0: English vs non-English ratings differ significantly." if p_val < 0.05
                else "Fail to reject H0: no significant language effect."
            ),
        }

    def chi2_language_genre(self) -> Dict[str, object]:
        """Chi-square test of independence between language and primary genre."""
        ct = pd.crosstab(self.df["Original_Language"], self.df["Primary_Genre"])
        # Drop all-zero columns/rows for stability
        ct = ct.loc[:, (ct.sum() > 0)]
        ct = ct.loc[(ct.sum(axis=1) > 0)]
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return {"error": "Insufficient categories"}
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        return {
            "test": "Chi-square test of independence",
            "variables": ("Original_Language", "Primary_Genre"),
            "chi2_statistic": float(chi2),
            "dof": int(dof),
            "p_value": float(p),
            "interpretation": (
                "Reject H0: language and genre are associated." if p < 0.05
                else "Fail to reject H0: language and genre appear independent."
            ),
        }

    # ----- KMeans clustering ------------------------------------------------

    def kmeans_clusters(self, k: int = 4, random_state: int = 42) -> Tuple[pd.DataFrame, KMeans, pd.DataFrame]:
        """Cluster films on standardized Popularity, Vote_Count, Vote_Average."""
        feats = ["Log_Popularity", "Log_Vote_Count", "Vote_Average"]
        X = self.df[feats].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        clusters = km.fit_predict(X_scaled)
        cluster_col = pd.Series(clusters, index=X.index, name="Cluster")

        df_clustered = self.df.loc[X.index].copy()
        df_clustered["Cluster"] = cluster_col.values

        # Profile each cluster
        profile = (
            df_clustered.groupby("Cluster")
            .agg(
                n_films=("Title", "count"),
                avg_popularity=("Popularity", "mean"),
                avg_vote_count=("Vote_Count", "mean"),
                avg_vote_average=("Vote_Average", "mean"),
                median_year=("Release_Year", "median"),
            )
            .round(2)
        )

        # Human-readable labels based on profile
        def _label(row):
            if row["avg_vote_count"] >= 1500:
                return "Blockbuster"
            if row["avg_popularity"] >= 500 and row["avg_vote_count"] < 1500:
                return "Mainstream"
            if row["avg_vote_average"] >= 7.5:
                return "Cult / Critically Acclaimed"
            return "Niche / Indie"

        profile["segment_label"] = profile.apply(_label, axis=1)
        df_clustered["Segment"] = df_clustered["Cluster"].map(profile["segment_label"])

        return df_clustered, km, profile

    def elbow_inertia(self, k_max: int = 10) -> pd.DataFrame:
        feats = ["Log_Popularity", "Log_Vote_Count", "Vote_Average"]
        X = StandardScaler().fit_transform(self.df[feats].dropna())
        rows = []
        for k in range(1, k_max + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
            rows.append({"k": k, "inertia": km.inertia_})
        return pd.DataFrame(rows)

    # ----- Outlier detection ------------------------------------------------

    def detect_outliers(self) -> pd.DataFrame:
        """Flag outliers using IQR, Z-score, and IsolationForest for Popularity/Vote_Count/Vote_Average."""
        df = self.df.copy()
        flags = pd.DataFrame(index=df.index)

        for col in ["Popularity", "Vote_Count", "Vote_Average"]:
            x = df[col]
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            flags[f"{col}_iqr_outlier"] = ((x < lower) | (x > upper)).astype(int)
            z = (x - x.mean()) / x.std(ddof=0)
            flags[f"{col}_zscore_outlier"] = (z.abs() > 3).astype(int)

        # IsolationForest on log-popularity, log-vote, vote_average
        feats = ["Log_Popularity", "Log_Vote_Count", "Vote_Average"]
        iso = IsolationForest(contamination=0.05, random_state=42)
        pred = iso.fit_predict(df[feats].fillna(0))
        flags["isolation_forest_outlier"] = (pred == -1).astype(int)

        flags["any_outlier"] = (flags.sum(axis=1) > 0).astype(int)
        return flags

    # ----- Feature selection ------------------------------------------------

    def feature_selection(self) -> Dict[str, object]:
        """Rank candidate predictors of Vote_Average via three methods."""
        df = self.df.copy()
        features = ["Log_Popularity", "Log_Vote_Count", "Num_Genres",
                    "Release_Year", "Release_Month", "Title_Length",
                    "Overview_Length", "Popularity_per_Vote"]

        # Method 1: Pearson correlation with target
        pearson = df[features + ["Vote_Average"]].corr(method="pearson")["Vote_Average"].drop("Vote_Average")
        spearman = df[features + ["Vote_Average"]].corr(method="spearman")["Vote_Average"].drop("Vote_Average")

        # Method 2: Mutual information
        X = df[features].fillna(0)
        y = df["Vote_Average"].fillna(df["Vote_Average"].mean())
        mi = mutual_info_regression(X, y, random_state=42)

        # Method 3: RFE with linear regression
        lr = LinearRegression()
        rfe = RFE(lr, n_features_to_select=3)
        rfe.fit(X, y)
        rfe_rank = pd.Series(rfe.ranking_, index=features, name="rfe_rank")

        table = pd.DataFrame({
            "pearson_with_target": pearson,
            "abs_pearson": pearson.abs(),
            "spearman_with_target": spearman,
            "mutual_info": mi,
            "rfe_rank": rfe_rank,
        }).sort_values("abs_pearson", ascending=False)

        # Feature importance from RandomForest
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X, y)
        table["rf_importance"] = pd.Series(rf.feature_importances_, index=features)

        return {
            "table": table.round(4),
            "rfe_top3": list(rfe_rank[rfe_rank == 1].index),
            "rf_top3": list(pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False).head(3).index),
        }

    # ----- Genre deep dive --------------------------------------------------

    def genre_deep_dive(self) -> Dict[str, object]:
        """Per-genre statistics + top films per genre.

        Returns:
            - stats: DataFrame with count, avg/median Popularity, Vote_Count, Vote_Average per genre
            - top_films: dict {genre -> DataFrame of top 3 films by Popularity}
            - violin_data: long-form DataFrame for violin/box plots
        """
        df = self.df
        # Use ALL genres per film (a film in "Action, Adventure" counts in both)
        rows = []
        for _, film in df.iterrows():
            genres = film["Genre_List"] if isinstance(film["Genre_List"], list) else []
            for g in genres:
                rows.append({
                    "Genre": g,
                    "Popularity": film["Popularity"],
                    "Vote_Count": film["Vote_Count"],
                    "Vote_Average": film["Vote_Average"],
                    "Title": film["Title"],
                    "Release_Year": film["Release_Year"],
                })
        long_df = pd.DataFrame(rows)

        stats_df = (
            long_df.groupby("Genre")
            .agg(
                n_films=("Title", "count"),
                avg_popularity=("Popularity", "mean"),
                median_popularity=("Popularity", "median"),
                avg_vote_count=("Vote_Count", "mean"),
                avg_vote_average=("Vote_Average", "mean"),
                median_vote_average=("Vote_Average", "median"),
            )
            .round(2)
            .sort_values("n_films", ascending=False)
        )

        # Top 3 films per genre by popularity
        top_films = {}
        for genre in stats_df.index:
            sub = long_df[long_df["Genre"] == genre].nlargest(3, "Popularity")
            top_films[genre] = sub[["Title", "Release_Year", "Popularity", "Vote_Average"]].reset_index(drop=True)

        return {
            "stats": stats_df,
            "top_films": top_films,
            "violin_data": long_df[["Genre", "Vote_Average", "Popularity", "Vote_Count"]],
        }

    # ----- Temporal trends --------------------------------------------------

    def temporal_trends(self) -> Dict[str, object]:
        """Yearly, monthly, and seasonal trends in volume, rating, popularity."""
        df = self.df

        # Per-year stats (only years with >=3 films to reduce noise)
        yearly = (
            df.groupby("Release_Year")
            .agg(
                n_films=("Title", "count"),
                avg_vote_average=("Vote_Average", "mean"),
                avg_popularity=("Popularity", "mean"),
                median_vote_count=("Vote_Count", "median"),
            )
            .round(3)
        )
        yearly = yearly[yearly["n_films"] >= 3].reset_index()

        # Per-month stats (pooled across all years)
        monthly = (
            df.groupby("Release_Month")
            .agg(
                n_films=("Title", "count"),
                avg_vote_average=("Vote_Average", "mean"),
                avg_popularity=("Popularity", "mean"),
            )
            .round(3)
            .reset_index()
        )
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly["Month_Name"] = monthly["Release_Month"].apply(lambda m: month_names[m - 1])

        # Seasonal
        season_map = {1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                      6: "Summer", 7: "Summer", 8: "Summer",
                      9: "Fall", 10: "Fall", 11: "Fall", 12: "Winter"}
        df = df.assign(Release_Season=df["Release_Month"].map(season_map))
        seasonal = (
            df.groupby("Release_Season")
            .agg(
                n_films=("Title", "count"),
                avg_vote_average=("Vote_Average", "mean"),
                avg_popularity=("Popularity", "mean"),
            )
            .round(3)
            .reindex(["Winter", "Spring", "Summer", "Fall"])
            .reset_index()
        )

        # Decade
        df = df.assign(Decade=(df["Release_Year"] // 10 * 10).astype(str) + "s")
        decade = (
            df.groupby("Decade")
            .agg(
                n_films=("Title", "count"),
                avg_vote_average=("Vote_Average", "mean"),
                avg_popularity=("Popularity", "mean"),
            )
            .round(3)
            .reset_index()
        )

        # Decade × primary-genre heatmap data
        decade_genre = pd.crosstab(df["Decade"], df["Primary_Genre"])

        return {
            "yearly": yearly,
            "monthly": monthly,
            "seasonal": seasonal,
            "decade": decade,
            "decade_genre": decade_genre,
        }

    # ----- Popularity vs rating paradox ------------------------------------

    def popularity_rating_paradox(self) -> Dict[str, object]:
        """Quadrant analysis + overhyped / hidden gems lists.

        Quadrants defined on standardized Log_Popularity and Vote_Average:
          - High pop / High rating: Blockbuster Hits
          - High pop / Low rating:  Overhyped Flops
          - Low pop / High rating:  Hidden Gems
          - Low pop / Low rating:   Quiet Releases
        """
        df = self.df.copy()
        # Use median split for stability against outliers
        pop_med = df["Log_Popularity"].median()
        rating_med = df["Vote_Average"].median()
        df["Quadrant"] = np.where(
            (df["Log_Popularity"] >= pop_med) & (df["Vote_Average"] >= rating_med), "Blockbuster Hits",
            np.where(
                (df["Log_Popularity"] >= pop_med) & (df["Vote_Average"] < rating_med), "Overhyped Flops",
                np.where(
                    (df["Log_Popularity"] < pop_med) & (df["Vote_Average"] >= rating_med), "Hidden Gems",
                    "Quiet Releases",
                ),
            ),
        )

        quadrant_counts = df["Quadrant"].value_counts().to_frame("n_films")
        quadrant_counts["pct"] = (quadrant_counts["n_films"] / len(df) * 100).round(1)

        # Overhyped: high popularity, low rating. Rank by (popularity - rating * 1000) — high pop, low rating
        overhyped = (
            df[df["Quadrant"] == "Overhyped Flops"]
            .assign(hype_score=lambda d: d["Popularity"] / (d["Vote_Average"] + 0.5))
            .nlargest(10, "hype_score")
            [["Title", "Release_Year", "Primary_Genre", "Popularity", "Vote_Average", "Vote_Count"]]
            .reset_index(drop=True)
        )

        # Hidden gems: low popularity, high rating. Filter to require >= 50 votes to avoid noise.
        hidden = (
            df[(df["Quadrant"] == "Hidden Gems") & (df["Vote_Count"] >= 50)]
            .assign(gem_score=lambda d: d["Vote_Average"] * 100 / (np.log1p(d["Popularity"]) + 1))
            .nlargest(10, "gem_score")
            [["Title", "Release_Year", "Primary_Genre", "Popularity", "Vote_Average", "Vote_Count"]]
            .reset_index(drop=True)
        )

        return {
            "df": df[["Title", "Release_Year", "Primary_Genre", "Popularity", "Vote_Average",
                      "Vote_Count", "Log_Popularity", "Quadrant"]],
            "counts": quadrant_counts,
            "overhyped": overhyped,
            "hidden_gems": hidden,
            "thresholds": {"popularity_median": float(pop_med), "rating_median": float(rating_med)},
        }

    # ----- Random forest regressor -----------------------------------------

    def random_forest_model(self, test_size: float = 0.2, random_state: int = 42) -> Dict[str, object]:
        """Train/test split + RandomForest regression of Vote_Average.

        Returns metrics (R², RMSE, MAE), predictions DataFrame, and feature importances.
        """
        df = self.df.copy()
        features = ["Log_Popularity", "Log_Vote_Count", "Num_Genres",
                    "Release_Year", "Release_Month", "Title_Length",
                    "Overview_Length", "Popularity_per_Vote"]
        X = df[features].fillna(0)
        y = df["Vote_Average"].fillna(df["Vote_Average"].mean())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        rf = RandomForestRegressor(n_estimators=300, random_state=random_state)
        rf.fit(X_train, y_train)

        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)

        metrics = {
            "train_r2": float(r2_score(y_train, y_pred_train)),
            "test_r2": float(r2_score(y_test, y_pred_test)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
            "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }

        # Predictions for scatter (use test set)
        preds = pd.DataFrame({
            "actual": y_test.values,
            "predicted": y_pred_test,
            "residual": y_test.values - y_pred_test,
        })

        # Feature importance
        importances = (
            pd.Series(rf.feature_importances_, index=features, name="importance")
            .sort_values(ascending=False)
            .to_frame()
        )

        return {
            "metrics": metrics,
            "predictions": preds,
            "importances": importances,
            "features": features,
        }

    # ----- PCA projection ---------------------------------------------------

    def pca_projection(self, n_components: int = 2) -> Dict[str, object]:
        """Standardize numeric features and project to 2D via PCA."""
        features = ["Log_Popularity", "Log_Vote_Count", "Vote_Average",
                    "Num_Genres", "Release_Year", "Title_Length", "Overview_Length"]
        X = self.df[features].dropna().copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=n_components)
        components = pca.fit_transform(X_scaled)

        proj = self.df.loc[X.index].copy()
        for i in range(n_components):
            proj[f"PC{i+1}"] = components[:, i]

        # Loadings (how much each original feature contributes to each PC)
        loadings = pd.DataFrame(
            pca.components_.T,
            index=features,
            columns=[f"PC{i+1}" for i in range(n_components)],
        ).round(3)

        # Variance explained
        var_explained = pd.DataFrame({
            "component": [f"PC{i+1}" for i in range(n_components)],
            "variance_explained": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
        }).round(4)

        # Also include the KMeans segment (k=4) for color coding in the scatter
        _, _, _ = self.kmeans_clusters(k=4)
        df_clustered, _, _ = self.kmeans_clusters(k=4)
        if "Segment" in df_clustered.columns:
            proj["Segment"] = df_clustered.loc[proj.index, "Segment"]

        return {
            "projection": proj[["Title", "Release_Year", "Primary_Genre", "Popularity",
                                 "Vote_Average", "PC1", "PC2", "Segment"] if "Segment" in proj.columns
                                else ["Title", "Release_Year", "Primary_Genre", "Popularity",
                                      "Vote_Average", "PC1", "PC2"]],
            "loadings": loadings,
            "variance_explained": var_explained,
            "n_features": len(features),
        }

    # ----- Genre co-occurrence ---------------------------------------------

    def genre_cooccurrence(self) -> pd.DataFrame:
        """Pairwise genre co-occurrence matrix (how often two genres appear together)."""
        # Build a one-hot encoding of genres per film
        all_genres = sorted({g for lst in self.df["Genre_List"] for g in lst if isinstance(lst, list)})
        genre_matrix = pd.DataFrame(0, index=self.df.index, columns=all_genres, dtype=int)
        for idx, lst in self.df["Genre_List"].items():
            if isinstance(lst, list):
                for g in lst:
                    if g in genre_matrix.columns:
                        genre_matrix.loc[idx, g] = 1

        # Co-occurrence = matrix.T @ matrix
        cooc = genre_matrix.T @ genre_matrix
        # Diagonal is total films per genre; off-diagonal is pairwise co-occurrence count
        # Compute Jaccard similarity for the off-diagonal
        n_films_per_genre = pd.Series(np.diag(cooc), index=cooc.index)
        jaccard = pd.DataFrame(
            np.zeros_like(cooc, dtype=float),
            index=cooc.index, columns=cooc.columns,
        )
        for g1 in cooc.index:
            for g2 in cooc.columns:
                if g1 == g2:
                    jaccard.loc[g1, g2] = 1.0
                else:
                    union = (genre_matrix[g1] | genre_matrix[g2]).sum()
                    jaccard.loc[g1, g2] = cooc.loc[g1, g2] / union if union > 0 else 0.0
        return jaccard.round(3)

    def genre_pair_top(self, top_n: int = 15) -> pd.DataFrame:
        """Top-N most frequent genre pairs (excluding self-pairs)."""
        cooc = self.genre_cooccurrence()
        # Get raw counts via a fresh computation
        all_genres = list(cooc.index)
        genre_matrix = pd.DataFrame(0, index=self.df.index, columns=all_genres, dtype=int)
        for idx, lst in self.df["Genre_List"].items():
            if isinstance(lst, list):
                for g in lst:
                    if g in genre_matrix.columns:
                        genre_matrix.loc[idx, g] = 1
        raw_cooc = genre_matrix.T @ genre_matrix

        rows = []
        for i, g1 in enumerate(all_genres):
            for j, g2 in enumerate(all_genres):
                if i < j:
                    rows.append({
                        "genre_a": g1,
                        "genre_b": g2,
                        "co_count": int(raw_cooc.loc[g1, g2]),
                        "jaccard": float(cooc.loc[g1, g2]),
                    })
        return (
            pd.DataFrame(rows)
            .sort_values("co_count", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    # ----- Bootstrap confidence intervals ----------------------------------

    def bootstrap_ci(self, n_boot: int = 5000, random_state: int = 42) -> Dict[str, object]:
        """Bootstrap 95% confidence intervals for key statistics.

        Returns CIs for:
          - mean Vote_Average
          - mean Log_Popularity
          - Pearson correlation between Log_Vote_Count and Vote_Average
        """
        rng = np.random.default_rng(random_state)
        df = self.df

        vote = df["Vote_Average"].dropna().values
        log_pop = df["Log_Popularity"].dropna().values
        sub = df[["Log_Vote_Count", "Vote_Average"]].dropna()
        lv = sub["Log_Vote_Count"].values
        va = sub["Vote_Average"].values

        boot_means_vote = np.array([
            rng.choice(vote, size=len(vote), replace=True).mean()
            for _ in range(n_boot)
        ])
        boot_means_logpop = np.array([
            rng.choice(log_pop, size=len(log_pop), replace=True).mean()
            for _ in range(n_boot)
        ])
        boot_corr = np.array([
            (lambda idx: np.corrcoef(lv[idx], va[idx])[0, 1])(rng.integers(0, len(lv), size=len(lv)))
            for _ in range(n_boot)
        ])

        def _ci(arr):
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)),
                "ci_low": float(np.percentile(arr, 2.5)),
                "ci_high": float(np.percentile(arr, 97.5)),
            }

        return {
            "n_boot": n_boot,
            "mean_vote_average": _ci(boot_means_vote),
            "mean_log_popularity": _ci(boot_means_logpop),
            "corr_log_vote_vs_rating": _ci(boot_corr),
            "sample_stats": {
                "mean_vote_average": float(vote.mean()),
                "mean_log_popularity": float(log_pop.mean()),
                "pearson_corr": float(np.corrcoef(lv, va)[0, 1]),
            },
        }

    # ----- Effect size (Cohen's d) for the language t-test -----------------

    def cohen_d_language(self) -> Dict[str, object]:
        """Cohen's d effect size for English vs non-English Vote_Average."""
        en = self.df.loc[self.df["Original_Language"] == "en", "Vote_Average"].dropna()
        non_en = self.df.loc[self.df["Original_Language"] != "en", "Vote_Average"].dropna()
        if len(en) < 2 or len(non_en) < 2:
            return {"error": "Insufficient samples"}
        # Pooled standard deviation
        n1, n2 = len(en), len(non_en)
        s1, s2 = en.std(ddof=1), non_en.std(ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        d = (en.mean() - non_en.mean()) / pooled_std if pooled_std > 0 else np.nan
        magnitude = (
            "negligible" if abs(d) < 0.2 else
            "small" if abs(d) < 0.5 else
            "medium" if abs(d) < 0.8 else
            "large"
        )
        return {
            "cohen_d": float(d),
            "magnitude": magnitude,
            "mean_en": float(en.mean()),
            "mean_non_en": float(non_en.mean()),
            "pooled_std": float(pooled_std),
            "n_en": int(n1),
            "n_non_en": int(n2),
        }

    # ----- Feature engineering proposals -----------------------------------

    def feature_engineering_proposals(self) -> List[Dict[str, str]]:
        """Concrete proposals for additional columns, with rationale and code."""
        proposals = [
            {
                "column": "Log_Popularity",
                "rationale": "Popularity is heavily right-skewed (skew > 5). log1p transform restores near-normality, which stabilizes OLS / clustering.",
                "code": "df['Log_Popularity'] = np.log1p(df['Popularity'])",
                "status": "Implemented",
            },
            {
                "column": "Log_Vote_Count",
                "rationale": "Vote_Count spans 4 orders of magnitude. Log transform reduces heteroscedasticity in regression residuals.",
                "code": "df['Log_Vote_Count'] = np.log1p(df['Vote_Count'])",
                "status": "Implemented",
            },
            {
                "column": "Popularity_per_Vote",
                "rationale": "A film's hype-per-vote ratio signals marketing-driven vs organic popularity — useful for anomaly detection.",
                "code": "df['Popularity_per_Vote'] = df['Popularity'] / (df['Vote_Count'] + 1)",
                "status": "Implemented",
            },
            {
                "column": "Is_Blockbuster",
                "rationale": "Binary indicator (Vote_Count >= 1000) separating wide-release films from niche/indie titles.",
                "code": "df['Is_Blockbuster'] = (df['Vote_Count'] >= 1000).astype(int)",
                "status": "Implemented",
            },
            {
                "column": "Title_Length / Overview_Length",
                "rationale": "Length-based text features are cheap proxies for marketing style. Title length correlates with franchise films.",
                "code": "df['Title_Length'] = df['Title'].str.len(); df['Overview_Length'] = df['Overview'].str.len()",
                "status": "Implemented",
            },
            {
                "column": "Primary_Genre / Num_Genres",
                "rationale": "Genre is comma-separated; first element is the dominant genre. Count gives genre breadth.",
                "code": "df['Primary_Genre'] = df['Genre'].str.split(',').str[0].str.strip(); df['Num_Genres'] = df['Genre'].str.count(',') + 1",
                "status": "Implemented",
            },
            {
                "column": "Release_Season",
                "rationale": "Categorical season (Winter/Spring/Summer/Fall) captures release-strategy cycles (summer blockbusters vs award-season films).",
                "code": "season_map = {1:'Winter',2:'Winter',3:'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Fall',10:'Fall',11:'Fall',12:'Winter'}\ndf['Release_Season'] = df['Release_Month'].map(season_map)",
                "status": "Proposed",
            },
            {
                "column": "Sentiment_Score",
                "rationale": "Run VADER or TextBlob on Overview text to extract sentiment polarity; correlates with tone of the film.",
                "code": "from textblob import TextBlob; df['Sentiment_Score'] = df['Overview'].apply(lambda t: TextBlob(t).sentiment.polarity)",
                "status": "Proposed",
            },
            {
                "column": "Decade",
                "rationale": "Grouping release year into decades (1990s, 2000s, 2010s, 2020s) gives a stable categorical for trend analysis.",
                "code": "df['Decade'] = (df['Release_Year'] // 10 * 10).astype(str) + 's'",
                "status": "Proposed",
            },
        ]
        return proposals

    def standardization_proposals(self) -> List[Dict[str, str]]:
        return [
            {
                "column": "Popularity",
                "current_range": f"[{self.df['Popularity'].min():.2f}, {self.df['Popularity'].max():.2f}]",
                "recommendation": "StandardScaler (z-score) before KMeans / PCA. Use log1p first to tame skew.",
                "code": "StandardScaler().fit_transform(np.log1p(df[['Popularity']]))",
            },
            {
                "column": "Vote_Count",
                "current_range": f"[{self.df['Vote_Count'].min()}, {self.df['Vote_Count'].max()}]",
                "recommendation": "MinMaxScaler to [0,1] for neural nets, or log1p + StandardScaler for linear models.",
                "code": "MinMaxScaler().fit_transform(np.log1p(df[['Vote_Count']]))",
            },
            {
                "column": "Vote_Average",
                "current_range": f"[{self.df['Vote_Average'].min()}, {self.df['Vote_Average'].max()}]",
                "recommendation": "Already bounded [0,10]. Center only (subtract mean) for regression intercept interpretability.",
                "code": "df['Vote_Average_centered'] = df['Vote_Average'] - df['Vote_Average'].mean()",
            },
            {
                "column": "Title_Length / Overview_Length",
                "current_range": f"titles: [0, {self.df['Title_Length'].max()}], overviews: [0, {self.df['Overview_Length'].max()}]",
                "recommendation": "StandardScaler — both are positively skewed and benefit from z-scoring.",
                "code": "StandardScaler().fit_transform(df[['Title_Length', 'Overview_Length']])",
            },
        ]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main(csv_path: str = "data/movies.csv") -> None:
    ma = MovieAnalysis(csv_path=csv_path)

    _print_section("EDA SUMMARY")
    eda = ma.eda_summary()
    print(f"Shape: {eda['shape']}")
    print(f"Year range: {eda['year_range']}")
    print(f"Unique titles: {eda['n_unique_titles']}, genres: {eda['n_unique_genres']}, languages: {eda['n_languages']}")
    print("\nMissing values:")
    print(eda["missing"].to_string())

    _print_section("DESCRIPTIVE STATISTICS (numeric)")
    print(eda["describe_numeric"].round(2).to_string())

    _print_section("DISTRIBUTION TESTS")
    print(ma.distribution_tests().round(4).to_string())

    _print_section("CORRELATIONS (Pearson)")
    print(ma.correlation_matrices()["pearson"].round(3).to_string())

    _print_section("TOP 15 CORRELATION PAIRS (Pearson)")
    print(ma.top_correlations("pearson", 15).to_string(index=False))

    _print_section("VARIANCE INFLATION FACTORS")
    print(ma.variance_inflation_factors().round(3).to_string())

    _print_section("OLS REGRESSION: Vote_Average ~ ...")
    ols = ma.ols_vote_average()
    if "error" in ols:
        print(f"Error: {ols['error']}")
    else:
        print(f"R² = {ols['rsquared']:.4f} | Adj R² = {ols['rsquared_adj']:.4f} | F p-value = {ols['f_pvalue']:.4g}")
        print(ols["summary_str"][:3000])

    _print_section("HYPOTHESIS TEST: ANOVA genre vs Vote_Average")
    print(ma.anova_genre_vote())

    _print_section("HYPOTHESIS TEST: t-test English vs non-English Vote_Average")
    print(ma.ttest_language_vote())

    _print_section("HYPOTHESIS TEST: chi2 language vs genre")
    print(ma.chi2_language_genre())

    _print_section("KMEANS CLUSTERING (k=4)")
    _, _, profile = ma.kmeans_clusters(k=4)
    print(profile.to_string())

    _print_section("FEATURE SELECTION")
    fs = ma.feature_selection()
    print(fs["table"].to_string())
    print(f"\nRFE top-3: {fs['rfe_top3']}")
    print(f"RF top-3:  {fs['rf_top3']}")

    _print_section("FEATURE ENGINEERING PROPOSALS")
    for p in ma.feature_engineering_proposals():
        print(f"  - {p['column']} [{p['status']}]: {p['rationale']}")

    _print_section("STANDARDIZATION PROPOSALS")
    for p in ma.standardization_proposals():
        print(f"  - {p['column']} (range {p['current_range']}): {p['recommendation']}")


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/movies.csv"
    main(csv)
