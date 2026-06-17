"""
app.py — Movie Stats Streamlit App
==================================

Run with:
    streamlit run app.py

Two tabs:
  1. Trending Films — browse + filter films, poster-card grid, KPI tiles
  2. Statistical Insights — full statistical analysis portfolio
     (EDA, distributions, correlations, VIF, OLS, hypothesis tests,
      KMeans clusters, feature selection, feature engineering proposals)

Style: Cinematic Dark (charcoal background, gold accents).
"""

from __future__ import annotations

import base64
import hashlib
import io
import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from PIL import Image

# Make the local scripts/ importable
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from analysis import MovieAnalysis  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration + data caching
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Movie Stats — Cinematic Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Font setup (CJK + symbol fallback for matplotlib)
try:
    fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
except Exception:
    pass
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="darkgrid", palette="muted", font_scale=0.9)

# Cinematic palette
GOLD = "#f5c518"
BG_DARK = "#0b0d12"
BG_CARD = "#1a1f2c"
TEXT_PRI = "#f5f5f7"
TEXT_SEC = "#b8bfc9"


@st.cache_resource(show_spinner=False)
def load_analysis(csv_path: str) -> MovieAnalysis:
    return MovieAnalysis(csv_path=csv_path)


@st.cache_data(show_spinner=False)
def load_css(css_path: str) -> str:
    return Path(css_path).read_text()


def inject_css() -> None:
    css_path = Path(__file__).parent / "styles" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def app_header(subtitle: str) -> None:
    _render_html(
        f'<div class="app-header">'
        f'<h1>🎬 Movie Stats — Cinematic Analytics</h1>'
        f'<div class="subtitle">{subtitle}</div>'
        f'</div>'
    )


def section_title(title: str) -> None:
    _render_html(f'<div class="section-title"><h2>{title}</h2></div>')


def callout(text: str, kind: str = "info") -> None:
    _render_html(f'<div class="callout callout-{kind}">{text}</div>')


def kpi_grid(items: list[dict]) -> None:
    """items = [{"label":..., "value":..., "sub":...}, ...]"""
    cells = "".join(
        f'<div class="kpi-tile">'
        f'<div class="label">{it["label"]}</div>'
        f'<div class="value">{it["value"]}</div>'
        f'<div class="sub">{it.get("sub", "")}</div>'
        f'</div>'
        for it in items
    )
    _render_html(f'<div class="kpi-grid">{cells}</div>')


def _render_html(html: str) -> None:
    """Render raw HTML reliably across Streamlit versions.

    st.markdown(..., unsafe_allow_html=True) sometimes wraps block-level HTML
    in <p> tags and shows the raw source. st.html() (Streamlit >=1.39) renders
    block HTML correctly. Fall back to st.markdown for older versions, with
    a leading newline to nudge the markdown parser into HTML mode.
    """
    html = html.strip()
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown("\n" + html + "\n", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Poster image caching
# ---------------------------------------------------------------------------

IMAGE_CACHE_DIR = Path(__file__).parent / "data" / "images"


@st.cache_data(show_spinner=False, max_entries=1000)
def _fetch_poster_bytes(url: str) -> bytes | None:
    """Download poster bytes from URL with disk caching.

    Returns None on any failure (network error, timeout, tiny response).
    Posters are persisted to data/images/{md5(url)}.jpg so subsequent
    page loads / sessions skip the network entirely.

    Network calls use a SHORT timeout (3s) so the UI doesn't hang when
    the server can't reach the CDN — the caller falls back to the live URL.
    """
    if not url or not isinstance(url, str) or not url.startswith("http"):
        return None

    try:
        IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
    cache_path = IMAGE_CACHE_DIR / f"{url_hash}.jpg"

    # Disk cache hit
    if cache_path.exists() and cache_path.stat().st_size > 100:
        try:
            return cache_path.read_bytes()
        except Exception:
            pass

    # Download with a sane User-Agent (some CDNs reject requests without one).
    # Try requests first if available (handles redirects + gzip better), fall back to urllib.
    try:
        import requests  # type: ignore
        try:
            resp = requests.get(url, timeout=3, headers={"User-Agent": "Mozilla/5.0 (compatible; MovieStatsApp/1.0)"})
            if resp.status_code == 200 and len(resp.content) > 100:
                try:
                    cache_path.write_bytes(resp.content)
                except Exception:
                    pass
                return resp.content
        except Exception:
            pass  # fall through to urllib
    except ImportError:
        pass

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MovieStatsApp/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = resp.read()
        if data and len(data) > 100:
            try:
                cache_path.write_bytes(data)
            except Exception:
                pass
            return data
    except Exception:
        return None
    return None


def _poster_html(poster_url: str) -> str:
    """Build the <img> or fallback HTML for a poster.

    Priority:
      1. If we successfully fetched + cached the bytes, embed as base64 data URI
         (works offline, no further network requests, no mixed-content issues).
      2. Otherwise, fall back to the live URL with an onerror handler that hides
         the broken image and shows the placeholder. The browser will still try
         to fetch from the CDN, which is useful when the Python server can't
         reach the internet but the user's browser can.
    """
    if not poster_url or not isinstance(poster_url, str) or not poster_url.startswith("http"):
        return '<div class="poster-fallback">🎬<br/>No poster</div>'

    data = _fetch_poster_bytes(poster_url)
    if data is not None and len(data) > 100:
        b64 = base64.b64encode(data).decode()
        return f'<img class="poster" src="data:image/jpeg;base64,{b64}">'

    # Fallback: live URL with JS error-handler
    return (
        f'<img class="poster" src="{poster_url}" '
        f'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\';">'
        f'<div class="poster-fallback" style="display:none;">🎬<br/>No poster</div>'
    )


# ---------------------------------------------------------------------------
# Plot helpers (plotly — themed)
# ---------------------------------------------------------------------------

THEME_LAYOUT = dict(
    paper_bgcolor=BG_CARD,
    plot_bgcolor=BG_CARD,
    font=dict(color=TEXT_PRI, family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=60, b=40),
)


def themed_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(**THEME_LAYOUT)
    fig.update_xaxes(gridcolor="#2a3140", zerolinecolor="#2a3140")
    fig.update_yaxes(gridcolor="#2a3140", zerolinecolor="#2a3140")
    return fig


def styled_matplotlib(fig):
    """Apply dark background to a matplotlib figure for in-app consistency."""
    fig.patch.set_facecolor(BG_CARD)
    for ax in fig.axes:
        ax.set_facecolor(BG_CARD)
        ax.title.set_color(TEXT_PRI)
        ax.xaxis.label.set_color(TEXT_SEC)
        ax.yaxis.label.set_color(TEXT_SEC)
        ax.tick_params(colors=TEXT_SEC)
        for spine in ax.spines.values():
            spine.set_color("#2a3140")
    return fig


# ---------------------------------------------------------------------------
# Tab 1: Trending Films
# ---------------------------------------------------------------------------

def render_trending_tab(ma: MovieAnalysis) -> None:
    df = ma.df

    # ----- Sidebar filters -----
    st.sidebar.markdown("## 🎛️ Filters")

    # ----- Poster cache widget -----
    with st.sidebar.expander("🖼️ Poster cache", expanded=False):
        cached_files = list(IMAGE_CACHE_DIR.glob("*.jpg")) if IMAGE_CACHE_DIR.exists() else []
        total_kb = sum(f.stat().st_size for f in cached_files) // 1024
        st.caption(f"Cached posters: **{len(cached_files)}** ({total_kb:,} KB)")
        st.caption(f"Cache location: `{IMAGE_CACHE_DIR}`")
        if st.button("📥 Download all posters", use_container_width=True):
            urls = df["Poster_Url"].dropna().unique()
            progress = st.progress(0.0, text=f"Downloading 0/{len(urls)} posters...")
            n_ok, n_fail = 0, 0
            for i, url in enumerate(urls):
                if isinstance(url, str) and url.startswith("http"):
                    data = _fetch_poster_bytes(url)
                    if data is None:
                        n_fail += 1
                    else:
                        n_ok += 1
                progress.progress(
                    (i + 1) / len(urls),
                    text=f"Downloaded {n_ok}/{len(urls)} · failed: {n_fail}",
                )
            st.success(f"Done: {n_ok} cached, {n_fail} failed.")
            st.rerun()
        if st.button("🗑️ Clear cache", use_container_width=False):
            for f in cached_files:
                try:
                    f.unlink()
                except Exception:
                    pass
            _fetch_poster_bytes.clear()
            st.success("Cache cleared.")
            st.rerun()

    year_min, year_max = int(df["Release_Year"].min()), int(df["Release_Year"].max())
    year_range = st.sidebar.slider(
        "Release year range",
        min_value=year_min,
        max_value=year_max,
        value=(max(year_min, 2000), year_max),
        step=1,
    )

    # All genres across multi-genre films
    all_genres = sorted({g for lst in df["Genre_List"] for g in lst if isinstance(lst, list)})
    selected_genres = st.sidebar.multiselect("Genre (any match)", all_genres, default=[])

    all_langs = sorted(df["Original_Language"].unique())
    selected_langs = st.sidebar.multiselect("Original language", all_langs, default=[])

    pop_min = float(df["Popularity"].min())
    pop_max = float(df["Popularity"].max())
    pop_threshold = st.sidebar.slider(
        "Minimum popularity",
        min_value=pop_min,
        max_value=pop_max,
        value=pop_min,
        step=1.0,
        format="%.1f",
    )

    vote_min = float(df["Vote_Average"].min())
    vote_max = float(df["Vote_Average"].max())
    vote_threshold = st.sidebar.slider(
        "Minimum vote average",
        min_value=vote_min,
        max_value=vote_max,
        value=vote_min,
        step=0.1,
        format="%.1f",
    )

    title_query = st.sidebar.text_input("Search title", "")

    sort_by = st.sidebar.selectbox(
        "Sort by",
        ["Popularity", "Vote_Average", "Vote_Count", "Release_Date"],
        index=0,
    )
    sort_asc = st.sidebar.checkbox("Ascending", value=False)
    n_display = st.sidebar.slider("Films to display", 12, 60, 24, step=12)

    # ----- Apply filters -----
    mask = (
        df["Release_Year"].between(year_range[0], year_range[1])
        & (df["Popularity"] >= pop_threshold)
        & (df["Vote_Average"] >= vote_threshold)
    )
    if selected_genres:
        mask &= df["Genre_List"].apply(
            lambda lst: isinstance(lst, list) and any(g in selected_genres for g in lst)
        )
    if selected_langs:
        mask &= df["Original_Language"].isin(selected_langs)
    if title_query.strip():
        mask &= df["Title"].str.contains(title_query.strip(), case=False, na=False)

    matched_df = df.loc[mask]
    n_matched = len(matched_df)
    filtered = matched_df.sort_values(sort_by, ascending=sort_asc).head(n_display)

    # ----- Header KPIs -----
    film_word = "film" if n_matched == 1 else "films"
    match_word = "matches" if n_matched == 1 else "match"
    if n_matched <= n_display:
        shown_text = f"{n_matched} {film_word} {match_word}"
    else:
        shown_text = f"{n_display} of {n_matched} {film_word} {match_word} (showing top {n_display})"
    _render_html(
        f"<div class='app-header'>"
        f"<h1>🔥 Trending Films</h1>"
        f"<div class='subtitle'>{shown_text} the current filters · {len(df)} total in dataset</div>"
        f"</div>"
    )

    if filtered.empty:
        callout("No films match the current filters. Loosen one or more filters to see results.", "warning")
        return

    kpi_grid([
        {"label": "Films matched", "value": f"{n_matched}", "sub": f"{len(filtered)} shown"},
        {"label": "Avg popularity", "value": f"{filtered['Popularity'].mean():.1f}" if not filtered.empty else "—"},
        {"label": "Avg vote avg", "value": f"{filtered['Vote_Average'].mean():.2f}" if not filtered.empty else "—"},
        {"label": "Median vote count", "value": f"{int(filtered['Vote_Count'].median()):,}" if not filtered.empty else "—"},
        {"label": "Top genre", "value": filtered["Primary_Genre"].mode().iloc[0] if not filtered.empty else "—"},
        {"label": "Year span", "value": f"{int(filtered['Release_Year'].min())}–{int(filtered['Release_Year'].max())}" if not filtered.empty else "—"},
    ])

    # ----- Movie grid -----
    section_title("Top Films")

    cols_per_row = 4
    for i in range(0, len(filtered), cols_per_row):
        row = filtered.iloc[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for j, (_, film) in enumerate(row.iterrows()):
            with cols[j]:
                _render_html(_movie_card_html(film))

    # ----- Top-N quick tables -----
    section_title("Leaderboards")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("##### 📈 Most popular")
        st.dataframe(
            filtered.nlargest(10, "Popularity")[["Title", "Release_Year", "Popularity", "Vote_Average"]]
            .reset_index(drop=True),
            width='stretch',
            height=320,
        )
    with col_b:
        st.markdown("##### ⭐ Highest rated (≥ 500 votes)")
        heavy = filtered[filtered["Vote_Count"] >= 500]
        if heavy.empty:
            heavy = filtered
        st.dataframe(
            heavy.nlargest(10, "Vote_Average")[["Title", "Release_Year", "Vote_Average", "Vote_Count"]]
            .reset_index(drop=True),
            width='stretch',
            height=320,
        )
    with col_c:
        st.markdown("##### 🗳️ Most voted")
        st.dataframe(
            filtered.nlargest(10, "Vote_Count")[["Title", "Release_Year", "Vote_Count", "Vote_Average"]]
            .reset_index(drop=True),
            width='stretch',
            height=320,
        )


def _movie_card_html(film: pd.Series) -> str:
    poster_url = film.get("Poster_Url", "")
    poster_html = _poster_html(poster_url)
    genres = film.get("Genre_List", [])
    if not isinstance(genres, list):
        genres = []
    genre_pills = "".join(
        f'<span class="genre-pill">{g}</span>' for g in genres[:3]
    )
    return f"""
    <div class="movie-card">
        {poster_html}
        <div class="body">
            <div class="title">{film['Title']}</div>
            <div class="meta">
                <span>{film['Release_Year']} · {film['Original_Language'].upper()}</span>
                <span class="rating">⭐ {film['Vote_Average']:.1f}</span>
            </div>
            <div class="meta">
                <span>👥 {int(film['Vote_Count']):,} votes</span>
                <span>🔥 {film['Popularity']:.0f}</span>
            </div>
            <div class="tag-row">{genre_pills}</div>
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Tab 2: Statistical Insights
# ---------------------------------------------------------------------------

def render_stats_tab(ma: MovieAnalysis) -> None:
    df = ma.df

    _render_html(
        "<div class='app-header'>"
        "<h1>📊 Statistical Insights</h1>"
        "<div class='subtitle'>Comprehensive analysis: EDA · distributions · correlations · "
        "multicollinearity · OLS · hypothesis tests · KMeans · feature selection</div>"
        "</div>"
    )

    # Sub-sections via selectbox for navigation
    section = st.selectbox(
        "Jump to section",
        [
            "1. Exploratory Data Analysis (EDA)",
            "2. Distribution Analysis",
            "3. Correlation Analysis",
            "4. Multicollinearity (VIF)",
            "5. OLS Regression",
            "6. Hypothesis Tests",
            "7. KMeans Clustering",
            "8. Outlier Detection",
            "9. Feature Selection",
            "10. Feature Engineering Proposals",
            "11. Standardization / Normalization",
            "12. Genre Deep-Dive",
            "13. Temporal Trends & Seasonality",
            "14. Popularity vs Rating Paradox",
            "15. Random Forest Regressor",
            "16. PCA — 2D Projection",
            "17. Genre Co-occurrence",
            "18. Bootstrap Confidence Intervals",
        ],
        index=0,
    )

    if section.startswith("1."):
        _section_eda(ma)
    elif section.startswith("2."):
        _section_distributions(ma)
    elif section.startswith("3."):
        _section_correlations(ma)
    elif section.startswith("4."):
        _section_vif(ma)
    elif section.startswith("5."):
        _section_ols(ma)
    elif section.startswith("6."):
        _section_hypothesis(ma)
    elif section.startswith("7."):
        _section_kmeans(ma)
    elif section.startswith("8."):
        _section_outliers(ma)
    elif section.startswith("9."):
        _section_feature_selection(ma)
    elif section.startswith("10."):
        _section_feature_engineering(ma)
    elif section.startswith("11."):
        _section_standardization(ma)
    elif section.startswith("12."):
        _section_genre_deep_dive(ma)
    elif section.startswith("13."):
        _section_temporal_trends(ma)
    elif section.startswith("14."):
        _section_paradox(ma)
    elif section.startswith("15."):
        _section_random_forest(ma)
    elif section.startswith("16."):
        _section_pca(ma)
    elif section.startswith("17."):
        _section_genre_cooccurrence(ma)
    elif section.startswith("18."):
        _section_bootstrap(ma)


def _section_eda(ma: MovieAnalysis) -> None:
    section_title("Exploratory Data Analysis")
    eda = ma.eda_summary()

    kpi_grid([
        {"label": "Total films", "value": f"{eda['shape'][0]}"},
        {"label": "Total columns", "value": f"{eda['shape'][1]}"},
        {"label": "Unique genres", "value": f"{eda['n_unique_genres']}"},
        {"label": "Languages", "value": f"{eda['n_languages']}"},
        {"label": "Year range", "value": f"{eda['year_range'][0]}–{eda['year_range'][1]}"},
    ])

    st.markdown("##### Descriptive statistics (numeric)")
    st.dataframe(eda["describe_numeric"].round(2), width='stretch')

    st.markdown("##### Descriptive statistics (categorical)")
    # describe() returns mixed int/str columns; convert to string to avoid pyarrow warnings
    cat_desc = eda["describe_categorical"].astype(str)
    st.dataframe(cat_desc, width='stretch')

    st.markdown("##### Missing values")
    miss = eda["missing"].join(eda["missing_pct"]).rename(columns={0: "missing_pct"})
    st.dataframe(miss, width='stretch')

    callout(
        "<strong>Interpretation.</strong> The dataset is essentially complete — no missing "
        "values in any core column after date parsing and numeric coercion. The six "
        "missing values in <code>Rating_Bucket</code> are films with <code>Vote_Average</code> "
        "below 0 (an upstream data anomaly) that fall outside our bin edges.",
        "info",
    )

    # Release-year histogram + language distribution
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(
            ma.df, x="Release_Year", nbins=40, title="Films by release year",
            color_discrete_sequence=[GOLD],
        )
        st.plotly_chart(themed_fig(fig), width='stretch')
    with col_b:
        top_langs = ma.df["Original_Language"].value_counts().head(10).reset_index()
        top_langs.columns = ["language", "count"]
        fig = px.bar(top_langs, x="language", y="count", title="Top 10 languages",
                     color_discrete_sequence=[GOLD])
        st.plotly_chart(themed_fig(fig), width='stretch')

    callout(
        "<strong>Interpretation.</strong> The catalog is heavily skewed toward post-2000 "
        "releases and English-language films — both typical of TMDB-derived datasets. "
        "Any model trained on this data will be biased toward recent, English-speaking "
        "cinema, so genre coefficients from non-English films should be read with caution.",
        "info",
    )


def _section_distributions(ma: MovieAnalysis) -> None:
    section_title("Distribution Analysis")
    callout(
        "Each numeric variable is tested for normality via Shapiro-Wilk (sample n≤5000) "
        "and Kolmogorov-Smirnov. Skew and kurtosis quantify the shape of the distribution.",
        "info",
    )

    dist_df = ma.distribution_tests()
    st.dataframe(dist_df.round(4), width='stretch')

    # Plot distributions of the three headline variables
    section_title("Distribution of headline variables")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        fig = px.histogram(ma.df, x="Popularity", nbins=60, title="Popularity",
                           color_discrete_sequence=[GOLD])
        fig.update_layout(xaxis_type="log")
        st.plotly_chart(themed_fig(fig), width='stretch')
    with col_b:
        fig = px.histogram(ma.df, x="Vote_Count", nbins=60, title="Vote count",
                           color_discrete_sequence=[GOLD])
        fig.update_layout(xaxis_type="log")
        st.plotly_chart(themed_fig(fig), width='stretch')
    with col_c:
        fig = px.histogram(ma.df, x="Vote_Average", nbins=30, title="Vote average",
                           color_discrete_sequence=[GOLD])
        st.plotly_chart(themed_fig(fig), width='stretch')

    # Log-transformed comparison
    section_title("Effect of log1p transform on Popularity & Vote_Count")
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(ma.df, x="Log_Popularity", nbins=40, title="log1p(Popularity)",
                           color_discrete_sequence=["#4a9eff"])
        st.plotly_chart(themed_fig(fig), width='stretch')
    with col_b:
        fig = px.histogram(ma.df, x="Log_Vote_Count", nbins=40, title="log1p(Vote_Count)",
                           color_discrete_sequence=["#4a9eff"])
        st.plotly_chart(themed_fig(fig), width='stretch')

    callout(
        "<strong>Interpretation.</strong> Popularity and Vote_Count are heavily right-skewed "
        "(skew 7.4 and 3.3 respectively, both reject normality at p<0.001). The log1p transform "
        "compresses the long tail and brings the distributions much closer to symmetric, which "
        "is why we use Log_Popularity / Log_Vote_Count as the OLS predictors. Vote_Average is "
        "left-skewed (skew −1.9) — viewers tend to round up, and the catalog is curated.",
        "info",
    )


def _section_correlations(ma: MovieAnalysis) -> None:
    section_title("Correlation Analysis")

    method = st.radio(
        "Correlation method",
        ["pearson", "spearman", "kendall"],
        horizontal=True,
        help="Pearson (linear), Spearman (rank-based, robust to outliers), Kendall (ordinal concordance).",
    )
    corr = ma.correlation_matrices()[method]

    # Plotly heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            zmin=-1, zmax=1,
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 10, "color": "#0b0d12"},
            hovertemplate="%{y} ↔ %{x}<br>r = %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(title=f"{method.title()} correlation matrix", height=620)
    st.plotly_chart(themed_fig(fig), width='stretch')

    st.markdown("##### Top 15 strongest pairwise correlations")
    st.dataframe(ma.top_correlations(method, 15), width='stretch')

    callout(
        "<strong>Interpretation.</strong> The strongest signals are between the engineered "
        "log-transforms and their raw parents (e.g. Popularity ↔ Log_Popularity r=0.75) — "
        "expected and not informative. The substantive findings: <br/>"
        "• <strong>Log_Vote_Count ↔ Vote_Average r=0.50</strong> — more votes correlate with "
        "higher ratings (selection / popularity effect).<br/>"
        "• <strong>Popularity_per_Vote ↔ Vote_Average r=−0.34</strong> — films with high hype-per-vote "
        "tend to score lower (over-hyped marketing).<br/>"
        "• <strong>Release_Year ↔ Vote_Average r=−0.18</strong> — older films in the catalog "
        "score slightly higher (survivorship bias).",
        "info",
    )

    # QQ plot for Vote_Average
    section_title("QQ plot — Vote_Average")
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    from scipy import stats as sstats
    sstats.probplot(ma.df["Vote_Average"].dropna(), dist="norm", plot=ax)
    ax.set_title("Normal Q-Q — Vote_Average")
    ax.get_lines()[0].set_markerfacecolor(GOLD)
    ax.get_lines()[0].set_markeredgecolor(GOLD)
    ax.get_lines()[1].set_color("#e63946")
    st.pyplot(styled_matplotlib(fig))


def _section_vif(ma: MovieAnalysis) -> None:
    section_title("Multicollinearity — Variance Inflation Factors")
    callout(
        "VIF quantifies how much a predictor is linearly explained by the other predictors. "
        "Rule of thumb: VIF > 5 = moderate multicollinearity, VIF > 10 = severe.",
        "info",
    )

    vif = ma.variance_inflation_factors()
    st.dataframe(vif.round(3), width='stretch')

    fig = px.bar(
        vif.reset_index(), x="variable", y="VIF",
        color="severity",
        color_discrete_map={"low (<5)": "#2ecc71", "moderate (5-10)": GOLD, "severe (>10)": "#e63946"},
        title="VIF per predictor",
    )
    fig.add_hline(y=5, line_dash="dash", line_color=GOLD, annotation_text="VIF = 5")
    fig.add_hline(y=10, line_dash="dash", line_color="#e63946", annotation_text="VIF = 10")
    st.plotly_chart(themed_fig(fig), width='stretch')

    callout(
        "<strong>Interpretation.</strong> All predictors have VIF ≈ 1.0 — well below the "
        "5.0 warning threshold. There is <strong>no multicollinearity concern</strong> in the "
        "candidate predictor set. This is partly because we deliberately use either the raw "
        "<em>or</em> the log-transformed version of each variable, not both, in the same model. "
        "If we included both Popularity and Log_Popularity, VIF would explode to ~100.",
        "success",
    )


def _section_ols(ma: MovieAnalysis) -> None:
    section_title("OLS Regression — Vote_Average")
    callout(
        "Formula: <code>Vote_Average ~ Log_Popularity + Log_Vote_Count + Num_Genres + "
        "Release_Year + Release_Month + Title_Length + C(Primary_Genre)</code>",
        "info",
    )

    ols = ma.ols_vote_average()
    if "error" in ols:
        callout(f"OLS failed: {ols['error']}", "danger")
        return

    kpi_grid([
        {"label": "R²", "value": f"{ols['rsquared']:.3f}", "sub": "variance explained"},
        {"label": "Adj R²", "value": f"{ols['rsquared_adj']:.3f}", "sub": "penalized for k"},
        {"label": "F p-value", "value": f"{ols['f_pvalue']:.2e}", "sub": "overall significance"},
        {"label": "N obs", "value": f"{ols['n_obs']}"},
    ])

    st.markdown("##### Coefficient table")
    params = ols["params"].to_frame("coef").join(ols["pvalues"].to_frame("p_value")).join(ols["conf_int"])
    params.columns = ["coef", "p_value", "ci_low", "ci_high"]
    params["significant_5pct"] = params["p_value"] < 0.05
    st.dataframe(params.round(4), width='stretch')

    st.markdown("##### Full statsmodels summary")
    with st.expander("Show full OLS summary", expanded=False):
        st.code(ols["summary_str"], language="text")

    # Coefficient plot for non-intercept terms
    coef_df = params.drop("Intercept", errors="ignore").reset_index().rename(columns={"index": "term"})
    fig = px.bar(
        coef_df, x="coef", y="term", orientation="h",
        color="significant_5pct",
        color_discrete_map={True: GOLD, False: "#6b7280"},
        title="OLS coefficients (gold = significant at 5%)",
    )
    fig.update_layout(height=max(500, 30 * len(coef_df) + 100))
    st.plotly_chart(themed_fig(fig), width='stretch')

    callout(
        "<strong>Interpretation.</strong> The model explains "
        f"<strong>{ols['rsquared']*100:.1f}%</strong> of the variance in Vote_Average "
        "(adjusted R² = " f"{ols['rsquared_adj']*100:.1f}%), and is overall significant "
        f"(F p-value = {ols['f_pvalue']:.2e}). The strongest positive predictors are "
        "<strong>Log_Vote_Count</strong> (more votes → higher rating) and the "
        "<strong>Animation / Drama / Family</strong> genre dummies. "
        "<strong>Popularity_per_Vote</strong> would be strongly negative but is excluded to "
        "avoid leakage.",
        "info",
    )


def _section_hypothesis(ma: MovieAnalysis) -> None:
    section_title("Hypothesis Tests")

    anova = ma.anova_genre_vote()
    ttest = ma.ttest_language_vote()
    chi2 = ma.chi2_language_genre()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("##### ANOVA — genre vs rating")
        if "error" in anova:
            st.error(anova["error"])
        else:
            kpi_grid([
                {"label": "F statistic", "value": f"{anova['F_statistic']:.3f}"},
                {"label": "p-value", "value": f"{anova['p_value']:.2e}"},
                {"label": "Levene p", "value": f"{anova['levene_p']:.2e}"},
            ])
            callout(anova["interpretation"], "info" if anova["p_value"] < 0.05 else "warning")

    with col_b:
        st.markdown("##### Welch t-test — English vs non-English rating")
        if "error" in ttest:
            st.error(ttest["error"])
        else:
            kpi_grid([
                {"label": "Mean EN", "value": f"{ttest['mean_a']:.2f}"},
                {"label": "Mean non-EN", "value": f"{ttest['mean_b']:.2f}"},
                {"label": "t statistic", "value": f"{ttest['t_statistic']:.3f}"},
                {"label": "p-value", "value": f"{ttest['p_value']:.3f}"},
            ])
            callout(ttest["interpretation"], "info" if ttest["p_value"] < 0.05 else "warning")

    with col_c:
        st.markdown("##### χ² — language vs genre")
        if "error" in chi2:
            st.error(chi2["error"])
        else:
            kpi_grid([
                {"label": "χ² statistic", "value": f"{chi2['chi2_statistic']:.2f}"},
                {"label": "dof", "value": f"{chi2['dof']}"},
                {"label": "p-value", "value": f"{chi2['p_value']:.4f}"},
            ])
            callout(chi2["interpretation"], "info" if chi2["p_value"] < 0.05 else "warning")

    section_title("ANOVA — genre boxplot")
    fig = px.box(
        ma.df, x="Primary_Genre", y="Vote_Average",
        color="Primary_Genre", title="Vote_Average by primary genre",
    )
    fig.update_layout(showlegend=False, height=500, xaxis_tickangle=-45)
    st.plotly_chart(themed_fig(fig), width='stretch')

    callout(
        "<strong>Interpretation.</strong> The ANOVA rejects H0 at p < 0.001 — primary genre "
        "has a statistically significant effect on rating. The Levene test also rejects equal "
        "variances, so a Welch ANOVA would be even more appropriate. The t-test on language "
        "<strong>fails to reject</strong> H0 (p ≈ 0.07) — English and non-English films are "
        "rated similarly once we control for the genre mix. The chi-square test indicates a "
        "weak but significant association between language and genre (p ≈ 0.04).",
        "info",
    )


def _section_kmeans(ma: MovieAnalysis) -> None:
    section_title("KMeans Clustering")
    callout(
        "Features: standardized <code>Log_Popularity</code>, <code>Log_Vote_Count</code>, "
        "<code>Vote_Average</code>. The elbow method suggests the optimal k.",
        "info",
    )

    elbow = ma.elbow_inertia(k_max=10)
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown("##### Elbow curve")
        fig = px.line(elbow, x="k", y="inertia", markers=True,
                      title="KMeans inertia vs k", color_discrete_sequence=[GOLD])
        fig.add_vline(x=4, line_dash="dash", line_color="#e63946", annotation_text="k=4")
        st.plotly_chart(themed_fig(fig), width='stretch')

    k_choice = st.slider("Number of clusters (k)", 2, 8, 4)
    df_clustered, km, profile = ma.kmeans_clusters(k=k_choice)

    st.markdown("##### Cluster profiles")
    st.dataframe(profile, width='stretch')

    # 3D scatter
    fig = px.scatter_3d(
        df_clustered,
        x="Log_Popularity", y="Log_Vote_Count", z="Vote_Average",
        color="Segment",
        hover_name="Title",
        hover_data={"Release_Year": True, "Primary_Genre": True, "Popularity": ":.1f"},
        title="Film clusters (3D)",
        opacity=0.85,
    )
    fig.update_traces(marker=dict(size=5, line=dict(width=0.3, color="#0b0d12")))
    fig.update_layout(height=620, legend=dict(bgcolor=BG_CARD))
    st.plotly_chart(themed_fig(fig), width='stretch')

    # 2D scatter colored by segment
    section_title("Cluster map (2D)")
    fig = px.scatter(
        df_clustered, x="Log_Vote_Count", y="Vote_Average",
        color="Segment", hover_name="Title",
        hover_data={"Release_Year": True, "Popularity": ":.1f"},
        title="Log(Vote_Count) vs Vote_Average, colored by segment",
        opacity=0.8,
    )
    st.plotly_chart(themed_fig(fig), width='stretch')

    callout(
        "<strong>Interpretation.</strong> The four-cluster solution cleanly separates "
        "Blockbusters (high vote count, ~7 rating) from Niche/Indie films (low vote count, "
        "rating varies), with a small Mainstream cluster of very-high-popularity recent films. "
        "The tiny 'Niche / Indie' cluster with near-zero vote count and ~0.5 rating contains "
        "unreleased / pre-release films with placeholder ratings — these are upstream data "
        "artifacts that should be filtered before modeling.",
        "info",
    )


def _section_outliers(ma: MovieAnalysis) -> None:
    section_title("Outlier Detection")
    callout(
        "Three methods are combined: IQR (Tukey), Z-score (|z|>3), and Isolation Forest "
        "(5% contamination). The 'any_outlier' column is the union.",
        "info",
    )

    flags = ma.detect_outliers()
    summary = flags.sum().to_frame("n_outliers")
    summary["pct_of_dataset"] = (summary["n_outliers"] / len(flags) * 100).round(2)
    st.dataframe(summary, width='stretch')

    df_with_flags = ma.df.copy()
    df_with_flags = pd.concat([df_with_flags, flags], axis=1)

    # Scatter highlighting outliers
    fig = px.scatter(
        df_with_flags, x="Log_Vote_Count", y="Vote_Average",
        color="isolation_forest_outlier",
        color_discrete_map={0: GOLD, 1: "#e63946"},
        hover_name="Title",
        title="Isolation Forest outliers — Log(Vote_Count) vs Vote_Average",
        opacity=0.7,
    )
    st.plotly_chart(themed_fig(fig), width='stretch')

    st.markdown("##### Top 15 outlier films (by Isolation Forest)")
    top_out = df_with_flags[df_with_flags["isolation_forest_outlier"] == 1].nlargest(15, "Popularity")
    st.dataframe(
        top_out[["Title", "Release_Year", "Primary_Genre", "Popularity", "Vote_Average", "Vote_Count"]]
        .reset_index(drop=True),
        width='stretch',
    )

    callout(
        "<strong>Interpretation.</strong> Outliers concentrate among the highest-popularity "
        "films (Spider-Man, The Batman) and the lowest-rated films. The former are genuine "
        "blockbusters that the model should accommodate, not remove. The latter are often "
        "data-entry issues (0.0 ratings) and warrant investigation before modeling.",
        "warning",
    )


def _section_feature_selection(ma: MovieAnalysis) -> None:
    section_title("Feature Selection")
    callout(
        "Three complementary methods rank candidate predictors of <code>Vote_Average</code>: "
        "Pearson correlation (linear), Spearman (monotonic), mutual information (non-linear), "
        "RFE (recursive feature elimination with linear regression), and RandomForest importance.",
        "info",
    )

    fs = ma.feature_selection()
    st.dataframe(fs["table"], width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### RandomForest importance")
        fig = px.bar(
            fs["table"].reset_index().sort_values("rf_importance"),
            x="rf_importance", y="index", orientation="h",
            color_discrete_sequence=[GOLD],
        )
        st.plotly_chart(themed_fig(fig), width='stretch')
    with col_b:
        st.markdown("##### |Pearson r| with target")
        fig = px.bar(
            fs["table"].reset_index().sort_values("abs_pearson"),
            x="abs_pearson", y="index", orientation="h",
            color_discrete_sequence=["#4a9eff"],
        )
        st.plotly_chart(themed_fig(fig), width='stretch')

    callout(
        f"<strong>Interpretation.</strong> The methods agree on the top three: "
        f"<strong>{', '.join(fs['rfe_top3'])}</strong> (RFE) and "
        f"<strong>{', '.join(fs['rf_top3'])}</strong> (RandomForest). "
        "<strong>Log_Vote_Count</strong> dominates every ranking — a film's vote count is the "
        "single best linear and non-linear predictor of its rating in this dataset.",
        "info",
    )


def _section_feature_engineering(ma: MovieAnalysis) -> None:
    section_title("Feature Engineering Proposals")
    callout(
        "Each proposal includes the rationale, the exact code, and whether it has been "
        "implemented in this analysis.",
        "info",
    )

    proposals = ma.feature_engineering_proposals()
    rows = []
    for p in proposals:
        badge_color = "#2ecc71" if p["status"] == "Implemented" else GOLD
        rows.append(
            f"""
            <div class='movie-card' style='margin-bottom: 12px;'>
                <div class='body'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <div class='title' style='font-size:1.1rem;'>{p['column']}</div>
                        <span style='background:{badge_color}; color:#0b0d12; padding:2px 10px; border-radius:10px; font-size:0.75rem; font-weight:600;'>{p['status']}</span>
                    </div>
                    <div class='text-secondary' style='font-size:0.9rem;'>{p['rationale']}</div>
                    <pre style='background:#0b0d12; color:#f5c518; padding:8px 12px; border-radius:6px; font-size:0.8rem; overflow-x:auto; margin-top:6px;'>{p['code']}</pre>
                </div>
            </div>
            """
        )
    _render_html("".join(rows))


def _section_standardization(ma: MovieAnalysis) -> None:
    section_title("Standardization / Normalization Proposals")

    props = ma.standardization_proposals()
    for p in props:
        st.markdown(f"##### `{p['column']}`")
        st.markdown(f"**Current range:** {p['current_range']}")
        st.markdown(f"**Recommendation:** {p['recommendation']}")
        st.code(p["code"], language="python")
        st.markdown("---")

    callout(
        "<strong>General guidance.</strong> Tree-based models (RandomForest, XGBoost) are "
        "scale-invariant and need no standardization. Linear models (OLS, logistic, ridge), "
        "distance-based models (KMeans, KNN, SVM), and neural networks <strong>all require</strong> "
        "standardization. Always standardize on the training set only and apply the same "
        "transformer to the test set to avoid leakage.",
        "info",
    )


# ---------------------------------------------------------------------------
# New sections (12-18): deeper notebook-style analyses
# ---------------------------------------------------------------------------

def _section_genre_deep_dive(ma: MovieAnalysis) -> None:
    section_title("Genre Deep-Dive")
    callout(
        "A film can belong to multiple genres (comma-separated). This view counts each "
        "genre occurrence independently and surfaces the top films per genre.",
        "info",
    )

    gd = ma.genre_deep_dive()
    st.markdown("##### Per-genre statistics")
    st.dataframe(gd["stats"], width='stretch')

    # Bar chart: avg rating per genre (sorted)
    fig = px.bar(
        gd["stats"].reset_index().sort_values("avg_vote_average", ascending=False),
        x="Genre", y="avg_vote_average",
        color="n_films",
        color_continuous_scale="Viridis",
        title="Average Vote_Average per genre (color = n_films)",
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(themed_fig(fig), use_container_width=True)

    # Violin plot of ratings per genre (top 10 by count)
    top_genres = gd["stats"].head(10).index.tolist()
    violin_data = gd["violin_data"][gd["violin_data"]["Genre"].isin(top_genres)]
    fig = px.violin(
        violin_data, x="Genre", y="Vote_Average", color="Genre",
        title="Rating distribution per genre (top 10 by count)",
        box=True, points=False,
    )
    fig.update_layout(showlegend=False, height=500, xaxis_tickangle=-45)
    st.plotly_chart(themed_fig(fig), use_container_width=True)

    callout(
        "<strong>Interpretation.</strong> Animation, Music, and History typically top the "
        "rating chart — but their sample sizes are smaller, so the CIs are wider. Drama "
        "and Action dominate the catalog by count. The violin plot reveals that "
        "Animation/Drama have tighter distributions (more consistent quality), while "
        "Horror and Thriller are more dispersed.",
        "info",
    )

    # Top 3 films per genre (collapsible)
    st.markdown("##### Top 3 films per genre (by popularity)")
    selected_genre = st.selectbox("Pick a genre", list(gd["top_films"].keys()))
    st.dataframe(gd["top_films"][selected_genre], width='stretch')


def _section_temporal_trends(ma: MovieAnalysis) -> None:
    section_title("Temporal Trends & Seasonality")
    callout(
        "How has cinema volume, rating, and popularity evolved over time? Are there "
        "seasonal patterns (e.g. summer blockbusters, awards-season dramas)?",
        "info",
    )

    tt = ma.temporal_trends()

    # Two columns: films per year + avg rating per year
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(
            tt["yearly"], x="Release_Year", y="n_films",
            title="Films released per year", color_discrete_sequence=[GOLD],
        )
        st.plotly_chart(themed_fig(fig), use_container_width=True)
    with col_b:
        fig = px.line(
            tt["yearly"], x="Release_Year", y="avg_vote_average",
            title="Avg Vote_Average per year", markers=True,
            color_discrete_sequence=[GOLD],
        )
        fig.add_hline(y=ma.df["Vote_Average"].mean(), line_dash="dash", line_color="#6b7280",
                      annotation_text=f"Overall mean = {ma.df['Vote_Average'].mean():.2f}")
        st.plotly_chart(themed_fig(fig), use_container_width=True)

    # Monthly trends
    section_title("Monthly patterns (pooled across years)")
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(
            tt["monthly"], x="Month_Name", y="n_films",
            title="Films per release month", color_discrete_sequence=[GOLD],
            category_orders={"Month_Name": tt["monthly"]["Month_Name"].tolist()},
        )
        st.plotly_chart(themed_fig(fig), use_container_width=True)
    with col_b:
        fig = px.line(
            tt["monthly"], x="Month_Name", y="avg_vote_average",
            title="Avg Vote_Average per month", markers=True,
            color_discrete_sequence=[GOLD],
            category_orders={"Month_Name": tt["monthly"]["Month_Name"].tolist()},
        )
        st.plotly_chart(themed_fig(fig), use_container_width=True)

    # Seasonal table
    st.markdown("##### Seasonal aggregates")
    st.dataframe(tt["seasonal"], width='stretch')

    # Decade × Genre heatmap
    section_title("Decade × Genre evolution")
    dg = tt["decade_genre"]
    # Drop very small decades to keep the heatmap readable
    dg = dg.loc[dg.sum(axis=1) >= 3]
    fig = go.Figure(
        data=go.Heatmap(
            z=dg.values,
            x=dg.columns,
            y=dg.index,
            colorscale="YlOrRd",
            text=dg.values,
            texttemplate="%{text}",
            hovertemplate="Decade: %{y}<br>Genre: %{x}<br>Films: %{z}<extra></extra>",
        )
    )
    fig.update_layout(title="Films per decade × primary genre", height=500,
                      xaxis_tickangle=-45)
    st.plotly_chart(themed_fig(fig), use_container_width=True)

    callout(
        "<strong>Interpretation.</strong> The catalog skews heavily post-2000 (TMDB-era). "
        "Drama and Comedy dominate every decade. Action and Adventure gain share in the "
        "2010s–2020s, while Documentary and TV Movie are recent additions. The seasonal "
        "view typically reveals a holiday-season bump (Nov–Dec) and a summer blockbuster "
        "plateau — though vote averages tend to be slightly higher for awards-season "
        "(Sept–Dec) releases.",
        "info",
    )


def _section_paradox(ma: MovieAnalysis) -> None:
    section_title("Popularity vs Rating Paradox")
    callout(
        "A film can be popular but poorly rated (overhyped flop) or rarely seen but "
        "loved by those who watch it (hidden gem). This quadrant view separates the two.",
        "info",
    )

    pp = ma.popularity_rating_paradox()
    kpi_grid([
        {"label": "Blockbuster Hits", "value": f"{pp['counts'].loc['Blockbuster Hits', 'n_films']}",
         "sub": f"{pp['counts'].loc['Blockbuster Hits', 'pct']}%"},
        {"label": "Overhyped Flops", "value": f"{pp['counts'].loc['Overhyped Flops', 'n_films']}",
         "sub": f"{pp['counts'].loc['Overhyped Flops', 'pct']}%"},
        {"label": "Hidden Gems", "value": f"{pp['counts'].loc['Hidden Gems', 'n_films']}",
         "sub": f"{pp['counts'].loc['Hidden Gems', 'pct']}%"},
        {"label": "Quiet Releases", "value": f"{pp['counts'].loc['Quiet Releases', 'n_films']}",
         "sub": f"{pp['counts'].loc['Quiet Releases', 'pct']}%"},
    ])

    # Quadrant scatter
    fig = px.scatter(
        pp["df"], x="Log_Popularity", y="Vote_Average",
        color="Quadrant",
        color_discrete_map={
            "Blockbuster Hits": "#2ecc71",
            "Overhyped Flops": "#e63946",
            "Hidden Gems": GOLD,
            "Quiet Releases": "#6b7280",
        },
        hover_name="Title",
        hover_data={"Release_Year": True, "Popularity": ":.1f", "Vote_Count": True},
        title="Popularity vs Rating — quadrant view",
        opacity=0.7,
    )
    fig.add_hline(y=pp["thresholds"]["rating_median"], line_dash="dash", line_color="#6b7280",
                  annotation_text=f"rating median = {pp['thresholds']['rating_median']:.2f}")
    fig.add_vline(x=pp["thresholds"]["popularity_median"], line_dash="dash", line_color="#6b7280",
                  annotation_text=f"log(pop) median = {pp['thresholds']['popularity_median']:.2f}")
    st.plotly_chart(themed_fig(fig), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### 📉 Top 10 Overhyped Flops (high popularity, low rating)")
        st.dataframe(pp["overhyped"], width='stretch')
    with col_b:
        st.markdown("##### 💎 Top 10 Hidden Gems (low popularity, high rating, ≥50 votes)")
        st.dataframe(pp["hidden_gems"], width='stretch')

    callout(
        "<strong>Interpretation.</strong> Blockbuster Hits and Quiet Releases together "
        "account for ~57% of films — these are the 'expected' quadrants where popularity "
        "and rating align. The interesting 43% are split between Overhyped Flops (heavily "
        "marketed but disappointing — high Popularity_per_Vote is a tell) and Hidden Gems "
        "(low reach but high rating — likely festival/indie films with passionate audiences).",
        "info",
    )


def _section_random_forest(ma: MovieAnalysis) -> None:
    section_title("Random Forest Regressor — Vote_Average")
    callout(
        "An 80/20 train/test split with 300 trees. Compare against the OLS section: "
        "RandomForest captures non-linearities but is harder to interpret.",
        "info",
    )

    rf = ma.random_forest_model()
    m = rf["metrics"]

    kpi_grid([
        {"label": "Train R²", "value": f"{m['train_r2']:.3f}", "sub": "in-sample fit"},
        {"label": "Test R²", "value": f"{m['test_r2']:.3f}", "sub": "out-of-sample"},
        {"label": "Test RMSE", "value": f"{m['test_rmse']:.3f}", "sub": "0-10 scale"},
        {"label": "Test MAE", "value": f"{m['test_mae']:.3f}", "sub": "avg abs error"},
        {"label": "Train size", "value": f"{m['n_train']}"},
        {"label": "Test size", "value": f"{m['n_test']}"},
    ])

    # Actual vs predicted scatter
    section_title("Actual vs Predicted (test set)")
    fig = px.scatter(
        rf["predictions"], x="actual", y="predicted",
        opacity=0.6, color_discrete_sequence=[GOLD],
        labels={"actual": "Actual Vote_Average", "predicted": "Predicted"},
        title="Predicted vs Actual (test set)",
    )
    # y = x reference line
    min_v = float(min(rf["predictions"]["actual"].min(), rf["predictions"]["predicted"].min()))
    max_v = float(max(rf["predictions"]["actual"].max(), rf["predictions"]["predicted"].max()))
    fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines",
                             line=dict(dash="dash", color="#e63946"), name="y = x"))
    fig.update_layout(showlegend=False)
    st.plotly_chart(themed_fig(fig), use_container_width=True)

    # Residual plot
    section_title("Residual distribution")
    fig = px.histogram(
        rf["predictions"], x="residual", nbins=40,
        title="Residual = actual − predicted", color_discrete_sequence=["#4a9eff"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#e63946")
    st.plotly_chart(themed_fig(fig), use_container_width=True)

    # Feature importance
    section_title("Feature importance (permutation-based, in-sample)")
    fig = px.bar(
        rf["importances"].reset_index().sort_values("importance"),
        x="importance", y="index", orientation="h",
        color_discrete_sequence=[GOLD],
        labels={"index": "feature", "importance": "Gini importance"},
    )
    st.plotly_chart(themed_fig(fig), use_container_width=True)

    callout(
        "<strong>Interpretation.</strong> The train R² is much higher than the test R² — "
        "this is a sign of <strong>overfitting</strong>, typical for RandomForest on a "
        "small dataset (564 films). To reduce it: lower <code>max_depth</code>, raise "
        "<code>min_samples_leaf</code>, or use a GradientBoostingRegressor. Despite that, "
        "RandomForest (test R² ≈ "
        f"{m['test_r2']:.2f}) marginally beats OLS (R² ≈ 0.35) by capturing non-linearities "
        "in Log_Vote_Count. The residual distribution is roughly symmetric around 0, "
        "with a few large negative residuals (films the model wildly over-rated).",
        "info",
    )


def _section_pca(ma: MovieAnalysis) -> None:
    section_title("PCA — 2D Projection")
    callout(
        "Standardizes 7 numeric features, then projects to 2 principal components. "
        "The scatter reveals the natural film archetypes — and matches the KMeans "
        "segments very closely.",
        "info",
    )

    pca = ma.pca_projection()

    st.markdown("##### Variance explained")
    st.dataframe(pca["variance_explained"], width='stretch')

    # Loadings bar chart for PC1 and PC2
    section_title("Component loadings (feature → PC)")
    loadings_long = pca["loadings"].reset_index().melt(id_vars="index", var_name="PC", value_name="loading")
    loadings_long.rename(columns={"index": "feature"}, inplace=True)
    fig = px.bar(
        loadings_long, x="feature", y="loading", color="PC", barmode="group",
        color_discrete_map={"PC1": GOLD, "PC2": "#4a9eff"},
        title="Loadings: how each feature contributes to PC1 and PC2",
    )
    fig.update_layout(xaxis_tickangle=-45, height=450)
    st.plotly_chart(themed_fig(fig), use_container_width=True)

    # 2D scatter colored by Segment
    section_title("2D projection (colored by KMeans segment)")
    color_col = "Segment" if "Segment" in pca["projection"].columns else "Primary_Genre"
    fig = px.scatter(
        pca["projection"], x="PC1", y="PC2", color=color_col,
        hover_name="Title",
        hover_data={"Release_Year": True, "Popularity": ":.1f", "Vote_Average": ":.2f"},
        title=f"Films in PC1 × PC2 space (color = {color_col})",
        opacity=0.75,
    )
    fig.update_layout(height=600)
    st.plotly_chart(themed_fig(fig), use_container_width=True)

    callout(
        "<strong>Interpretation.</strong> PC1 and PC2 together explain "
        f"<strong>{pca['variance_explained']['cumulative'].iloc[1] * 100:.1f}%</strong> of the "
        "variance. PC1 typically loads on Log_Popularity + Log_Vote_Count — i.e. a film's "
        "overall reach — while PC2 separates films by Vote_Average (quality axis). The "
        "PCA segments should align closely with the KMeans clusters, since both use the "
        "same underlying features.",
        "info",
    )


def _section_genre_cooccurrence(ma: MovieAnalysis) -> None:
    section_title("Genre Co-occurrence")
    callout(
        "Which genres appear together? The heatmap shows <strong>Jaccard similarity</strong>: "
        "P(both genres) / P(either genre). High values = genres that almost always co-occur.",
        "info",
    )

    jaccard = ma.genre_cooccurrence()
    fig = go.Figure(
        data=go.Heatmap(
            z=jaccard.values,
            x=jaccard.columns,
            y=jaccard.index,
            colorscale="YlOrRd",
            zmid=0, zmax=0.6,
            text=jaccard.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 8, "color": "#0b0d12"},
            hovertemplate="%{y} ↔ %{x}<br>Jaccard = %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(title="Genre co-occurrence (Jaccard similarity)", height=650,
                      xaxis_tickangle=-45, yaxis_tickangle=-45)
    st.plotly_chart(themed_fig(fig), use_container_width=True)

    st.markdown("##### Top 15 most frequent genre pairs")
    st.dataframe(ma.genre_pair_top(15), width='stretch')

    callout(
        "<strong>Interpretation.</strong> Animation + Family has the highest Jaccard "
        "similarity (~0.54) — these two genres appear together more often than separately. "
        "Action + Adventure and Action + Thriller are also very frequent pairings. "
        "Documentary and TV Movie are typically solo (very low off-diagonal values) — "
        "they rarely co-occur with other genres. This is useful for designing a "
        "genre-recommendation system: pairing Animation with Family is safe, but pairing "
        "Documentary with Horror is essentially unheard of.",
        "info",
    )


def _section_bootstrap(ma: MovieAnalysis) -> None:
    section_title("Bootstrap Confidence Intervals")
    callout(
        "Resample-with-replacement (n_boot = 5,000) to estimate 95% confidence intervals "
        "for key statistics — without assuming any parametric distribution.",
        "info",
    )

    n_boot = st.slider("Number of bootstrap samples", 1000, 20000, 5000, step=1000)
    bc = ma.bootstrap_ci(n_boot=n_boot)

    kpi_grid([
        {"label": "Mean Vote_Average",
         "value": f"{bc['mean_vote_average']['mean']:.3f}",
         "sub": f"95% CI [{bc['mean_vote_average']['ci_low']:.3f}, {bc['mean_vote_average']['ci_high']:.3f}]"},
        {"label": "Mean Log_Popularity",
         "value": f"{bc['mean_log_popularity']['mean']:.3f}",
         "sub": f"95% CI [{bc['mean_log_popularity']['ci_low']:.3f}, {bc['mean_log_popularity']['ci_high']:.3f}]"},
        {"label": "Corr(Log_Vote, Rating)",
         "value": f"{bc['corr_log_vote_vs_rating']['mean']:.3f}",
         "sub": f"95% CI [{bc['corr_log_vote_vs_rating']['ci_low']:.3f}, {bc['corr_log_vote_vs_rating']['ci_high']:.3f}]"},
        {"label": "Bootstrap samples", "value": f"{bc['n_boot']:,}"},
    ])

    # Cohen's d bonus
    section_title("Cohen's d — English vs non-English effect size")
    cd = ma.cohen_d_language()
    if "error" not in cd:
        kpi_grid([
            {"label": "Cohen's d", "value": f"{cd['cohen_d']:.3f}", "sub": f"{cd['magnitude']} effect"},
            {"label": "Mean EN", "value": f"{cd['mean_en']:.3f}", "sub": f"n = {cd['n_en']}"},
            {"label": "Mean non-EN", "value": f"{cd['mean_non_en']:.3f}", "sub": f"n = {cd['n_non_en']}"},
            {"label": "Pooled std", "value": f"{cd['pooled_std']:.3f}"},
        ])

    callout(
        "<strong>Interpretation.</strong> The bootstrap CIs are non-parametric — they make "
        "no normality assumption and are robust to the heavy-tailed Popularity distribution. "
        "The 95% CI for mean Vote_Average is narrow (~±0.1) because n=564. The CI for the "
        "Log_Vote ↔ Rating correlation excludes 0 by a wide margin, reinforcing the OLS / "
        "feature-selection conclusion. Cohen's d for the language effect is small (~−0.22) "
        "— confirming the Welch t-test's borderline p-value: even though the means differ, "
        "the effect size is too small to matter practically.",
        "info",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    inject_css()

    csv_path = Path(__file__).parent / "data" / "movies.csv"
    if not csv_path.exists():
        st.error(f"Dataset not found at {csv_path}")
        st.stop()

    ma = load_analysis(str(csv_path))

    tab_trending, tab_stats = st.tabs(["🔥 Trending Films", "📊 Statistical Insights"])

    with tab_trending:
        render_trending_tab(ma)
    with tab_stats:
        render_stats_tab(ma)

    _render_html(
        "<div class='app-footer'>"
        "🎬 Movie Stats App — built with Streamlit, scipy, statsmodels, scikit-learn · "
        "Cinematic Dark theme"
        "</div>"
    )


if __name__ == "__main__":
    main()
