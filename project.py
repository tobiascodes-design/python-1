"""
CORD-19 Metadata Explorer
Single-file script that performs:
 - Loading (with optional download) of metadata.csv
 - Cleaning & preprocessing
 - Analysis & visualizations (saves PNGs to ./outputs)
 - Streamlit app to explore interactively

Usage:
 1) Place metadata.csv in the same folder OR allow the script to attempt a download.
 2) Run analyses (and save plots) with:
      python cord19_full_pipeline.py
 3) Run the Streamlit app with:
      streamlit run cord19_full_pipeline.py

Dependencies:
  pandas, numpy, matplotlib, seaborn, requests, wordcloud, streamlit
  Install with: pip install pandas numpy matplotlib seaborn requests wordcloud streamlit

Note: the script is defensive (try/except) and will skip features if optional libs are missing.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Core libs
try:
    import pandas as pd
    import numpy as np
except Exception as e:
    print("Missing core libraries. Please install pandas and numpy.")
    raise

# Optional plotting libs
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")
    HAS_PLOTTING = True
except Exception:
    HAS_PLOTTING = False

# Optional extras
try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except Exception:
    HAS_WORDCLOUD = False

# Streamlit will be imported only when running the app; keep optional
try:
    import streamlit as st
    HAS_STREAMLIT = True
except Exception:
    HAS_STREAMLIT = False

# Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("cord19")

# Paths
DATA_FILENAME = "metadata.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default: simple safe content types for potential download (not strictly necessary for metadata)
CORD19_METADATA_URLS = [
    # Known mirrors may change — this is a best-effort fallback. If this fails, download metadata.csv manually.
    "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2022-12-19/metadata.csv",
    "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2021-08-13/metadata.csv",
]

# ----------------------
# Utilities
# ----------------------

def try_download_metadata(target_path=DATA_FILENAME):
    """Attempt to download metadata.csv from a list of candidate URLs.
    This is best-effort and will not raise on failure; returns True if a file exists at the end.
    """
    if os.path.exists(target_path):
        log.info(f"Found existing {target_path}, skipping download.")
        return True

    if not HAS_REQUESTS:
        log.warning("requests not available — cannot attempt download. Please place metadata.csv in the script folder.")
        return False

    for url in CORD19_METADATA_URLS:
        try:
            log.info(f"Attempting download from: {url}")
            r = requests.get(url, stream=True, timeout=30)
            if r.status_code == 200:
                with open(target_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                log.info("Download complete.")
                return True
            else:
                log.warning(f"Server responded with status {r.status_code} for {url}")
        except Exception as e:
            log.warning(f"Download failed for {url}: {e}")
    log.warning("All download attempts failed — please obtain metadata.csv manually and place in the script directory.")
    return False


# ----------------------
# Data pipeline
# ----------------------

def load_metadata(path=DATA_FILENAME, nrows=None, low_memory=True):
    """Load metadata.csv into a pandas DataFrame with safe options and error handling."""
    if not os.path.exists(path):
        log.info(f"{path} not found on disk. Trying download...")
        try_download_metadata(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Please download metadata.csv and place it in the working directory.")

    try:
        df = pd.read_csv(path, nrows=nrows, low_memory=low_memory)
        log.info(f"Loaded metadata: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        log.error(f"Error reading CSV: {e}")
        raise


def explore_dataframe(df):
    """Quick overview prints and returns basic derived info."""
    log.info("\n--- Data preview (first 5 rows) ---")
    log.info(df.head().to_string())

    log.info("\n--- DataFrame shape ---")
    log.info(str(df.shape))

    log.info("\n--- Data types ---")
    log.info(df.dtypes)

    log.info("\n--- Missing values (top 20 columns) ---")
    mv = df.isnull().sum().sort_values(ascending=False).head(20)
    log.info(mv.to_string())

    log.info("\n--- Basic stats (numerical) ---")
    log.info(df.describe().to_string())

    return {
        "shape": df.shape,
        "missing_top": mv,
    }


def clean_metadata(df):
    """Clean the metadata DataFrame:
    - Parse dates
    - Create 'year' column
    - Create abstract word count
    - Drop extremely sparse columns (configurable)
    - Basic normalization for journal/source
    """
    df = df.copy()

    # Drop columns with >80% missing values (configurable)
    thresh = int(0.2 * len(df))  # keep columns with at least 20% non-null
    drop_cols = [c for c in df.columns if df[c].notnull().sum() < thresh]
    if drop_cols:
        log.info(f"Dropping {len(drop_cols)} very sparse columns: {drop_cols[:10]}...")
        df.drop(columns=drop_cols, inplace=True)

    # Parse publish_time into datetime
    if "publish_time" in df.columns:
        df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
        df["year"] = df["publish_time"].dt.year
    else:
        df["year"] = np.nan

    # Abstract word count
    if "abstract" in df.columns:
        df["abstract_text"] = df["abstract"].fillna("").astype(str)
        df["abstract_word_count"] = df["abstract_text"].apply(lambda x: len(x.split()))
    else:
        df["abstract_word_count"] = np.nan

    # Normalize journal/source columns
    for col in ("journal", "source_x"):
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", np.nan)

    return df


# ----------------------
# Analysis & Visualization
# ----------------------


def save_or_show_plot(fig, filename, show=False):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    log.info(f"Saved plot: {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_publications_over_time(df):
    if not HAS_PLOTTING:
        log.warning("Plotting libraries not available; skipping time series plot.")
        return
    if "year" not in df.columns:
        log.warning("Year column missing; cannot plot publications over time.")
        return

    counts = df["year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(counts.index, counts.values, marker="o")
    ax.set_title("Publications by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Publications")
    save_or_show_plot(fig, "publications_by_year.png")


def plot_top_journals(df, top_n=15):
    if not HAS_PLOTTING:
        log.warning("Plotting libraries not available; skipping top journals plot.")
        return
    if "journal" not in df.columns:
        log.warning("Journal column missing; skipping top journals.")
        return

    top = df["journal"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    sns.barplot(x=top.values, y=top.index, ax=ax)
    ax.set_title(f"Top {top_n} Journals by Number of Publications")
    ax.set_xlabel("Count")
    save_or_show_plot(fig, "top_journals.png")


def plot_title_wordcloud(df, max_words=200):
    if not HAS_WORDCLOUD:
        log.warning("wordcloud not available; skipping word cloud.")
        return
    if "title" not in df.columns:
        log.warning("title column missing; skipping word cloud.")
        return

    text = " ".join(df["title"].dropna().astype(str).tolist()).lower()
    wc = WordCloud(width=1200, height=600, background_color="white", max_words=max_words).generate(text)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Frequent Words in Titles")
    save_or_show_plot(fig, "title_wordcloud.png")


def plot_source_distribution(df, top_n=10):
    if not HAS_PLOTTING:
        log.warning("Plotting libraries not available; skipping source distribution plot.")
        return
    if "source_x" not in df.columns:
        log.warning("source_x column missing; skipping source distribution.")
        return

    top = df["source_x"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    sns.barplot(x=top.values, y=top.index, ax=ax)
    ax.set_title("Top Sources by Paper Count")
    ax.set_xlabel("Count")
    save_or_show_plot(fig, "source_distribution.png")


# ----------------------
# Streamlit App
# ----------------------


def streamlit_app(df):
    """Defines a Streamlit app to explore the dataset interactively. Requires streamlit to be installed."""
    if not HAS_STREAMLIT:
        raise RuntimeError("Streamlit is not installed. Install it with `pip install streamlit` to run the app.`")

    st.set_page_config(page_title="CORD-19 Explorer", layout="wide")
    st.title("📊 CORD-19 Metadata Explorer")
    st.write("Interactive exploration of the CORD-19 metadata file (metadata.csv).")

    # Sidebar filters
    st.sidebar.header("Filters")
    min_year = int(df["year"].min()) if df["year"].notnull().any() else 0
    max_year = int(df["year"].max()) if df["year"].notnull().any() else 0
    yr_range = st.sidebar.slider("Select year range", min_value=min_year, max_value=max_year, value=(min_year, max_year))

    journals = df["journal"].dropna().unique().tolist() if "journal" in df.columns else []
    sel_journal = st.sidebar.selectbox("Filter by journal (optional)", options=["All"] + sorted(journals)[:500])

    source_opts = df["source_x"].dropna().unique().tolist() if "source_x" in df.columns else []
    sel_source = st.sidebar.selectbox("Filter by source (optional)", options=["All"] + sorted(source_opts)[:500])

    filtered = df.copy()
    filtered = filtered[(filtered["year"] >= yr_range[0]) & (filtered["year"] <= yr_range[1])]
    if sel_journal != "All":
        filtered = filtered[filtered["journal"] == sel_journal]
    if sel_source != "All":
        filtered = filtered[filtered["source_x"] == sel_source]

    st.sidebar.markdown(f"**Filtered results:** {len(filtered)} rows")

    # Main layout
    st.subheader("Sample of filtered data")
    st.dataframe(filtered.head(50))

    # Publications by year
    if "year" in filtered.columns:
        st.subheader("Publications by year")
        year_counts = filtered["year"].value_counts().sort_index()
        st.bar_chart(year_counts)

    # Top journals
    if "journal" in filtered.columns:
        st.subheader("Top journals (filtered)")
        top_j = filtered["journal"].value_counts().head(15)
        st.bar_chart(top_j)

    # Wordcloud
    if HAS_WORDCLOUD and "title" in filtered.columns:
        st.subheader("Title word cloud")
        titles = " ".join(filtered["title"].dropna().astype(str).tolist())
        wc = WordCloud(width=800, height=400, background_color="white").generate(titles)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    # Download sample
    st.markdown("---")
    st.subheader("Download a sample CSV")
    st.download_button("Download filtered sample (CSV)", data=filtered.head(200).to_csv(index=False), file_name="cord19_sample.csv")


# ----------------------
# Main runnable flow
# ----------------------

def main(args):
    # Load a small subset by default for speed unless user asks full file
    nrows = None if args.full else 200000  # if metadata is huge, default to limit

    df = load_metadata(nrows=nrows)

    info = explore_dataframe(df)

    df_clean = clean_metadata(df)

    # Re-explore after cleaning
    log.info("\n--- After cleaning ---")
    explore_dataframe(df_clean)

    # Analysis & plots
    plot_publications_over_time(df_clean)
    plot_top_journals(df_clean, top_n=20)
    plot_title_wordcloud(df_clean)
    plot_source_distribution(df_clean)

    # Save a small cleaned CSV for convenience
    cleaned_path = os.path.join(OUTPUT_DIR, "metadata_cleaned_sample.csv")
    df_clean.head(5000).to_csv(cleaned_path, index=False)
    log.info(f"Saved cleaned sample CSV: {cleaned_path}")

    if args.run_streamlit:
        if not HAS_STREAMLIT:
            log.error("Streamlit is not installed. Install it with `pip install streamlit` to run the app.")
        else:
            log.info("Launching Streamlit app... (this will block until the app is stopped)")
            # When using streamlit, the recommended way is to run `streamlit run file.py` from the CLI.
            # But we also provide a programmatic entrypoint for convenience.
            try:
                streamlit_app(df_clean)
            except Exception as e:
                log.error(f"Streamlit app error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CORD-19 metadata explorer - single-file pipeline")
    parser.add_argument("--full", action="store_true", help="Load the full metadata.csv (might be large). By default a large but limited subset is used for speed.")
    parser.add_argument("--streamlit", dest="run_streamlit", action="store_true", help="Run the streamlit app after analysis (requires streamlit).")
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        log.error(f"Fatal error: {e}")
        sys.exit(1)

# End of file
