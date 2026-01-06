import ast
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD  # good for sparse matrices [web:228]

st.set_page_config(page_title="SO 2025 Developer Personas", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def safe_list(x):
    """Fix: CSV stores lists as strings. Convert back safely."""
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        x = x.strip()
        # if it's like "['a','b']"
        if x.startswith("[") and x.endswith("]"):
            try:
                return ast.literal_eval(x)
            except Exception:
                return [x]
        # if it's pipe-separated fallback
        if " | " in x:
            return x.split(" | ")
        return [x]
    return [str(x)]

def pct_table(df, col, cluster_col="cluster", top_n=8):
    """Percent distribution per cluster for a categorical column."""
    top_levels = df[col].value_counts(dropna=True).head(top_n).index
    tmp = df[df[col].isin(top_levels)].copy()
    if tmp.empty:
        return pd.DataFrame()
    return pd.crosstab(tmp[cluster_col], tmp[col], normalize="index").round(3)

@st.cache_data
def load_data():
    persona = pd.read_csv("cluster_persona_report.csv")
    users = pd.read_csv("stack_overflow_2025_segmented_users.csv")
    return persona, users

@st.cache_data
def build_projection(users_df, feature_cols, cluster_col="cluster", sample_n=12000, seed=42):
    """2D projection for nice scatter plot using preprocessing + TruncatedSVD."""
    df = users_df.copy()

    # sample for speed
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=seed)

    # keep selected columns that exist
    cols = [c for c in feature_cols if c in df.columns]
    df = df[cols + [cluster_col]].copy()

    # split types
    num_cols = df[cols].select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in cols if c not in num_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))  # keep safe if sparse later
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols)
        ],
        remainder="drop"
    )

    X = preprocess.fit_transform(df[cols])

    # TruncatedSVD works well on sparse, one-hot encoded matrices [web:228]
    svd = TruncatedSVD(n_components=2, random_state=seed)
    Z = svd.fit_transform(X)

    out = pd.DataFrame({
        "dim1": Z[:, 0],
        "dim2": Z[:, 1],
        "cluster": df[cluster_col].values
    })
    return out, svd.explained_variance_ratio_.sum()

# -----------------------------
# App
# -----------------------------
persona, users = load_data()

st.title("Stack Overflow 2025 Developer Personas 🧑‍💻✨")
st.caption("Interactive segmentation dashboard (K-Means clusters) with visuals + persona insights.")

# ---- Sidebar filters
st.sidebar.header("Controls")
cluster_ids = sorted(users["cluster"].dropna().unique().tolist())
selected_cluster = st.sidebar.selectbox("Select cluster", cluster_ids, index=0)

# Optional filters
show_country = st.sidebar.checkbox("Enable Country filter", value=False)
country = None
if show_country and "Country" in users.columns:
    country = st.sidebar.selectbox("Country", sorted(users["Country"].dropna().unique().tolist()))

df_view = users.copy()
if country is not None:
    df_view = df_view[df_view["Country"] == country]

# ---- KPIs
colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Total users", f"{len(df_view):,}")
with colB:
    st.metric("Clusters", f"{len(cluster_ids)}")
with colC:
    st.metric("Selected cluster", str(selected_cluster))
with colD:
    st.metric("Cluster size", f"{(df_view['cluster']==selected_cluster).sum():,}")

st.divider()

# -----------------------------
# Section: Cluster size visual
# -----------------------------
st.subheader("1) Cluster distribution")
cluster_counts = df_view["cluster"].value_counts().sort_index().reset_index()
cluster_counts.columns = ["cluster", "count"]
fig_counts = px.bar(cluster_counts, x="cluster", y="count", text="count",
                    title="Users per cluster", color="cluster")
st.plotly_chart(fig_counts, use_container_width=True)  # Plotly charts in Streamlit [web:230]

# -----------------------------
# Section: Experience visuals
# -----------------------------
st.subheader("2) Experience comparison")
exp_cols = [c for c in ["WorkExp", "YearsCode"] if c in df_view.columns]
if exp_cols:
    # Use numeric only if already numeric; otherwise show value counts
    df_sel = df_view[df_view["cluster"].isin(cluster_ids)].copy()
    for c in exp_cols:
        if pd.api.types.is_numeric_dtype(df_sel[c]):
            tmp = df_sel.groupby("cluster")[c].median().reset_index()
            fig = px.bar(tmp, x="cluster", y=c, title=f"Median {c} by cluster", color="cluster")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # categorical years bins - show percent table as heatmap-like
            pt = pct_table(df_sel, c, top_n=8)
            if not pt.empty:
                fig = px.imshow(pt, text_auto=True, aspect="auto", title=f"{c} distribution (top categories)")
                st.plotly_chart(fig, use_container_width=True)
else:
    st.info("WorkExp/YearsCode not available in this exported file.")

# -----------------------------
# Section: SO + AI distributions
# -----------------------------
st.subheader("3) Stack Overflow & AI behavior")
focus_cols = [c for c in ["SOVisitFreq", "SODuration", "SOPartFreq", "SOFriction", "AISent", "AIAcc", "AIComplex", "AIFrustration"] if c in df_view.columns]
if focus_cols:
    pick = st.selectbox("Pick a column to visualize", focus_cols)
    pt = pct_table(df_view, pick, top_n=10)
    if not pt.empty:
        fig = px.imshow(pt, text_auto=True, aspect="auto", title=f"{pick}: % distribution per cluster")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.histogram(df_view, x=pick, color="cluster", barmode="group", title=f"{pick} distribution")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("SO/AI columns not found in exported file. Add them to df_out2 before saving if needed.")

# -----------------------------
# Section: Persona card
# -----------------------------
st.subheader("4) Persona insights")

# try to map persona table by cluster
if "cluster" in persona.columns:
    row = persona[persona["cluster"] == selected_cluster]
    if not row.empty:
        row = row.iloc[0]
        persona_name = row["persona_name"] if "persona_name" in persona.columns else f"Cluster {selected_cluster}"
        st.markdown(f"### Persona: {persona_name}")

        left, right = st.columns([1, 2])
        with left:
            if "size" in persona.columns:
                st.metric("Persona size", f"{int(row['size']):,}")
            if "WorkExp" in persona.columns:
                st.metric("Median WorkExp", str(row["WorkExp"]))
            if "YearsCode" in persona.columns:
                st.metric("Median YearsCode", str(row["YearsCode"]))

        with right:
            # FIX: tokens list stored as string in CSV -> convert back
            tokens = safe_list(row["top_tokens"]) if "top_tokens" in persona.columns else []
            st.markdown("**Top tech tokens**")
            if tokens:
                for t in tokens[:12]:
                    st.write(f"• {t}")
            else:
                st.info("No tokens found in persona report.")
    else:
        st.warning("Selected cluster not found in persona report.")
else:
    st.warning("persona report must include 'cluster' column.")

# -----------------------------
# Section: 2D projection scatter (nice visual!)
# -----------------------------
st.subheader("5) 2D projection of respondents (SVD)")

# choose a small set of columns to project
projection_features = [
    "WorkExp", "YearsCode",
    "SOVisitFreq", "SODuration", "SOPartFreq",
    "AISent", "AIAcc",
    "DevType", "RemoteWork", "Employment"
]

if "cluster" in users.columns:
    proj_df, explained = build_projection(users, projection_features, sample_n=10000)
    fig_scatter = px.scatter(
        proj_df, x="dim1", y="dim2", color="cluster",
        title=f"2D projection (TruncatedSVD) — explained variance approx: {explained:.2%}",
        opacity=0.6
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("No 'cluster' column in users file.")
