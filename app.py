import ast
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD

# -----------------------------
# 1. Page Config & Professional UI Setup
# -----------------------------
st.set_page_config(
    page_title="Dev Personas 2025 | Analytics",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# 2. THEME ENGINE (CSS & Styling) 🎨
# -----------------------------
# This is where the magic happens. We inject CSS to override Streamlit's defaults.
st.markdown("""
    <style>
    /* MAIN BACKGROUND - Deep Dark Grey */
    .stApp {
        background-color: #0e1117;
        background-image: radial-gradient(circle at 50% 0%, #2b204a 0%, #0e1117 60%);
    }

    /* TYPOGRAPHY */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #ffffff;
        font-weight: 700;
        text-shadow: 0px 0px 10px rgba(162, 155, 254, 0.3);
    }
    p, label, .stMarkdown {
        color: #b2bec3;
    }

    /* GLASSMORPHISM CARDS for Metrics */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: #a29bfe;
    }
    div[data-testid="stMetricLabel"] {
        color: #a29bfe !important;
        font-size: 0.9rem;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        text-shadow: 0 0 10px rgba(162, 155, 254, 0.5);
    }

    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: #11141d;
        border-right: 1px solid rgba(255,255,255, 0.05);
    }

    /* FOOTER STYLING */
    .footer {
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        color: #636e72;
    }
    .social-link {
        color: #a29bfe;
        text-decoration: none;
        font-weight: bold;
        margin: 0 15px;
        font-size: 1.1rem;
        transition: color 0.3s;
    }
    .social-link:hover {
        color: #fff;
        text-shadow: 0 0 8px #a29bfe;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# 3. Helpers & Data Loading
# -----------------------------
def safe_list(x):
    """Fix: CSV stores lists as strings. Convert back safely."""
    if isinstance(x, list): return x
    if pd.isna(x): return []
    if isinstance(x, str):
        x = x.strip()
        if x.startswith("[") and x.endswith("]"):
            try: return ast.literal_eval(x)
            except Exception: return [x]
        if " | " in x: return x.split(" | ")
        return [x]
    return [str(x)]

def pct_table(df, col, cluster_col="cluster", top_n=8):
    """Percent distribution per cluster for a categorical column."""
    top_levels = df[col].value_counts(dropna=True).head(top_n).index
    tmp = df[df[col].isin(top_levels)].copy()
    if tmp.empty: return pd.DataFrame()
    return pd.crosstab(tmp[cluster_col], tmp[col], normalize="index").round(3)

@st.cache_data
def load_data():
    # ⚠️ Ensure these files exist in your folder
    try:
        persona = pd.read_csv("cluster_persona_report.csv")
        users = pd.read_csv("stack_overflow_2025_segmented_users.csv")
        return persona, users
    except FileNotFoundError:
        st.error("🚨 Error: CSV files not found. Please upload 'cluster_persona_report.csv' and 'stack_overflow_2025_segmented_users.csv'.")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data
def build_projection(users_df, feature_cols, cluster_col="cluster", sample_n=12000, seed=42):
    """2D projection for nice scatter plot using preprocessing + TruncatedSVD."""
    df = users_df.copy()
    if len(df) > sample_n: df = df.sample(sample_n, random_state=seed)
    
    cols = [c for c in feature_cols if c in df.columns]
    df = df[cols + [cluster_col]].copy()
    
    num_cols = df[cols].select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in cols if c not in num_cols]

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    
    preprocess = ColumnTransformer([("num", numeric_pipe, num_cols), ("cat", categorical_pipe, cat_cols)], remainder="drop")
    
    X = preprocess.fit_transform(df[cols])
    svd = TruncatedSVD(n_components=2, random_state=seed)
    Z = svd.fit_transform(X)
    
    out = pd.DataFrame({"dim1": Z[:, 0], "dim2": Z[:, 1], "cluster": df[cluster_col].values})
    return out, svd.explained_variance_ratio_.sum()

# Helper for Consistent Plot Styling
def update_plot_style(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#b2bec3"),
        title_font=dict(size=20, color="#ffffff"),
    )
    return fig

# -----------------------------
# 4. App Logic
# -----------------------------
persona, users = load_data()

if not users.empty:
    # ---- Sidebar filters ----
    st.sidebar.title("🎛️ Control Panel")
    st.sidebar.markdown("---")
    
    cluster_ids = sorted(users["cluster"].dropna().unique().tolist())
    selected_cluster = st.sidebar.selectbox("Select Target Cluster", cluster_ids, index=0)

    show_country = st.sidebar.checkbox("Enable Country Filter", value=False)
    country = None
    if show_country and "Country" in users.columns:
        country = st.sidebar.selectbox("Region / Country", sorted(users["Country"].dropna().unique().tolist()))

    df_view = users.copy()
    if country is not None:
        df_view = df_view[df_view["Country"] == country]

    # ---- HEADER ----
    st.title("Stack Overflow 2025 Developer Personas ⚡")
    st.markdown("""
    Welcome to the **Developer Segmentation Engine**. Explore behavioral clusters, AI adoption patterns, 
    and tech stacks using K-Means clustering.
    """)
    st.markdown("---")

    # ---- KEY METRICS (Glass Cards) ----
    colA, colB, colC, colD = st.columns(4)
    with colA: st.metric("Total Users", f"{len(df_view):,}", delta="Live Data")
    with colB: st.metric("Total Clusters", f"{len(cluster_ids)}")
    with colC: st.metric("Active Cluster", f"Type {selected_cluster}")
    with colD: st.metric("Cluster Size", f"{(df_view['cluster']==selected_cluster).sum():,}")

    st.markdown("<br>", unsafe_allow_html=True) # Spacer

    # ---- 1. CLUSTER DISTRIBUTION ----
    st.subheader("📊 Cluster Ecology")
    cluster_counts = df_view["cluster"].value_counts().sort_index().reset_index()
    cluster_counts.columns = ["cluster", "count"]
    
    # Custom color scale: Purples
    fig_counts = px.bar(
        cluster_counts, x="cluster", y="count", text="count",
        color="count", color_continuous_scale="Purples",
        title="Distribution of Developer Types"
    )
    fig_counts.update_traces(marker_line_color='rgba(255, 255, 255, 0.1)', marker_line_width=1.5)
    st.plotly_chart(update_plot_style(fig_counts), use_container_width=True)

    # ---- 2. EXPERIENCE & DEMOGRAPHICS ----
    st.subheader("🧠 Experience & Tenure")
    exp_cols = [c for c in ["WorkExp", "YearsCode"] if c in df_view.columns]
    
    if exp_cols:
        col1, col2 = st.columns(2)
        df_sel = df_view[df_view["cluster"].isin(cluster_ids)].copy()
        
        for idx, c in enumerate(exp_cols):
            # Alternate columns for layout
            target_col = col1 if idx % 2 == 0 else col2
            
            with target_col:
                if pd.api.types.is_numeric_dtype(df_sel[c]):
                    tmp = df_sel.groupby("cluster")[c].median().reset_index()
                    fig = px.bar(
                        tmp, x="cluster", y=c, 
                        title=f"Median {c}", 
                        color=c, color_continuous_scale="Purples"
                    )
                    st.plotly_chart(update_plot_style(fig), use_container_width=True)
                else:
                    pt = pct_table(df_sel, c, top_n=8)
                    if not pt.empty:
                        fig = px.imshow(
                            pt, text_auto=True, aspect="auto", 
                            title=f"{c} Heatmap", color_continuous_scale="Purples"
                        )
                        st.plotly_chart(update_plot_style(fig), use_container_width=True)

    # ---- 3. AI & SO BEHAVIOR ----
    st.markdown("---")
    st.subheader("🤖 AI Sentiment & Platform Usage")
    
    focus_cols = [c for c in ["SOVisitFreq", "SODuration", "AISent", "AIAcc", "AIComplex"] if c in df_view.columns]
    if focus_cols:
        pick = st.selectbox("Select Dimension to Analyze", focus_cols)
        
        pt = pct_table(df_view, pick, top_n=10)
        if not pt.empty:
            fig = px.imshow(
                pt, text_auto=True, aspect="auto", 
                title=f"Heatmap: {pick} vs Cluster",
                color_continuous_scale="Purples" # Keeping the theme consistent
            )
            st.plotly_chart(update_plot_style(fig), use_container_width=True)
        else:
            fig = px.histogram(
                df_view, x=pick, color="cluster", barmode="group", 
                title=f"Distribution: {pick}", color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(update_plot_style(fig), use_container_width=True)

    # ---- 4. PERSONA DEEP DIVE ----
    st.markdown("---")
    st.subheader(f"✨ Persona Deep Dive: Cluster {selected_cluster}")

    if "cluster" in persona.columns:
        row = persona[persona["cluster"] == selected_cluster]
        if not row.empty:
            row = row.iloc[0]
            persona_name = row.get("persona_name", f"Archetype {selected_cluster}")
            
            # Layout for Persona Card
            with st.container():
                p_col1, p_col2 = st.columns([1, 2])
                
                with p_col1:
                    st.markdown(f"""
                    <div style="background: rgba(162, 155, 254, 0.1); padding: 20px; border-radius: 10px; border: 1px solid #a29bfe;">
                        <h2 style="color: #a29bfe; margin:0;">{persona_name}</h2>
                        <hr style="border-color: rgba(255,255,255,0.2);">
                        <p style="font-size: 1.2rem; color: white;">👥 <b>Size:</b> {int(row.get('size', 0)):,}</p>
                        <p>💼 <b>Avg Exp:</b> {row.get('WorkExp', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with p_col2:
                    tokens = safe_list(row.get("top_tokens", []))
                    st.markdown("### 🛠️ Key Technology Stack")
                    
                    # Display chips for tokens
                    chips_html = ""
                    for t in tokens[:12]:
                        chips_html += f'<span style="display: inline-block; background-color: #2d3436; color: #dfe6e9; padding: 5px 12px; margin: 3px; border-radius: 15px; border: 1px solid #636e72; font-size: 0.85rem;">{t}</span>'
                    st.markdown(chips_html, unsafe_allow_html=True)

    # ---- 5. 2D PROJECTION ----
    st.markdown("---")
    st.subheader("🌌 2D Latent Space Projection (SVD)")
    st.caption("Visualizing high-dimensional user data in 2D space. Points colored by Cluster ID.")

    projection_features = [
        "WorkExp", "YearsCode", "SOVisitFreq", "SODuration", 
        "SOPartFreq", "AISent", "AIAcc", "DevType"
    ]

    if "cluster" in users.columns:
        with st.spinner("Crunching the numbers with SVD..."):
            proj_df, explained = build_projection(users, projection_features, sample_n=10000)
        
        fig_scatter = px.scatter(
            proj_df, x="dim1", y="dim2", color="cluster",
            title=f"User Manifold (Variance Explained: {explained:.2%})",
            color_continuous_scale="Viridis", # Nice contrast for scatter
            opacity=0.7
        )
        fig_scatter.update_traces(marker=dict(size=4))
        st.plotly_chart(update_plot_style(fig_scatter), use_container_width=True)

    # ---- FOOTER ----
    st.markdown("""
    <div class="footer">
        <p>Built with ❤️ and Python by an Ambitious Data Scientist</p>
        <p>
            <a href="https://www.linkedin.com/in/mayank-goyal-mg09/" target="_blank" class="social-link">🔗 LinkedIn</a> | 
            <a href="https://github.com/mayank-goyal09" target="_blank" class="social-link">💻 GitHub</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Awaiting Data Upload...")
