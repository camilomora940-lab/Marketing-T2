"""
Trabajo 2 – Segmentación de Mercado con K-Means y LCA
Grupo 29: Francisco Araneda, Benjamín Borquez, Martín Lagos, Camilo Mora, Isidora Salas
DII UdeC – Marketing 2026

Replica fiel del análisis realizado en Trabajo_2.ipynb
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Trabajo 2 – Segmentación de Mercado | Grupo 29",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS PREMIUM
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0d1117;
    --bg-card: #161b22;
    --bg-card2: #1c2128;
    --accent-1: #7c3aed;
    --accent-2: #2563eb;
    --accent-3: #059669;
    --accent-4: #d97706;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --border: #30363d;
    --grad1: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%);
    --grad2: linear-gradient(135deg, #059669 0%, #0891b2 100%);
    --grad3: linear-gradient(135deg, #d97706 0%, #dc2626 100%);
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text-primary); }

/* ── Hero ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(124,58,237,.18) 0%, rgba(37,99,235,.12) 50%, rgba(5,150,105,.08) 100%);
    border: 1px solid rgba(124,58,237,.30);
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(124,58,237,.25), transparent 70%);
    border-radius: 50%;
}
.hero-banner h1 {
    background: var(--grad1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0 0 10px;
    line-height: 1.2;
}
.hero-banner p { color: var(--text-secondary); font-size: 1.05rem; margin: 0; }
.hero-banner .team { color: var(--text-secondary); font-size: .88rem; margin-top: 12px; font-style: italic; }

/* ── Section title ── */
.section-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-primary);
    border-left: 4px solid #7c3aed;
    padding-left: 14px;
    margin: 28px 0 18px;
}

/* ── Glass cards ── */
.glass {
    background: rgba(22, 27, 34, .92);
    backdrop-filter: blur(14px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 18px;
}
.glass h4 { margin-top: 0; font-size: 1.05rem; font-weight: 600; }

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,.30);
    transition: transform .2s, box-shadow .2s;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 36px rgba(124,58,237,.18);
}
div[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-size: .82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .05em;
}
div[data-testid="stMetricValue"] { font-weight: 800 !important; font-size: 1.75rem !important; }

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: .92rem;
    padding: 10px 22px;
    border-radius: 8px 8px 0 0;
    letter-spacing: .02em;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-card);
    border-right: 1px solid var(--border);
}

/* ── Info boxes ── */
.info-box {
    background: rgba(37,99,235,.08);
    border: 1px solid rgba(37,99,235,.25);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: .93rem;
    color: var(--text-secondary);
}
.info-box strong { color: var(--text-primary); }

/* ── Cluster badge ── */
.cluster-badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: .8rem;
    letter-spacing: .04em;
    margin: 3px 5px;
}

/* ── Step indicator ── */
.step-indicator {
    display: flex;
    align-items: center;
    gap: 12px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 10px;
}
.step-num {
    background: var(--grad1);
    color: white;
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: .9rem;
    flex-shrink: 0;
}
.step-text { font-size: .95rem; color: var(--text-primary); }

/* ── Table style ── */
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAPPINGS (del notebook)
# ─────────────────────────────────────────────
GENDER_MAP = {'Male': 0, 'Female': 1, 'Other': 2}
GENDER_INV = {0: 'Male', 1: 'Female', 2: 'Other'}

ACQUISITION_MAP = {
    'Direct': 0, 'Social Media': 1, 'Organic Search': 2,
    'Email Campaign': 3, 'Paid Ad': 4, 'Referral': 5
}
ACQUISITION_INV = {v: k for k, v in ACQUISITION_MAP.items()}

MEMBERSHIP_MAP = {'Free': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}
MEMBERSHIP_INV = {0: 'Free', 1: 'Silver', 2: 'Gold', 3: 'Platinum'}

REGION_MAP = {
    'United States': 'North America', 'Germany': 'Europe', 'Italy': 'Europe',
    'Australia': 'Oceania', 'India': 'Asia', 'United Kingdom': 'Europe',
    'Spain': 'Europe', 'Japan': 'Asia', 'Turkey': 'Asia',
    'Canada': 'North America', 'Brazil': 'South America', 'France': 'Europe',
    'UAE': 'Middle East', 'South Africa': 'Africa', 'Netherlands': 'Europe',
    'Mexico': 'North America', 'Sweden': 'Europe', 'Singapore': 'Asia',
    'Poland': 'Europe', 'South Korea': 'Asia'
}
REGION_CODE = {
    'North America': 0, 'Europe': 1, 'Oceania': 2, 'Asia': 3,
    'South America': 4, 'Middle East': 5, 'Africa': 6
}

CATEGORY_MAP = {
    'Food & Grocery': 0, 'Toys & Games': 1, 'Home & Kitchen': 2, 'Electronics': 3,
    'Clothing & Apparel': 4, 'Office Supplies': 5, 'Sports & Outdoors': 6,
    'Beauty & Personal Care': 7, 'Books': 8, 'Health & Wellness': 9,
    'Travel & Luggage': 10, 'Jewelry & Accessories': 11, 'Pet Supplies': 12, 'Automotive': 13
}

PAYMENT_MAP = {
    'Credit Card': 0, 'Debit Card': 1, 'PayPal': 2,
    'UPI / Digital Wallet': 3, 'Bank Transfer': 4,
    'Cryptocurrency': 5, 'Buy Now Pay Later': 6
}
PAYMENT_INV = {v: k for k, v in PAYMENT_MAP.items()}

CLUSTER_COLORS_4 = ['#7c3aed', '#2563eb', '#059669', '#d97706']
CLUSTER_COLORS_5 = ['#64748b'] + CLUSTER_COLORS_4  # Gris para el grupo -1
CLUSTER_COLORS_LCA4 = ['#64748b', '#0ea5e9', '#f43f5e', '#10b981']  # Gris + 3 clusters de LCA
CLUSTER_COLORS_3 = ['#7c3aed', '#2563eb', '#059669']

# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df_orders = pd.read_csv('orders.csv')
    df_customers = pd.read_csv('customers_clustered.csv')
    return df_orders, df_customers


@st.cache_data
def prepare_data(_df_customers):
    """Aplica los mismos encodings del notebook."""
    df = _df_customers.copy()
    df['gender_raw'] = df['gender'].copy()
    df['membership_raw'] = df['membership_tier'].copy()
    df['acquisition_raw'] = df['acquisition_channel'].copy()
    df['country_raw'] = df['country'].copy()
    df['payment_raw'] = df['preferred_payment_method'].copy()

    df['gender'] = df['gender'].map(GENDER_MAP)
    df['acquisition_channel'] = df['acquisition_channel'].map(ACQUISITION_MAP)
    df['membership_tier'] = df['membership_tier'].map(MEMBERSHIP_MAP)
    df['preferred_payment_method'] = df['preferred_payment_method'].map(PAYMENT_MAP)
    df['region'] = df['country'].map(REGION_MAP)
    df['region_code'] = df['region'].map(REGION_CODE)
    df['preferred_category'] = df['preferred_category'].map(CATEGORY_MAP)

    df = df.rename(columns={
        'days_since_last_purchase': 'recency',
        'total_orders': 'frequency',
        'total_spend_usd': 'monetary'
    })
    return df


@st.cache_data
def build_enriched(_df_orders, _df_customers_raw):
    """
    Merge completo orders + customers con todas las variables originales.
    Devuelve el DataFrame enriquecido y la tabla de co-compra por categoría preferida.
    """
    # Merge: cada fila = una orden, con los atributos del cliente
    df_merged = _df_orders.merge(
        _df_customers_raw[[
            'customer_id', 'gender', 'age', 'country',
            'membership_tier', 'acquisition_channel',
            'preferred_category', 'days_since_last_purchase',
            'total_orders', 'total_spend_usd', 'churned',
            'newsletter_subscribed', 'returns_made', 'wishlist_items',
        ]],
        on='customer_id', how='left'
    )

    # Revenue por segmento x categoría de producto
    seg_cat = df_merged.groupby(['preferred_category', 'category']).agg(
        n_orders=('order_id', 'count'),
        revenue=('total_amount_usd', 'sum'),
        avg_ticket=('total_amount_usd', 'mean'),
        avg_discount=('discount_pct', 'mean'),
        pct_returned=('returned', 'mean'),
    ).reset_index()

    # Co-compra: clientes cuya categoría preferida es X → ¿qué otras categorías compran?
    copurchase = df_merged.groupby(['preferred_category', 'category']).agg(
        n_customers=('customer_id', 'nunique'),
        revenue=('total_amount_usd', 'sum'),
    ).reset_index()
    # Excluir cuando la categoría comprada == categoría preferida (compra esperada)
    copurchase_cross = copurchase[copurchase['preferred_category'] != copurchase['category']].copy()

    # Revenue mensual por categoría de producto
    df_merged['order_date'] = pd.to_datetime(_df_orders['order_date'])
    monthly_cat = df_merged.groupby(['year', 'month', 'category']).agg(
        revenue=('total_amount_usd', 'sum'),
        n_orders=('order_id', 'count'),
    ).reset_index()
    monthly_cat['period'] = pd.to_datetime(
        monthly_cat[['year', 'month']].assign(day=1)
    )

    return df_merged, seg_cat, copurchase_cross, monthly_cat


# ─────────────────────────────────────────────
# CARGAR DATOS
# ─────────────────────────────────────────────
df_orders_raw, df_customers_raw = load_data()
df = prepare_data(df_customers_raw)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
import os
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", use_container_width=True)

st.sidebar.markdown("""
<div style="text-align:center; padding: 14px 0 6px;">
    <h3 style="margin:0; font-weight:800;
       background: linear-gradient(135deg,#7c3aed,#2563eb);
       -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
       Trabajo 2
    </h3>
    <p style="color:#8b949e; font-size:.82rem; margin:4px 0 0;">Segmentación de Mercado</p>
    <p style="color:#6e7681; font-size:.75rem; margin:2px 0 0;">Grupo 29 – DII UdeC 2026</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("### ⚙️ Opciones de análisis")

show_raw_data = st.sidebar.checkbox("Mostrar datos crudos", value=False)
run_lca = st.sidebar.checkbox("Ejecutar modelos LCA (StepMix)", value=True,
                               help="Requiere que stepmix esté instalado. El cálculo puede tardar unos segundos.")

st.sidebar.markdown("---")
with st.sidebar.expander("📖 Acerca del análisis"):
    st.markdown("""
    **Metodologías:**
    - **K-Means** sobre variables RFM y sociodemográficas
    - **LCA** (Latent Class Analysis) con StepMix
    
    **Datasets:**
    - `customers.csv` – perfil de clientes
    - `orders.csv` – historial de órdenes
    
    **Número de clusters:**
    - RFM: **k = 4** (codo + silhouette)
    - Sociodemográfico: **k = 3** (BIC + entropía)
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Francisco Araneda · Benjamín Borquez · Martín Lagos · Camilo Mora · Isidora Salas")

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>📊 Segmentación de Mercado</h1>
    <p>Análisis comparativo de <strong>K-Means</strong> y <strong>Latent Class Analysis (LCA)</strong>
       sobre variables RFM y sociodemográficas de clientes de e-commerce.</p>
    <p class="team">Grupo 29: Francisco Araneda, Benjamín Borquez, Martín Lagos, Camilo Mora, Isidora Salas</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPIs GENERALES
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("👥 Clientes", f"{len(df_customers_raw):,}")
k2.metric("📦 Órdenes", f"{len(df_orders_raw):,}")
k3.metric("💰 Gasto Promedio", f"${df_customers_raw['total_spend_usd'].mean():,.0f}")
k4.metric("🔁 Frecuencia Media", f"{df_customers_raw['total_orders'].mean():.1f}")
k5.metric("📅 Recency Media", f"{df_customers_raw['days_since_last_purchase'].mean():.0f}d")
k6.metric("🌍 Países", f"{df_customers_raw['country'].nunique()}")

st.markdown("---")

# ─────────────────────────────────────────────
# TABS PRINCIPALES
# ─────────────────────────────────────────────
tab_datos, tab_rfm_kmeans, tab_socio_kmeans, tab_lca_rfm, tab_lca_socio, tab_comp_rfm, tab_comp_socio, tab_perfiles, tab_cruce, tab_estrategia = st.tabs([
    "1. Datos",
    "2. K-Means RFM",
    "3. K-Means Socio",
    "4. LCA RFM",
    "5. LCA Socio",
    "6. Comp. RFM",
    "7. Comp. Socio",
    "8. Perfiles",
    "9. Cruce Segmentos",
    "10. Estrategia",
])

# ════════════════════════════════════════════
# TAB 1: DATOS
# ════════════════════════════════════════════
with tab_datos:
    st.markdown('<div class="section-title">Exploración de Datos</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🧑‍💼 Dataset de Clientes (`customers.csv`)")
        st.markdown(f"""
        <div class="info-box">
        <strong>Dimensiones:</strong> {df_customers_raw.shape[0]:,} filas × {df_customers_raw.shape[1]} columnas<br>
        <strong>Nulos:</strong> {df_customers_raw.isnull().sum().sum()} valores faltantes<br>
        <strong>Duplicados:</strong> {df_customers_raw.duplicated().sum()} filas duplicadas
        </div>
        """, unsafe_allow_html=True)

        # Nulls por columna
        nulls_c = df_customers_raw.isnull().sum()
        nulls_c = nulls_c[nulls_c > 0]
        if len(nulls_c) > 0:
            st.markdown("**Columnas con nulos:**")
            st.dataframe(nulls_c.to_frame("Nulos"), use_container_width=True)
        else:
            st.success("✅ Sin valores nulos en customers.csv")

        if show_raw_data:
            st.dataframe(df_customers_raw.head(20), use_container_width=True)

    with col_b:
        st.markdown("#### 📦 Dataset de Órdenes (`orders.csv`)")
        st.markdown(f"""
        <div class="info-box">
        <strong>Dimensiones:</strong> {df_orders_raw.shape[0]:,} filas × {df_orders_raw.shape[1]} columnas<br>
        <strong>Nulos:</strong> {df_orders_raw.isnull().sum().sum()} valores faltantes<br>
        <strong>Duplicados:</strong> {df_orders_raw.duplicated().sum()} filas duplicadas
        </div>
        """, unsafe_allow_html=True)

        # Nulls por columna
        nulls_o = df_orders_raw.isnull().sum()
        nulls_o = nulls_o[nulls_o > 0]
        if len(nulls_o) > 0:
            st.markdown("**Columnas con nulos:**")
            st.dataframe(nulls_o.to_frame("Nulos"), use_container_width=True)
        else:
            st.success("✅ Sin valores nulos en orders.csv")

        if show_raw_data:
            st.dataframe(df_orders_raw.head(20), use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Estadísticas Descriptivas – Variables RFM</div>', unsafe_allow_html=True)

    rfm_stats = df_customers_raw[['days_since_last_purchase', 'total_orders', 'total_spend_usd']].rename(columns={
        'days_since_last_purchase': 'Recency (días)',
        'total_orders': 'Frequency (órdenes)',
        'total_spend_usd': 'Monetary (USD)'
    })
    st.dataframe(rfm_stats.describe().round(2).style.format("{:.2f}"), use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Distribuciones de Variables RFM</div>', unsafe_allow_html=True)

    fig_rfm_hist = make_subplots(rows=1, cols=3, subplot_titles=["Recency (días)", "Frequency (órdenes)", "Monetary (USD)"])
    rfm_cols = ['days_since_last_purchase', 'total_orders', 'total_spend_usd']
    rfm_names = ['Recency', 'Frequency', 'Monetary']
    rfm_colors = ['#7c3aed', '#2563eb', '#059669']

    for i, (col, name, color) in enumerate(zip(rfm_cols, rfm_names, rfm_colors)):
        fig_rfm_hist.add_trace(
            go.Histogram(x=df_customers_raw[col], name=name, marker_color=color, opacity=0.75, nbinsx=40),
            row=1, col=i+1
        )

    fig_rfm_hist.update_layout(
        template='plotly_dark', height=380,
        font=dict(family='Inter'), showlegend=False,
        title_text='<b>Distribución de Variables RFM</b>',
    )
    st.plotly_chart(fig_rfm_hist, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Encodings Aplicados (del Notebook)</div>', unsafe_allow_html=True)

    enc_col1, enc_col2, enc_col3 = st.columns(3)
    with enc_col1:
        st.markdown("**Género**")
        st.dataframe(pd.DataFrame.from_dict(GENDER_MAP, orient='index', columns=['Código']), use_container_width=True)
        st.markdown("**Membresía**")
        st.dataframe(pd.DataFrame.from_dict(MEMBERSHIP_MAP, orient='index', columns=['Código']), use_container_width=True)
    with enc_col2:
        st.markdown("**Canal de Adquisición**")
        st.dataframe(pd.DataFrame.from_dict(ACQUISITION_MAP, orient='index', columns=['Código']), use_container_width=True)
    with enc_col3:
        st.markdown("**Región**")
        st.dataframe(pd.DataFrame.from_dict(REGION_CODE, orient='index', columns=['Código']), use_container_width=True)


# ════════════════════════════════════════════
# TAB 2: K-MEANS RFM
# ════════════════════════════════════════════
with tab_rfm_kmeans:
    st.markdown('<div class="section-title">K-Means sobre Variables RFM</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>Variables:</strong> Recency, Frequency, Monetary<br>
    <strong>Preprocesamiento:</strong> Estandarización con StandardScaler<br>
    <strong>Selección de K:</strong> Método del Codo + Silhouette Score → <strong>k = 4</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    import json
    import plotly.express as px
    with open('metrics.json', 'r') as f:
        metrics = json.load(f)

    st.markdown("#### 🔍 Selección del K óptimo")
    col1, col2 = st.columns(2)
    with col1:
        fig_elbow = px.line(x=metrics['km_rfm']['K'], y=metrics['km_rfm']['inertia'], markers=True,
                            title='<b>Elbow Method (Inertia)</b>',
                            labels={'x':'Número de Clusters (k)', 'y':'Inertia'})
        fig_elbow.update_layout(template='plotly_dark', height=350)
        fig_elbow.add_vline(x=4, line_dash='dash', line_color='#ff4b4b', annotation_text='K=4 Seleccionado', annotation_position='top right')
        st.plotly_chart(fig_elbow, use_container_width=True)
    with col2:
        fig_silh = px.line(x=metrics['km_rfm']['K'], y=metrics['km_rfm']['silhouette'], markers=True,
                           title='<b>Silhouette Score</b>',
                           labels={'x':'Número de Clusters (k)', 'y':'Silhouette Score'})
        fig_silh.update_layout(template='plotly_dark', height=350)
        fig_silh.add_vline(x=4, line_dash='dash', line_color='#ff4b4b', annotation_text='K=4 Seleccionado', annotation_position='bottom right')
        st.plotly_chart(fig_silh, use_container_width=True)
    st.markdown("---")

    st.markdown("#### 📊 Resultado del Clustering (k = 4)")

    # Distribución de clusters
    cluster_counts_rfm = df['cluster_rfmkmeans'].value_counts().sort_index()
    col_dist1, col_dist2 = st.columns(2)

    with col_dist1:
        fig_pie_rfm = px.pie(
            values=cluster_counts_rfm.values,
            names=[f"Cluster {i}" for i in cluster_counts_rfm.index],
            title='<b>Distribución de clientes por cluster RFM</b>',
            color_discrete_sequence=CLUSTER_COLORS_4,
            hole=0.42,
        )
        fig_pie_rfm.update_traces(textposition='inside', textinfo='percent+label', textfont_size=13)
        fig_pie_rfm.update_layout(template='plotly_dark', height=380, font=dict(family='Inter'),
                                   legend=dict(orientation='h', y=-0.1))
        st.plotly_chart(fig_pie_rfm, use_container_width=True)

    with col_dist2:
        st.markdown("**Clientes por cluster:**")
        for cluster, count in cluster_counts_rfm.items():
            pct = count / len(df) * 100
            color = CLUSTER_COLORS_4[cluster]
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:12px; margin:8px 0;
                        background:rgba(22,27,34,.8); border:1px solid #30363d;
                        border-radius:10px; padding:12px 16px;">
                <span class="cluster-badge" style="background:{color}20; color:{color}; border:1px solid {color}60;">
                    Cluster {cluster}
                </span>
                <div style="flex:1;">
                    <div style="background:#30363d; border-radius:6px; height:8px; overflow:hidden;">
                        <div style="width:{pct:.1f}%; height:100%; background:{color}; border-radius:6px;"></div>
                    </div>
                </div>
                <span style="color:#e6edf3; font-weight:700; min-width:60px;">{count:,} ({pct:.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)




# ════════════════════════════════════════════
# TAB 3: K-MEANS SOCIODEMOGRÁFICO
# ════════════════════════════════════════════
with tab_socio_kmeans:
    st.markdown('<div class="section-title">K-Means sobre Variables Sociodemográficas</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>Variables:</strong> age, gender, acquisition_channel, membership_tier<br>
    <strong>Preprocesamiento:</strong> Estandarización con StandardScaler<br>
    <strong>Variables:</strong> age, gender, acquisition_channel, membership_tier, preferred_payment_method<br>
    <strong>Preprocesamiento:</strong> Estandarización con StandardScaler<br>
    <strong>Selección de K:</strong> Método del Codo + Silhouette Score → <strong>k = 4</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    with open('metrics.json', 'r') as f:
        metrics = json.load(f)

    st.markdown("#### 🔍 Selección del K óptimo")
    col_sk1, col_sk2 = st.columns(2)
    with col_sk1:
        fig_elbow_s = px.line(x=metrics['km_socio']['K'], y=metrics['km_socio']['inertia'], markers=True,
                            title='<b>Elbow Method (Inertia)</b>',
                            labels={'x':'Número de Clusters (k)', 'y':'Inertia'})
        fig_elbow_s.update_layout(template='plotly_dark', height=350)
        fig_elbow_s.add_vline(x=4, line_dash='dash', line_color='#ff4b4b', annotation_text='K=4 Seleccionado', annotation_position='top right')
        st.plotly_chart(fig_elbow_s, use_container_width=True)
    with col_sk2:
        fig_silh_s = px.line(x=metrics['km_socio']['K'], y=metrics['km_socio']['silhouette'], markers=True,
                           title='<b>Silhouette Score</b>',
                           labels={'x':'Número de Clusters (k)', 'y':'Silhouette Score'})
        fig_silh_s.update_layout(template='plotly_dark', height=350)
        fig_silh_s.add_vline(x=4, line_dash='dash', line_color='#ff4b4b', annotation_text='K=4 Seleccionado', annotation_position='bottom right')
        st.plotly_chart(fig_silh_s, use_container_width=True)
    st.markdown("---")

    st.markdown("#### 📊 Resultado del Clustering Sociodemográfico (k = 4)")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        cluster_counts_socio = df['cluster_sociodemografico'].value_counts().sort_index()
        fig_pie_socio = px.pie(
            values=cluster_counts_socio.values,
            names=[f"Cluster {i}" if i != -1 else "Grupo 18" for i in cluster_counts_socio.index],
            title='<b>Distribución por cluster sociodemográfico</b>',
            color_discrete_sequence=CLUSTER_COLORS_5, hole=0.42,
        )
        fig_pie_socio.update_traces(textposition='inside', textinfo='percent+label', textfont_size=13)
        fig_pie_socio.update_layout(template='plotly_dark', height=380, font=dict(family='Inter'))
        st.plotly_chart(fig_pie_socio, use_container_width=True)
    with col_s2:
        st.markdown("**Clientes por cluster:**")
        for cluster_raw, count in cluster_counts_socio.items():
            pct = count / len(df) * 100
            cluster=int(cluster_raw)
            if cluster == -1:
                color = '#d97706'  # Naranja
                label_text = "Grupo 18"
            elif cluster == 0:
                color = '#64748b'  # Gris
                label_text = "Cluster 0"
            elif cluster == 1:
                color = '#7c3aed'  # Morado
                label_text = "Cluster 1"
            elif cluster == 2:
                color = '#059669'  # Verde
                label_text = "Cluster 2"
            elif cluster == 3:
                color = '#2563eb'  # Azul
                label_text = "Cluster 3"
            else:
                color = '#64748b'
                label_text = f"Cluster {cluster}"
                
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:12px; margin:8px 0;
                        background:rgba(22,27,34,.8); border:1px solid #30363d;
                        border-radius:10px; padding:12px 16px;">
                <span class="cluster-badge" style="background:{color}20; color:{color}; border:1px solid {color}60;">
                    {label_text}
                </span>
                <div style="flex:1;">
                    <div style="background:#30363d; border-radius:6px; height:8px; overflow:hidden;">
                        <div style="width:{pct:.1f}%; height:100%; background:{color}; border-radius:6px;"></div>
                    </div>
                </div>
                <span style="color:#e6edf3; font-weight:700; min-width:60px;">{count:,} ({pct:.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)    





# ════════════════════════════════════════════
# TAB 4: LCA RFM
# ════════════════════════════════════════════
with tab_lca_rfm:
    st.markdown('<div class="section-title">Latent Class Analysis (LCA) – Variables RFM</div>', unsafe_allow_html=True)

    if not run_lca:
        st.info("ℹ️ El análisis LCA está desactivado. Actívalo desde el sidebar.")
    else:
        st.markdown("""
        <div class="info-box">
        <strong>Variables:</strong> Recency, Frequency, Monetary (tratadas como gaussianas)<br>
        <strong>Herramienta:</strong> StepMix (Mixture Model)<br>
        <strong>Selección de K:</strong> BIC + Entropía Relativa → <strong>k = 4</strong>
        </div>
        """, unsafe_allow_html=True)

        if True:
            st.markdown("---")

        with open('metrics.json', 'r') as f:
            metrics = json.load(f)

        st.markdown("#### 🔍 Selección del K óptimo – BIC y Entropía")
        col_lca_bic1, col_lca_bic2 = st.columns(2)
        with col_lca_bic1:
            fig_bic = px.line(x=metrics['lca_rfm']['K'], y=metrics['lca_rfm']['bic'], markers=True,
                                title='<b>BIC vs. número de clases</b>',
                                labels={'x':'Clases latentes (k)', 'y':'BIC'})
            fig_bic.update_layout(template='plotly_dark', height=350)
            fig_bic.add_vline(x=4, line_dash='dash', line_color='#ff4b4b', annotation_text='K=4 Seleccionado', annotation_position='top right')
            st.plotly_chart(fig_bic, use_container_width=True)
        with col_lca_bic2:
            fig_ent = px.line(x=metrics['lca_rfm']['K'], y=metrics['lca_rfm']['entropy'], markers=True,
                               title='<b>Entropía relativa vs. número de clases</b>',
                               labels={'x':'Clases latentes (k)', 'y':'Entropía Relativa'})
            fig_ent.update_layout(template='plotly_dark', height=350)
            fig_ent.update_yaxes(range=[0, 1.05])
            fig_ent.add_vline(x=4, line_dash='dash', line_color='#ff4b4b', annotation_text='K=4 Seleccionado', annotation_position='bottom right')
            st.plotly_chart(fig_ent, use_container_width=True)
        st.markdown("---")
        if True:
            st.markdown("#### 📊 Resultado del Clustering LCA (k = 4)")

            col_lca1, col_lca2 = st.columns(2)

            with col_lca1:
                cluster_counts_lca_rfm = df['lca_cluster_rfm'].value_counts().sort_index()
                fig_pie_lca_rfm = px.pie(
                    values=cluster_counts_lca_rfm.values,
                    names=[f"Clase {i}" for i in cluster_counts_lca_rfm.index],
                    title='<b>Distribución de clientes por clase LCA RFM</b>',
                    color_discrete_sequence=CLUSTER_COLORS_4, hole=0.42,
                )
                fig_pie_lca_rfm.update_traces(textposition='inside', textinfo='percent+label', textfont_size=13)
                fig_pie_lca_rfm.update_layout(template='plotly_dark', height=380, font=dict(family='Inter'))
                st.plotly_chart(fig_pie_lca_rfm, use_container_width=True)
            with col_lca2:
                st.markdown("**Clientes por cluster:**")
                for cluster_raw, count in cluster_counts_lca_rfm.items():
                    pct = count / len(df) * 100
                    cluster=int(cluster_raw)
                    if cluster == 0:
                        color = '#7c3aed'  # Morado
                        label_text = "Cluster 0"
                    elif cluster == 1:
                        color = '#059669'  # Verde
                        label_text = "Cluster 1"
                    elif cluster == 2:
                        color = '#d97706'  # Naranja
                        label_text = "Cluster 2"
                    elif cluster == 3:
                        color = '#2563eb'  # Azul
                        label_text = "Cluster 3"
                    else:
                        color = '#d97706'
                        label_text = f"Cluster {cluster}"
                    
                    st.markdown(f"""<div style="display:flex; align-items:center; gap:12px; margin:8px 0;
                        background:rgba(22,27,34,.8); border:1px solid #30363d;
                        border-radius:10px; padding:12px 16px;">
                <span class="cluster-badge" style="background:{color}20; color:{color}; border:1px solid {color}60;">
                    {label_text}
                </span>
                <div style="flex:1;">
                    <div style="background:#30363d; border-radius:6px; height:8px; overflow:hidden;">
                        <div style="width:{pct:.1f}%; height:100%; background:{color}; border-radius:6px;"></div>
                    </div>
                </div>
                <span style="color:#e6edf3; font-weight:700; min-width:60px;">{count:,} ({pct:.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)    
                




# ════════════════════════════════════════════
# TAB 5: LCA SOCIODEMOGRÁFICO
# ════════════════════════════════════════════
with tab_lca_socio:
    st.markdown('<div class="section-title">Latent Class Analysis (LCA) – Variables Sociodemográficas</div>', unsafe_allow_html=True)

    if not run_lca:
        st.info("ℹ️ El análisis LCA está desactivado. Actívalo desde el sidebar.")
    else:
        st.markdown("""
        <div class="info-box">
        <strong>Variables:</strong> gender (cat.), acquisition_channel (cat.), membership_tier (cat.), preferred_payment_method (cat.), age (gaussiana)<br>
        <strong>Herramienta:</strong> StepMix (modelo mixto)<br>
        <strong>Selección de K:</strong> BIC + Entropía Relativa → <strong>k = 3</strong>
        </div>
        """, unsafe_allow_html=True)

        if True:
            st.markdown("---")

        with open('metrics.json', 'r') as f:
            metrics = json.load(f)

        st.markdown("#### 🔍 Selección del K óptimo – BIC y Entropía")
        col_s_lca_bic1, col_s_lca_bic2 = st.columns(2)
        with col_s_lca_bic1:
            fig_bic_s = px.line(x=metrics['lca_socio']['K'], y=metrics['lca_socio']['bic'], markers=True,
                                title='<b>BIC vs. número de clases</b>',
                                labels={'x':'Clases latentes (k)', 'y':'BIC'})
            fig_bic_s.update_layout(template='plotly_dark', height=350)
            fig_bic_s.add_vline(x=3, line_dash='dash', line_color='#ff4b4b', annotation_text='K=3 Seleccionado', annotation_position='top right')
            st.plotly_chart(fig_bic_s, use_container_width=True)
        with col_s_lca_bic2:
            fig_ent_s = px.line(x=metrics['lca_socio']['K'], y=metrics['lca_socio']['entropy'], markers=True,
                               title='<b>Entropía relativa vs. número de clases</b>',
                               labels={'x':'Clases latentes (k)', 'y':'Entropía Relativa'})
            fig_ent_s.update_layout(template='plotly_dark', height=350)
            fig_ent_s.update_yaxes(range=[0, 1.05])
            fig_ent_s.add_vline(x=3, line_dash='dash', line_color='#ff4b4b', annotation_text='K=3 Seleccionado', annotation_position='bottom right')
            st.plotly_chart(fig_ent_s, use_container_width=True)
        st.markdown("---")
        if True:
            st.markdown("#### 📊 Resultado del Clustering LCA Socio (k = 3)")

            col_s_lca1, col_s_lca2 = st.columns(2)

            with col_s_lca1:
                # Pie chart
                counts_lca_socio = df['cluster_sociodemografico_lca'].value_counts().sort_index()
                fig_pie_lca_socio = px.pie(
                    values=counts_lca_socio.values,
                    names=[f"Clase {i}" if i != -1 else "Grupo 18" for i in counts_lca_socio.index],
                    title='<b>Distribución por clase LCA Sociodemográfico</b>',
                    color_discrete_sequence=CLUSTER_COLORS_LCA4, hole=0.42,
                )
                fig_pie_lca_socio.update_traces(textposition='inside', textinfo='percent+label', textfont_size=13)
                fig_pie_lca_socio.update_layout(template='plotly_dark', height=380, font=dict(family='Inter'))
                st.plotly_chart(fig_pie_lca_socio, use_container_width=True)

            with col_s_lca2:
                # Scatter Age vs Monetary coloreado por LCA socio
                df_lca_s_plot = df.copy()
                df_lca_s_plot['Clase'] = df_lca_s_plot['cluster_sociodemografico_lca'].astype(str).map(
                    lambda x: "Grupo 18" if x == "-1" else f"Clase {x}"
                )
                df_lca_s_plot = df_lca_s_plot.sort_values('cluster_sociodemografico_lca')
                fig_lca_age_mon = px.scatter(
                    df_lca_s_plot, x='age', y='monetary',
                    color='Clase', color_discrete_sequence=CLUSTER_COLORS_LCA4,
                    title='<b>LCA Sociodemográfico – Age vs Monetary</b>',
                    labels={'age': 'Edad', 'monetary': 'Gasto Total (USD)'},
                    opacity=0.55, height=380,
                )
                fig_lca_age_mon.update_layout(template='plotly_dark', font=dict(family='Inter'))
                st.plotly_chart(fig_lca_age_mon, use_container_width=True)

            # Crosstabs de distribuciones
            st.markdown("#### 📊 Distribuciones Porcentuales por Clase")

            tab_cross1, tab_cross2 = st.tabs(["Membresía por Clase", "Canal de Adquisición por Clase"])

            with tab_cross1:
                tabla_pct_memb = pd.crosstab(df['cluster_sociodemografico_lca'],
                                              df['membership_tier'],
                                              normalize='index') * 100
                tabla_pct_memb.index = [f"Clase {i}" if i != -1 else "Grupo 18" for i in tabla_pct_memb.index]
                tabla_pct_memb.columns = [MEMBERSHIP_INV.get(c, str(c)) for c in tabla_pct_memb.columns]
                st.dataframe(tabla_pct_memb.style.format("{:.1f}%")
                             .background_gradient(cmap='Blues'), use_container_width=True)

                # Stacked bar
                tabla_pct_memb.index.name = 'Clase'
                tabla_pct_memb_reset = tabla_pct_memb.reset_index().melt(id_vars='Clase',
                                                                          var_name='Membresía', value_name='%')
                fig_memb_bar = px.bar(tabla_pct_memb_reset, x='Clase', y='%',
                                      color='Membresía', barmode='stack',
                                      title='<b>Distribución de Membresía por Clase LCA</b>',
                                      color_discrete_sequence=px.colors.qualitative.Set2)
                fig_memb_bar.update_layout(template='plotly_dark', height=360, font=dict(family='Inter'))
                st.plotly_chart(fig_memb_bar, use_container_width=True)

            with tab_cross2:
                tabla_pct_acq = pd.crosstab(df['cluster_sociodemografico_lca'],
                                             df['acquisition_channel'],
                                             normalize='index') * 100
                tabla_pct_acq.index = [f"Clase {i}" if i != -1 else "Grupo 18" for i in tabla_pct_acq.index]
                tabla_pct_acq.columns = [ACQUISITION_INV.get(c, str(c)) for c in tabla_pct_acq.columns]
                st.dataframe(tabla_pct_acq.style.format("{:.1f}%")
                             .background_gradient(cmap='Purples'), use_container_width=True)

                tabla_pct_acq.index.name = 'Clase'
                tabla_pct_acq_reset = tabla_pct_acq.reset_index().melt(id_vars='Clase',
                                                                         var_name='Canal', value_name='%')
                fig_acq_bar = px.bar(tabla_pct_acq_reset, x='Clase', y='%',
                                     color='Canal', barmode='stack',
                                     title='<b>Distribución de Canal de Adquisición por Clase LCA</b>',
                                     color_discrete_sequence=px.colors.qualitative.Set3)
                fig_acq_bar.update_layout(template='plotly_dark', height=360, font=dict(family='Inter'))
                st.plotly_chart(fig_acq_bar, use_container_width=True)



# ════════════════════════════════════════════
# TAB 6: COMPARACIÓN RFM
# ════════════════════════════════════════════
with tab_comp_rfm:
    st.markdown('<div class="section-title">Comparación de Modelos: K-Means vs LCA (RFM)</div>', unsafe_allow_html=True)
    
    if 'lca_cluster_rfm' not in df.columns:
        st.warning("⚠️ Debes activar y calcular el modelo LCA en el sidebar para ver las comparaciones.")
    else:
        st.markdown("### 📊 Gráficos de Dispersión 2D (Recency vs Monetary)")
        col_2d_r1, col_2d_r2 = st.columns(2)
        
        with col_2d_r1:
            df_r_plot1 = df.copy()
            df_r_plot1['Cluster'] = df_r_plot1['cluster_rfmkmeans'].astype(str).map(lambda x: f"KM-{x}")
            import plotly.express as px
            fig_2d_km_rfm = px.scatter(
                df_r_plot1, x='recency', y='monetary',
                color='Cluster', color_discrete_sequence=CLUSTER_COLORS_4,
                title='<b>K-Means RFM (k=4)</b>',
                labels={'recency': 'Recency', 'monetary': 'Monetary'},
                opacity=0.55, height=380,
            )
            fig_2d_km_rfm.update_layout(template='plotly_dark', font=dict(family='Inter'))
            st.plotly_chart(fig_2d_km_rfm, use_container_width=True)
            
        with col_2d_r2:
            df_r_plot2 = df.copy()
            df_r_plot2['Clase'] = df_r_plot2['lca_cluster_rfm'].astype(str).map(lambda x: f"LCA-{x}")
            fig_2d_lca_rfm = px.scatter(
                df_r_plot2, x='recency', y='monetary',
                color='Clase', color_discrete_sequence=CLUSTER_COLORS_4,
                title='<b>LCA RFM (k=4)</b>',
                labels={'recency': 'Recency', 'monetary': 'Monetary'},
                opacity=0.55, height=380,
            )
            fig_2d_lca_rfm.update_layout(template='plotly_dark', font=dict(family='Inter'))
            st.plotly_chart(fig_2d_lca_rfm, use_container_width=True)
            
        st.markdown("---")
        st.markdown("### 🌐 Gráficos de Dispersión 3D")
        col_3d_r1, col_3d_r2 = st.columns(2)
        
        with col_3d_r1:
            fig_3d_rfm = px.scatter_3d(
                df_r_plot1, x='recency', y='frequency', z='monetary',
                color='Cluster', color_discrete_sequence=CLUSTER_COLORS_4,
                title='<b>K-Means RFM (k=4)</b>',
                labels={'recency': 'Recency', 'frequency': 'Frequency', 'monetary': 'Monetary'},
                opacity=0.65, height=450,
            )
            fig_3d_rfm.update_traces(marker=dict(size=3.5))
            fig_3d_rfm.update_layout(template='plotly_dark', font=dict(family='Inter'), margin=dict(l=0, r=0, b=0, t=50))
            st.plotly_chart(fig_3d_rfm, use_container_width=True)
            
        with col_3d_r2:
            fig_3d_lca_rfm = px.scatter_3d(
                df_r_plot2, x='recency', y='frequency', z='monetary',
                color='Clase', color_discrete_sequence=CLUSTER_COLORS_4,
                title='<b>LCA RFM (k=4)</b>',
                labels={'recency': 'Recency', 'frequency': 'Frequency', 'monetary': 'Monetary'},
                opacity=0.65, height=450,
            )
            fig_3d_lca_rfm.update_traces(marker=dict(size=3.5))
            fig_3d_lca_rfm.update_layout(template='plotly_dark', font=dict(family='Inter'), margin=dict(l=0, r=0, b=0, t=50))
            st.plotly_chart(fig_3d_lca_rfm, use_container_width=True)


# ════════════════════════════════════════════
# TAB 7: COMPARACIÓN SOCIODEMOGRÁFICA
# ════════════════════════════════════════════
with tab_comp_socio:
    st.markdown('<div class="section-title">Comparación de Modelos: K-Means vs LCA (Sociodemográfico)</div>', unsafe_allow_html=True)
    
    if 'cluster_sociodemografico_lca' not in df.columns:
        st.warning("⚠️ Debes activar y calcular el modelo LCA en el sidebar para ver las comparaciones.")
    else:
        st.markdown("### 📊 Gráficos de Dispersión 2D (Age vs Monetary)")
        col_2d_s1, col_2d_s2 = st.columns(2)
        
        with col_2d_s1:
            df_s_plot1 = df.copy()
            df_s_plot1['Cluster'] = df_s_plot1['cluster_sociodemografico'].astype(str).map(
                lambda x: "Grupo 18" if x == "-1" else f"KM-{x}"
            )
            df_s_plot1 = df_s_plot1.sort_values('cluster_sociodemografico')
            fig_2d_km_socio = px.scatter(
                df_s_plot1, x='age', y='monetary',
                color='Cluster', color_discrete_sequence=CLUSTER_COLORS_5,
                title='<b>K-Means Sociodemográfico (k=4)</b>',
                labels={'age': 'Edad', 'monetary': 'Monetary'},
                opacity=0.55, height=380,
            )
            fig_2d_km_socio.update_layout(template='plotly_dark', font=dict(family='Inter'))
            st.plotly_chart(fig_2d_km_socio, use_container_width=True)
            
        with col_2d_s2:
            df_s_plot2 = df.copy()
            df_s_plot2['Clase'] = df_s_plot2['cluster_sociodemografico_lca'].astype(str).map(
                lambda x: "Grupo 18" if x == "-1" else f"LCA-{x}"
            )
            df_s_plot2 = df_s_plot2.sort_values('cluster_sociodemografico_lca')
            fig_2d_lca_socio = px.scatter(
                df_s_plot2, x='age', y='monetary',
                color='Clase', color_discrete_sequence=CLUSTER_COLORS_LCA4,
                title='<b>LCA Sociodemográfico (k=3)</b>',
                labels={'age': 'Edad', 'monetary': 'Monetary'},
                opacity=0.55, height=380,
            )
            fig_2d_lca_socio.update_layout(template='plotly_dark', font=dict(family='Inter'))
            st.plotly_chart(fig_2d_lca_socio, use_container_width=True)
            
        st.markdown("---")
        st.markdown("### 🌐 Gráficos de Dispersión 3D (Age vs Frequency vs Monetary)")
        col_3d_s1, col_3d_s2 = st.columns(2)
        
        with col_3d_s1:
            fig_3d_km_socio = px.scatter_3d(
                df_s_plot1, x='age', y='frequency', z='monetary',
                color='Cluster', color_discrete_sequence=CLUSTER_COLORS_5,
                title='<b>K-Means Sociodemográfico (k=4)</b>',
                labels={'age': 'Edad', 'frequency': 'Frecuencia', 'monetary': 'Gasto (USD)'},
                opacity=0.65, height=450,
            )
            fig_3d_km_socio.update_traces(marker=dict(size=3.5))
            fig_3d_km_socio.update_layout(template='plotly_dark', font=dict(family='Inter'), margin=dict(l=0, r=0, b=0, t=50))
            st.plotly_chart(fig_3d_km_socio, use_container_width=True)
            
        with col_3d_s2:
            fig_3d_lca_socio_comp = px.scatter_3d(
                df_s_plot2, x='age', y='frequency', z='monetary',
                color='Clase', color_discrete_sequence=CLUSTER_COLORS_LCA4,
                title='<b>LCA Sociodemográfico (k=3)</b>',
                labels={'age': 'Edad', 'frequency': 'Frecuencia', 'monetary': 'Gasto (USD)'},
                opacity=0.65, height=450,
            )
            fig_3d_lca_socio_comp.update_traces(marker=dict(size=3.5))
            fig_3d_lca_socio_comp.update_layout(template='plotly_dark', font=dict(family='Inter'), margin=dict(l=0, r=0, b=0, t=50))
            st.plotly_chart(fig_3d_lca_socio_comp, use_container_width=True)



# ════════════════════════════════════════════
# TAB 8: PERFILES DE CLUSTERS
# ════════════════════════════════════════════
with tab_perfiles:
    st.markdown('<div class="section-title">Perfiles Comparativos de Clusters</div>', unsafe_allow_html=True)

    # Verificar si hay columnas de LCA disponibles
    has_lca_rfm = 'lca_cluster_rfm' in df.columns
    has_lca_socio = 'cluster_sociodemografico_lca' in df.columns

    # ── Perfil K-Means RFM ──
    st.markdown("#### 🔵 Perfil K-Means RFM (k=4)")
    perfil_km_rfm = df.groupby('cluster_rfmkmeans')[['recency', 'frequency', 'monetary',
                                                       'avg_review_score', 'returns_made',
                                                       'wishlist_items', 'newsletter_subscribed',
                                                       'churned']].mean().round(2)
    perfil_km_rfm['n_clientes'] = df.groupby('cluster_rfmkmeans').size()
    perfil_km_rfm['% del total'] = (perfil_km_rfm['n_clientes'] / len(df) * 100).round(1)
    perfil_km_rfm.index = [f"Cluster {i}" for i in perfil_km_rfm.index]
    perfil_km_rfm.columns = ['Recency', 'Frequency', 'Monetary', 'Rating Medio',
                               'Devoluciones', 'Wishlist', 'Newsletter', 'Churn', 'Clientes', '%']
    st.dataframe(
        perfil_km_rfm.style.format({
            'Monetary': '${:.2f}',
            'Recency': '{:.1f}d',
            'Frequency': '{:.1f}',
            '%': '{:.1f}%',
            'Churn': '{:.2f}',
            'Newsletter': '{:.2f}',
        })
        .background_gradient(subset=['Monetary'], cmap='Greens')
        .background_gradient(subset=['Churn'], cmap='Reds')
        .background_gradient(subset=['Frequency'], cmap='Blues'),
        use_container_width=True
    )
    if has_lca_rfm:
        st.markdown("---")
        st.markdown("#### 🔶 Perfil LCA RFM (k=4)")
        perfil_lca = df.groupby('lca_cluster_rfm')[['recency', 'frequency', 'monetary',
                                                       'avg_review_score', 'returns_made',
                                                       'wishlist_items', 'newsletter_subscribed',
                                                       'churned']].mean().round(2)
        perfil_lca['n_clientes'] = df.groupby('lca_cluster_rfm').size()
        perfil_lca['% del total'] = (perfil_lca['n_clientes'] / len(df) * 100).round(1)
        perfil_lca.index = [f"Clase {i}" for i in perfil_lca.index]
        perfil_lca.columns = ['Recency', 'Frequency', 'Monetary', 'Rating Medio',
                               'Devoluciones', 'Wishlist', 'Newsletter', 'Churn', 'Clientes', '%']
        st.dataframe(
            perfil_lca.style.format({
                'Monetary': '${:.2f}',
                'Recency': '{:.1f}d',
                'Frequency': '{:.1f}',
                '%': '{:.1f}%',
                'Churn': '{:.2f}',
            })
            .background_gradient(subset=['Monetary'], cmap='Oranges')
            .background_gradient(subset=['Churn'], cmap='Reds')
            .background_gradient(subset=['Frequency'], cmap='Blues'),
            use_container_width=True
        )

    # ── Perfil K-Means Sociodemográfico ──
    if 'cluster_sociodemografico' in df.columns:
        st.markdown("---")
        st.markdown("#### 🟣 Perfil K-Means Sociodemográfico (Mean ± Std & Moda)")
        vars_socio = ['age', 'gender', 'acquisition_channel', 'membership_tier', 'preferred_payment_method']
        
        km_socio_mean = df.groupby('cluster_sociodemografico')[vars_socio].mean().round(2)
        km_socio_std = df.groupby('cluster_sociodemografico')[vars_socio].std().round(2).fillna(0)
        km_socio_moda = df.groupby('cluster_sociodemografico')[vars_socio].agg(lambda x: x.mode()[0])
        
        km_socio_disp = pd.DataFrame(index=km_socio_mean.index)
        for c in vars_socio:
            km_socio_disp[f"{c} (Mean±Std)"] = km_socio_mean[c].astype(str) + " ± " + km_socio_std[c].astype(str)
            km_socio_disp[f"{c} (Moda)"] = km_socio_moda[c]
            
        km_socio_disp.index = [f"Cluster {i}" if i != -1 else "Grupo 18" for i in km_socio_disp.index]
        st.dataframe(km_socio_disp, use_container_width=True)

    # ── Perfil LCA Sociodemográfico ──
    if has_lca_socio:
        st.markdown("---")
        st.markdown("#### 🟢 Perfil LCA Sociodemográfico (Mean ± Std & Moda)")
        vars_socio = ['age', 'gender', 'acquisition_channel', 'membership_tier', 'preferred_payment_method']
        
        lca_socio_mean = df.groupby('cluster_sociodemografico_lca')[vars_socio].mean().round(2)
        lca_socio_std = df.groupby('cluster_sociodemografico_lca')[vars_socio].std().round(2).fillna(0)
        lca_socio_moda = df.groupby('cluster_sociodemografico_lca')[vars_socio].agg(lambda x: x.mode()[0])
        
        lca_socio_disp = pd.DataFrame(index=lca_socio_mean.index)
        for c in vars_socio:
            lca_socio_disp[f"{c} (Mean±Std)"] = lca_socio_mean[c].astype(str) + " ± " + lca_socio_std[c].astype(str)
            lca_socio_disp[f"{c} (Moda)"] = lca_socio_moda[c]
            
        lca_socio_disp.index = [f"Clase {i}" if i != -1 else "Grupo 18" for i in lca_socio_disp.index]
        st.dataframe(lca_socio_disp, use_container_width=True)


# ════════════════════════════════════════════
# TAB 9: CRUCE DE SEGMENTOS
# ════════════════════════════════════════════
with tab_cruce:
    st.markdown('<div class="section-title">Cruce de Segmentos: RFM × Sociodemográfico</div>', unsafe_allow_html=True)

    has_lca_rfm = 'lca_cluster_rfm' in df.columns
    has_lca_socio = 'cluster_sociodemografico_lca' in df.columns

    if not has_lca_rfm or not has_lca_socio:
        st.warning("⚠️ Para ver el cruce completo de segmentos, activa los modelos LCA desde el sidebar y espera a que se calculen en las pestañas correspondientes.")
        
        # Solo K-Means vs K-Means
        st.markdown("#### K-Means RFM (k=4) × K-Means Sociodemográfico (k=3)")
        pivot_km = pd.pivot_table(
            df, values='customer_id', index='cluster_rfmkmeans',
            columns='cluster_sociodemografico', aggfunc='count', fill_value=0
        )
        pivot_km.index = [f"RFM-Cluster {i}" for i in pivot_km.index]
        pivot_km.columns = [f"Socio-Cluster {i}" if i != -1 else "Socio-Grupo 18" for i in pivot_km.columns]

        fig_heat_km = px.imshow(
            pivot_km.values,
            x=pivot_km.columns.tolist(),
            y=pivot_km.index.tolist(),
            color_continuous_scale='Viridis',
            text_auto=True,
            title='<b>Cruce K-Means RFM × K-Means Sociodemográfico (# clientes)</b>',
            aspect='auto',
        )
        fig_heat_km.update_layout(template='plotly_dark', height=400, font=dict(family='Inter'))
        st.plotly_chart(fig_heat_km, use_container_width=True)

    else:
        # ── Heatmap principal: LCA RFM × LCA Socio (como en el notebook) ──
        st.markdown("#### 🔥 Heatmap Principal: Clases RFM × Clases Demográficas (LCA)")

        pivot = pd.pivot_table(
            df, values='customer_id',
            index='lca_cluster_rfm',
            columns='cluster_sociodemografico_lca',
            aggfunc='count', fill_value=0
        )

        # Labels descriptivos
        rfm_labels = {
            0: "RFM-0 |  Cliente Ocasional (Promedio segmento ≈ $629 USD)",
            1: "RFM-1 |  Cliente Recurrente inactivo con presupuesto medio \n(Promedio Segmento ≈ $2786 USD)",
            2: "RFM-2 |  Cliente Frecuente con alto presupuesto\n(Promedio segmento≈ $6900 USD)",
            3: "RFM-3 |  Cliente Premium frecuente\n(Promedio Segmento ≈ $18264 USD)",
        }
        dem_labels = {
            -1: "Demo - (-1)\nHombre de\n18 años\nbúsqueda orgánica\nmembresía gratis\nY paga con TC",
            0: "Demo-0\nHombre de \n22 - 30  años que usa\nbusqueda organica tiene\nmembresia gratis \nY paga con TC",
            1: "Demo-1\nMujer de \n35 - 42 años\nbusqueda organica\nmembresia gratis \nY paga con TC",
            2: "Demo-2\nMujer de \n47- 58 años\nbusqueda organica\nmembresia gratis \nY paga con TC"
        }


        # Ordenar y renombrar índices
        pivot = pivot.reindex(sorted(pivot.index), axis=0)
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        try:
            pivot = pivot.rename(index=rfm_labels, columns=dem_labels)
        except Exception:
            pass

        fig_heat = px.imshow(
            pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            color_continuous_scale='RdPu',
            text_auto=True,
            title='<b>Cruce de Segmentos: Clases Demográficas vs. Clases RFM (LCA)</b>',
            aspect='auto',
            labels=dict(color='# Clientes'),
        )
        fig_heat.update_xaxes(tickangle=-20)
        fig_heat.update_layout(template='plotly_dark', height=500, font=dict(family='Inter', size=11))
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("""
        <div class="info-box">
        <strong>Cómo leer el heatmap:</strong> Cada celda muestra el número de clientes que pertenecen 
        simultáneamente a una clase RFM (filas) y una clase Demográfica (columnas). 
        Las celdas más oscuras indican mayor concentración de clientes en esa combinación de segmentos.
        </div>
        """, unsafe_allow_html=True)


        # ── Parallel Categories ──
        st.markdown("---")
        st.markdown("#### 📐 Flujos de Segmentación: Parallel Categories")

        col_pc1, col_pc2 = st.columns(2)

        with col_pc1:
            st.markdown("**K-Means: RFM vs Sociodemográfico**")
            df_pc_km = pd.DataFrame({
                'KMeans_RFM': df['cluster_rfmkmeans'].astype(str),
                'KMeans_Socio': df['cluster_sociodemografico'].astype(str),
                'color': df['cluster_rfmkmeans']
            })
            fig_pc_km = px.parallel_categories(
                df_pc_km, dimensions=['KMeans_RFM', 'KMeans_Socio'],
                color='color', color_continuous_scale='Viridis',
                title='<b>K-Means RFM → Sociodemográfico</b>',
            )
            fig_pc_km.update_layout(template='plotly_dark', height=400, font=dict(family='Inter'))
            st.plotly_chart(fig_pc_km, use_container_width=True)

        with col_pc2:
            st.markdown("**LCA: RFM vs Sociodemográfico**")
            df_pc_lca = pd.DataFrame({
                'LCA_RFM': df['lca_cluster_rfm'].astype(str),
                'LCA_Socio': df['cluster_sociodemografico_lca'].astype(str),
                'color': df['lca_cluster_rfm']
            })
            fig_pc_lca = px.parallel_categories(
                df_pc_lca, dimensions=['LCA_RFM', 'LCA_Socio'],
                color='color', color_continuous_scale='Plasma',
                title='<b>LCA RFM → LCA Sociodemográfico</b>',
            )
            fig_pc_lca.update_layout(template='plotly_dark', height=400, font=dict(family='Inter'))
            st.plotly_chart(fig_pc_lca, use_container_width=True)

        # ── Tabla cruzada K-Means RFM vs LCA RFM ──
        st.markdown("---")
        st.markdown("#### 📋 Matrices de Cruce: K-Means vs LCA")

        tab_cross_rfm, tab_cross_socio = st.tabs(["RFM: K-Means vs LCA", "Socio: K-Means vs LCA"])

        with tab_cross_rfm:
            tabla_km_lca_rfm = pd.crosstab(
                df['cluster_rfmkmeans'], df['lca_cluster_rfm'], margins=True
            )
            tabla_km_lca_rfm.index = [f"KMeans-{i}" if i != 'All' else 'Total' for i in tabla_km_lca_rfm.index]
            tabla_km_lca_rfm.columns = [f"LCA-{i}" if i != 'All' else 'Total' for i in tabla_km_lca_rfm.columns]
            st.markdown("**Matriz de cruce: K-Means RFM vs LCA RFM**")
            _rows_rfm2 = [r for r in tabla_km_lca_rfm.index if r != 'Total']
            _cols_rfm2 = [c for c in tabla_km_lca_rfm.columns if c != 'Total']
            st.dataframe(tabla_km_lca_rfm.style.background_gradient(cmap='Blues',
                         subset=pd.IndexSlice[_rows_rfm2, _cols_rfm2]), use_container_width=True)

        with tab_cross_socio:
            tabla_km_lca_socio = pd.crosstab(
                df['cluster_sociodemografico_lca'], df['cluster_sociodemografico'], margins=True
            )
            tabla_km_lca_socio.index = [f"LCA-{i}" if i != 'All' else 'Total' for i in tabla_km_lca_socio.index]
            tabla_km_lca_socio.columns = [f"KM-{i}" if i != 'All' else 'Total' for i in tabla_km_lca_socio.columns]
            st.markdown("**Matriz de cruce: LCA Socio vs K-Means Socio**")
            _rows_socio2 = [r for r in tabla_km_lca_socio.index if r != 'Total']
            _cols_socio2 = [c for c in tabla_km_lca_socio.columns if c != 'Total']
            st.dataframe(tabla_km_lca_socio.style.background_gradient(cmap='Purples',
                         subset=pd.IndexSlice[_rows_socio2, _cols_socio2]), use_container_width=True)



# ════════════════════════════════════════════
# TAB 10: ESTRATEGIA COMERCIAL
# ════════════════════════════════════════════
with tab_estrategia:
    st.markdown('<div class="section-title">🎯 Estrategia Comercial: Categorías y Segmentos</div>', unsafe_allow_html=True)

    with st.spinner("Construyendo dataset enriquecido (orders ✕ customers)..."):
        df_merged, seg_cat, copurchase_cross, monthly_cat = build_enriched(df_orders_raw, df_customers_raw)

    st.markdown("""
    <div class="info-box">
    <strong>Fuente:</strong> Merge completo de <code>orders.csv</code> + <code>customers.csv</code> 
    ({:,} transacciones de {:,} clientes)<br>
    <strong>Objetivo:</strong> Identificar qué categorías atacar por segmento, detectar patrones de co-compra
    y emitir recomendaciones accionables para la empresa.
    </div>
    """.format(len(df_merged), df_merged['customer_id'].nunique()), unsafe_allow_html=True)

    # ────────────────────────────────────────────
    # SECCIÓN 1: REVENUE POR CATEGORÍA DE PRODUCTO
    # ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 1. Revenue y Volumen por Categoría de Producto")

    cat_summary = df_merged.groupby('category').agg(
        revenue_total=('total_amount_usd', 'sum'),
        n_orders=('order_id', 'count'),
        n_clientes=('customer_id', 'nunique'),
        ticket_prom=('total_amount_usd', 'mean'),
        desc_prom=('discount_pct', 'mean'),
        pct_devuelto=('returned', 'mean'),
    ).reset_index().sort_values('revenue_total', ascending=False)
    cat_summary.columns = ['Categoría', 'Revenue Total', '# Órdenes',
                           '# Clientes', 'Ticket Promedio', 'Desc. Medio (%)', 'Tasa Devolución']

    c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
    c_kpi1.metric("🏆 Top Categoría", cat_summary.iloc[0]['Categoría'],
                   delta=f"${cat_summary.iloc[0]['Revenue Total']:,.0f} revenue")
    c_kpi2.metric("⚡ Menor Devolución",
                   cat_summary.nsmallest(1, 'Tasa Devolución').iloc[0]['Categoría'],
                   delta=f"{cat_summary.nsmallest(1,'Tasa Devolución').iloc[0]['Tasa Devolución']:.1%} devol.")
    c_kpi3.metric("💰 Mayor Ticket Medio",
                   cat_summary.nlargest(1, 'Ticket Promedio').iloc[0]['Categoría'],
                   delta=f"${cat_summary.nlargest(1,'Ticket Promedio').iloc[0]['Ticket Promedio']:,.0f}")

    col_rev1, col_rev2 = st.columns([1.4, 1])
    with col_rev1:
        fig_cat_rev = px.bar(
            cat_summary, x='Revenue Total', y='Categoría',
            orientation='h',
            color='Revenue Total', color_continuous_scale='Viridis',
            text=cat_summary['Revenue Total'].map(lambda x: f"${x/1e6:.1f}M"),
            title='<b>Revenue Total por Categoría de Producto</b>',
        )
        fig_cat_rev.update_traces(textposition='outside')
        fig_cat_rev.update_layout(template='plotly_dark', height=480,
                                   font=dict(family='Inter'), showlegend=False,
                                   yaxis=dict(autorange='reversed'),
                                   coloraxis_showscale=False)
        st.plotly_chart(fig_cat_rev, use_container_width=True)

    with col_rev2:
        fig_bubble = px.scatter(
            cat_summary, x='Ticket Promedio', y='Tasa Devolución',
            size='Revenue Total', color='Desc. Medio (%)',
            hover_name='Categoría',
            color_continuous_scale='RdYlGn_r',
            title='<b>Ticket vs Devolución (tamaño=Revenue)</b>',
            labels={'Ticket Promedio': 'Ticket Promedio (USD)', 'Tasa Devolución': 'Tasa Devolución'},
        )
        fig_bubble.update_layout(template='plotly_dark', height=480, font=dict(family='Inter'))
        st.plotly_chart(fig_bubble, use_container_width=True)

    st.dataframe(
        cat_summary.style.format({
            'Revenue Total': '${:,.0f}',
            'Ticket Promedio': '${:,.2f}',
            'Desc. Medio (%)': '{:.1f}%',
            'Tasa Devolución': '{:.1%}',
        })
        .background_gradient(subset=['Revenue Total'], cmap='Greens')
        .background_gradient(subset=['Tasa Devolución'], cmap='Reds'),
        use_container_width=True
    )

    # ────────────────────────────────────────────
    # SECCIÓN 2: CATEGORÍA PREFERIDA × CATEGORÍA COMPRADA
    # ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔄 2. Desglose de Movimiento por Categoría Preferida")
    st.markdown("""
    <div class="info-box">
    Para cada grupo de clientes según su <strong>categoría preferida</strong>,
    se muestra qué otras categorías compran efectivamente en sus órdenes. 
    Esto permite detectar oportunidades de <strong>cross-selling</strong>.
    </div>
    """, unsafe_allow_html=True)

    # Selector de categoría preferida
    all_pref_cats = sorted(df_customers_raw['preferred_category'].dropna().unique())
    sel_pref = st.selectbox(
        "Selecciona la categoría preferida del segmento de clientes:",
        options=all_pref_cats,
        key='sel_pref_cat'
    )

    # Clientes con esa categoría preferida
    clientes_pref = df_customers_raw[df_customers_raw['preferred_category'] == sel_pref]['customer_id'].unique()
    ordenes_pref = df_merged[df_merged['customer_id'].isin(clientes_pref)]

    col_cross1, col_cross2 = st.columns([1.3, 1])
    with col_cross1:
        # Qué categorías compran realmente
        cat_compradas = ordenes_pref.groupby('category').agg(
            n_ordenes=('order_id', 'count'),
            revenue=('total_amount_usd', 'sum'),
            n_clientes=('customer_id', 'nunique'),
            ticket=('total_amount_usd', 'mean'),
            desc_medio=('discount_pct', 'mean'),
        ).reset_index().sort_values('revenue', ascending=False)
        cat_compradas.columns = ['Categoría Comprada', '# Órdenes', 'Revenue', '# Clientes', 'Ticket Medio', 'Desc. Medio (%)']
        cat_compradas['Es preferida'] = cat_compradas['Categoría Comprada'] == sel_pref

        fig_cross = px.bar(
            cat_compradas, x='Categoría Comprada', y='Revenue',
            color='Es preferida',
            color_discrete_map={True: '#7c3aed', False: '#2563eb'},
            text=cat_compradas['Revenue'].map(lambda x: f"${x:,.0f}"),
            title=f'<b>Compras reales de clientes con preferencia: {sel_pref}</b>',
        )
        fig_cross.update_traces(textposition='outside', textfont_size=10)
        fig_cross.update_layout(template='plotly_dark', height=440, font=dict(family='Inter'),
                                 showlegend=True, xaxis_tickangle=-35)
        st.plotly_chart(fig_cross, use_container_width=True)

    with col_cross2:
        # Pie de distribución
        fig_cross_pie = px.pie(
            cat_compradas, values='Revenue', names='Categoría Comprada',
            title=f'<b>Distribución de Revenue – Clientes pref. {sel_pref}</b>',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig_cross_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=11)
        fig_cross_pie.update_layout(template='plotly_dark', height=440, font=dict(family='Inter'),
                                     showlegend=False)
        st.plotly_chart(fig_cross_pie, use_container_width=True)

    st.dataframe(
        cat_compradas.style.format({
            'Revenue': '${:,.0f}',
            'Ticket Medio': '${:,.2f}',
            'Desc. Medio (%)': '{:.1f}%',
        })
        .background_gradient(subset=['Revenue'], cmap='Blues')
        .apply(lambda row: ['background: rgba(124,58,237,.15)' if row['Es preferida'] else '' for _ in row], axis=1),
        use_container_width=True, height=260
    )

    # ────────────────────────────────────────────
    # SECCIÓN 3: HEATMAP CO-COMPRA GLOBAL
    # ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧱 3. Mapa de Co-compra: Categoría Preferida → Categoría Real")
    st.markdown("""
    <div class="info-box">
    El mapa muestra cuánto revenue generan las <strong>compras cruzadas</strong>: 
    filas = categoría preferida del cliente, columnas = categoría real comprada (excluyendo la misma).
    Las celdas más oscuras son las mejores oportunidades de <strong>cross-selling</strong>.
    </div>
    """, unsafe_allow_html=True)

    pivot_copurchase = copurchase_cross.pivot_table(
        index='preferred_category', columns='category',
        values='revenue', aggfunc='sum', fill_value=0
    )
    fig_copurchase = px.imshow(
        pivot_copurchase.values,
        x=pivot_copurchase.columns.tolist(),
        y=pivot_copurchase.index.tolist(),
        color_continuous_scale='Plasma',
        text_auto=False,
        title='<b>Revenue de Co-compra: Categoría Preferida × Categoría Real Comprada (USD)</b>',
        aspect='auto',
        labels=dict(color='Revenue (USD)', x='Categoría Comprada', y='Categoría Preferida'),
    )
    fig_copurchase.update_xaxes(tickangle=-40)
    fig_copurchase.update_layout(template='plotly_dark', height=540, font=dict(family='Inter', size=11))
    st.plotly_chart(fig_copurchase, use_container_width=True)

    # Top 5 oportunidades globales de cross-sell
    top_cross = copurchase_cross.sort_values('revenue', ascending=False).head(10).copy()
    top_cross.columns = ['Categoría Preferida', 'Categoría Cross-sell', '# Clientes', 'Revenue Cross-sell']
    st.markdown("**🔥 Top 10 oportunidades de cross-selling por revenue:**")
    st.dataframe(
        top_cross.style.format({'Revenue Cross-sell': '${:,.0f}'})
        .background_gradient(subset=['Revenue Cross-sell'], cmap='YlOrRd'),
        use_container_width=True
    )

    # ────────────────────────────────────────────
    # SECCIÓN 4: RECOMENDACIONES ESTRATÉGICAS – SEGMENTO CLIENTE OCASIONAL
    # ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 4. Recomendaciones Estratégicas: Segmento Cliente Ocasional (`df_target_selectiva`)")

    has_lca_rfm_strat = 'lca_cluster_rfm' in df.columns
    has_lca_socio_strat = 'cluster_sociodemografico_lca' in df.columns

    if not has_lca_rfm_strat or not has_lca_socio_strat:
        st.warning("Para ver el análisis del segmento ocasional, activa los modelos LCA desde el sidebar y ejecuta primero las pestañas LCA.")
    else:
        st.markdown("""
        <div class="info-box">
        <strong>Definición del segmento:</strong> Clientes con <strong>Clase RFM-0</strong>
        (Cliente de entrada / Ocasional, ≈$629 USD) y perfil demográfico <strong>Demo-1</strong>
        (mujer ~[35 - 42] años, búsqueda orgánica, membresía Free, paga con Tarjeta de Crédito).<br>
        <strong>Filtro aplicado:</strong> <code>lca_cluster_rfm == 0</code> AND <code>cluster_sociodemografico_lca == 1</code>
        </div>
        """, unsafe_allow_html=True)

        # Crear df_target_ocasional
        df_target_ocasional = df[
            (df['cluster_sociodemografico_lca'] == 1) &
            (df['lca_cluster_rfm'] == 0)
        ].copy()

        # KPIs del segmento
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Clientes en segmento", f"{len(df_target_ocasional):,}",
                  delta=f"{len(df_target_ocasional)/len(df)*100:.1f}% del total")
        t2.metric("Monetary medio", f"${df_target_ocasional['monetary'].mean():,.0f}")
        t3.metric("Recency media", f"{df_target_ocasional['recency'].mean():.0f} dias")
        t4.metric("Frecuencia media", f"{df_target_ocasional['frequency'].mean():.1f}")

        st.markdown("---")

        # Filtrar df_merged con los IDs del segmento
        target_ids = df_target_ocasional['customer_id'].values
        df_merged_target = df_merged[df_merged['customer_id'].isin(target_ids)].copy()

        if len(df_merged_target) > 0:
            st.markdown("#### Analisis de Venta Cruzada: Que compran segun su Categoria Preferida?")
            st.markdown("""
            <div class="info-box">
            Filas = categoría preferida del cliente · Columnas = categoría real comprada en sus órdenes.<br>
            Las celdas más oscuras son las mejores oportunidades de <strong>cross-selling</strong> para este segmento.
            </div>
            """, unsafe_allow_html=True)

            # Matriz de complementariedad
            matriz_comp2 = pd.crosstab(
                df_merged_target['preferred_category'],
                df_merged_target['category'],
                normalize='index'
            ) * 100

            fig_matriz2 = px.imshow(
                matriz_comp2.values,
                x=matriz_comp2.columns.tolist(),
                y=matriz_comp2.index.tolist(),
                color_continuous_scale='Blues',
                text_auto='.1f',
                aspect='auto',
                labels=dict(color='% Ordenes', x='Categoria Comprada', y='Categoria Preferida'),
            )
            fig_matriz2.update_layout(
                title='<b>Analisis de Venta Cruzada - Segmento Cliente Ocasional (Demo-0 x RFM-0)<br>'
                      '(% de ordenes: Filas = Cat. Preferida | Columnas = Cat. Comprada)</b>',
                template='plotly_dark', height=560,
                font=dict(family='Inter', size=10),
            )
            fig_matriz2.update_xaxes(tickangle=-40)
            st.plotly_chart(fig_matriz2, use_container_width=True)

            # Top 10 oportunidades de cross-sell en el segmento
            cross_seg = df_merged_target.groupby(['preferred_category', 'category']).agg(
                n_ordenes=('order_id', 'count'),
                revenue=('total_amount_usd', 'sum'),
                n_clientes=('customer_id', 'nunique'),
            ).reset_index()
            cross_seg = cross_seg[cross_seg['preferred_category'] != cross_seg['category']]
            cross_seg = cross_seg.sort_values('revenue', ascending=False).head(10).reset_index(drop=True)
            cross_seg.columns = ['Cat. Preferida', 'Cat. Cross-sell', '# Ordenes', 'Revenue', '# Clientes']

            st.markdown("**Top 10 oportunidades de cross-selling - Segmento Ocasional:**")
            st.dataframe(
                cross_seg.style.format({'Revenue': '${:,.0f}'})
                .background_gradient(subset=['Revenue'], cmap='YlOrRd'),
                use_container_width=True
            )
        else:
            st.info("No se encontraron ordenes para clientes de este segmento.")

        # Perfil descriptivo del segmento
        st.markdown("---")
        st.markdown("#### Perfil Descriptivo del Segmento Ocasional")
        perfil_cols_disp = ['age', 'recency', 'frequency', 'monetary']
        perfil_target_stats = df_target_ocasional[perfil_cols_disp].describe().round(2).loc[['mean', 'std', 'min', '50%', 'max']]
        perfil_target_stats.index = ['Media', 'Desv. Est.', 'Minimo', 'Mediana', 'Maximo']
        perfil_target_stats.columns = ['Edad', 'Recency (dias)', 'Frequency', 'Monetary (USD)']
        st.dataframe(
            perfil_target_stats.style.format("{:.2f}")
            .background_gradient(cmap='Blues'),
            use_container_width=True
        )

        # ────────────────────────────────────────────
        # GRÁFICOS DE RESPALDO PARA LA ESTRATEGIA
        # ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📈 Gráficos de Respaldo Estratégico")
        
        col_resp1, col_resp2 = st.columns(2)
        
        with col_resp1:
            st.markdown("**1. Categorías más compradas por el segmento** *(Justifica descuentos por volumen)*")
            top_cat_seg = df_merged_target.groupby('category')['quantity'].sum().sort_values(ascending=False).reset_index()
            fig_top_cat = px.bar(
                top_cat_seg, x='quantity', y='category', orientation='h',
                color='quantity', color_continuous_scale='viridis',
                labels={'quantity': 'Cantidad Total Comprada', 'category': 'Categoría'}
            )
            fig_top_cat.update_layout(template='plotly_dark', height=400, font=dict(family='Inter', size=11))
            fig_top_cat.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_top_cat, use_container_width=True)

        with col_resp2:
            st.markdown("**2. Impacto de la Membresía en el Gasto Total** *(Justifica incentivos de membresía)*")
            # Este gráfico se calcula sobre la base completa de clientes (df) para validar la diferencia entre tiers
            membership_spend = df.groupby('membership_tier')['monetary'].mean().sort_values(ascending=False).reset_index()
            fig_mem_spend = px.bar(
                membership_spend, x='membership_tier', y='monetary',
                color='monetary', color_continuous_scale='magma',
                labels={'monetary': 'Gasto Promedio Total (USD)', 'membership_tier': 'Nivel de Membresía'}
            )
            fig_mem_spend.update_layout(template='plotly_dark', height=400, font=dict(family='Inter', size=11))
            st.plotly_chart(fig_mem_spend, use_container_width=True)

    # ────────────────────────────────────────────
    # SECCIÓN 5: EVOLUCIÓN MENSUAL POR CATEGORÍA
    # ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📅 5. Evolución Mensual de Revenue por Categoría")


    top_cats = cat_summary.head(6)['Categoría'].tolist()
    monthly_top = monthly_cat[monthly_cat['category'].isin(top_cats)].copy()

    fig_monthly = px.line(
        monthly_top, x='period', y='revenue', color='category',
        title='<b>Revenue Mensual – Top 6 Categorías</b>',
        labels={'period': 'Período', 'revenue': 'Revenue (USD)', 'category': 'Categoría'},
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig_monthly.update_layout(template='plotly_dark', height=420, font=dict(family='Inter'),
                               hovermode='x unified')
    st.plotly_chart(fig_monthly, use_container_width=True)




# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.caption("📚 Trabajo 2 – Marketing – DII UdeC 2026")
with col_f2:
    st.markdown(
        "<div style='text-align:right; color:#6e7681; font-size:.8em;'>"
        "Grupo 29 · Francisco Araneda · Benjamín Borquez · Martín Lagos · Camilo Mora · Isidora Salas"
        "</div>",
        unsafe_allow_html=True
    )
