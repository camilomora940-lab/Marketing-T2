"""
Streamlit – Simulador de Estrategia de Promociones por Segmento y Categoría
Basado en el análisis LCA de Trabajo_2_def.ipynb
Grupo 29 – Marketing – DII UdeC 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Simulador de Promociones – Grupo 29",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ESTILOS CSS PREMIUM
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary: #0e1117;
    --bg-card: #1a1d27;
    --bg-card-hover: #22263a;
    --accent-1: #6366f1;
    --accent-2: #8b5cf6;
    --accent-3: #06b6d4;
    --accent-4: #f59e0b;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border-color: #2d3348;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --gradient-1: linear-gradient(135deg, #6366f1, #8b5cf6);
    --gradient-2: linear-gradient(135deg, #06b6d4, #3b82f6);
    --gradient-3: linear-gradient(135deg, #f59e0b, #ef4444);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,.25);
    transition: transform .2s, box-shadow .2s;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99,102,241,.15);
}
div[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-weight: 500;
    font-size: .85rem;
    text-transform: uppercase;
    letter-spacing: .04em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-weight: 700;
    font-size: 1.7rem;
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: .95rem;
    letter-spacing: .02em;
    border-radius: 8px 8px 0 0;
    padding: 10px 24px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    border-right: 1px solid var(--border-color);
}

/* ── Glass card helper ── */
.glass-card {
    background: rgba(26, 29, 39, .85);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}
.glass-card h3 {
    margin-top: 0;
    color: var(--text-primary);
}

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, rgba(99,102,241,.15), rgba(139,92,246,.10));
    border: 1px solid rgba(99,102,241,.25);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
}
.hero h1 {
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 8px;
}
.hero p {
    color: var(--text-secondary);
    font-size: 1.05rem;
    margin: 0;
}

/* ── Segment badge ── */
.seg-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: .82rem;
    letter-spacing: .02em;
    margin: 2px 4px;
}

/* ── Result box ── */
.result-box {
    background: rgba(16, 185, 129, .08);
    border: 1px solid rgba(16, 185, 129, .30);
    border-radius: 14px;
    padding: 20px 24px;
    margin: 12px 0;
}
.result-box.warning {
    background: rgba(245, 158, 11, .08);
    border-color: rgba(245, 158, 11, .30);
}
.result-box.danger {
    background: rgba(239, 68, 68, .08);
    border-color: rgba(239, 68, 68, .30);
}

/* ── Smooth transitions ── */
.stSelectbox, .stMultiSelect, .stSlider {
    transition: all .2s ease;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────
CATEGORY_LIST = [
    'Food & Grocery', 'Toys & Games', 'Home & Kitchen', 'Electronics',
    'Clothing & Apparel', 'Office Supplies', 'Sports & Outdoors',
    'Beauty & Personal Care', 'Books', 'Health & Wellness',
    'Travel & Luggage', 'Jewelry & Accessories', 'Pet Supplies', 'Automotive'
]

SEGMENT_COLORS = {
    0: '#6366f1',  # Indigo
    1: '#06b6d4',  # Cyan
    2: '#f59e0b',  # Amber
    3: '#ef4444',  # Red
    4: '#10b981',  # Emerald
    5: '#ec4899',  # Pink
    6: '#8b5cf6',  # Violet
}

SEGMENT_NAMES = {
    0: 'Segmento 0',
    1: 'Segmento 1',
    2: 'Segmento 2',
    3: 'Segmento 3',
}

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

GENDER_MAP = {'Male': 0, 'Female': 1, 'Other': 2}
MEMBERSHIP_MAP = {'Free': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}
ACQUISITION_MAP = {
    'Direct': 0, 'Social Media': 1, 'Organic Search': 2,
    'Email Campaign': 3, 'Paid Ad': 4, 'Referral': 5
}
CATEGORY_MAP = {cat: i for i, cat in enumerate(CATEGORY_LIST)}

# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df_orders = pd.read_csv('orders.csv')
    df_customers = pd.read_csv('customers.csv')
    return df_orders, df_customers


@st.cache_data
def build_promo_features(_df_orders, _df_customers):
    """Construye features de promoción y segmenta clientes."""
    # Features de promoción por cliente
    promo = _df_orders.groupby('customer_id').agg(
        avg_discount_pct=('discount_pct', 'mean'),
        max_discount_pct=('discount_pct', 'max'),
        pct_orders_with_discount=('discount_pct', lambda x: (x > 0).mean()),
        avg_discount_amount=('discount_amount_usd', 'mean'),
        total_discount_saved=('discount_amount_usd', 'sum'),
        total_revenue=('total_amount_usd', 'sum'),
        n_orders=('order_id', 'count'),
        avg_session_duration=('session_duration_minutes', 'mean'),
        avg_pages_viewed=('pages_viewed_before_purchase', 'mean'),
    ).reset_index()

    # Features de promoción por categoría por cliente
    cat_disc = _df_orders.groupby(['customer_id', 'category']).agg(
        cat_avg_disc=('discount_pct', 'mean'),
        cat_orders=('order_id', 'count'),
        cat_revenue=('total_amount_usd', 'sum'),
    ).reset_index()

    # Merge
    df = _df_customers.copy()
    df = df.merge(promo, on='customer_id', how='left').fillna(0)

    # Encodings
    df['region'] = df['country'].map(REGION_MAP)
    df['region_code'] = df['region'].map(REGION_CODE)
    df['gender_code'] = df['gender'].map(GENDER_MAP)
    df['membership_code'] = df['membership_tier'].map(MEMBERSHIP_MAP)
    df['acquisition_code'] = df['acquisition_channel'].map(ACQUISITION_MAP)
    df['category_code'] = df['preferred_category'].map(CATEGORY_MAP)

    # RFM rename
    df = df.rename(columns={
        'days_since_last_purchase': 'recency',
        'total_orders': 'frequency',
        'total_spend_usd': 'monetary',
    })

    return df, cat_disc


@st.cache_data
def fit_lca_model(_df):
    """Ajusta el modelo LCA con StepMix."""
    try:
        from stepmix.stepmix import StepMix
        from stepmix.utils import get_mixed_descriptor

        mm_data, mm_desc = get_mixed_descriptor(
            dataframe=_df,
            categorical=['gender_code', 'region_code', 'category_code',
                         'membership_code', 'acquisition_code'],
            continuous=['age', 'monetary', 'recency', 'frequency',
                        'avg_discount_pct', 'pct_orders_with_discount',
                        'total_discount_saved', 'avg_order_value_usd',
                        'wishlist_items'],
            binary=['newsletter_subscribed']
        )

        # Selección de K por BIC
        bic_vals = {}
        models = {}
        for k in range(2, 7):
            m = StepMix(n_components=k, measurement=mm_desc,
                        random_state=42, n_init=10, max_iter=1000, verbose=0)
            m.fit(mm_data)
            bic_vals[k] = m.bic(mm_data)
            models[k] = m

        best_k = min(bic_vals, key=bic_vals.get)
        best_model = models[best_k]

        labels = best_model.predict(mm_data)
        posteriors = best_model.predict_proba(mm_data)

        return labels, posteriors, bic_vals, best_k, mm_data
    except Exception as e:
        st.error(f"Error al ajustar StepMix: {e}")
        return None, None, None, None, None


@st.cache_data
def fit_lca_fixed_k(_df, k=4):
    """Ajusta LCA con un K fijo."""
    try:
        # pyrefly: ignore [missing-import]
        from stepmix.stepmix import StepMix
        # pyrefly: ignore [missing-import]
        from stepmix.utils import get_mixed_descriptor

        mm_data, mm_desc = get_mixed_descriptor(
            dataframe=_df,
            categorical=['gender_code', 'region_code', 'category_code',
                         'membership_code', 'acquisition_code'],
            continuous=['age', 'monetary', 'recency', 'frequency',
                        'avg_discount_pct', 'pct_orders_with_discount',
                        'total_discount_saved', 'avg_order_value_usd',
                        'wishlist_items'],
            binary=['newsletter_subscribed']
        )

        model = StepMix(n_components=k, measurement=mm_desc,
                        random_state=42, n_init=10, max_iter=1000, verbose=0)
        model.fit(mm_data)

        labels = model.predict(mm_data)
        posteriors = model.predict_proba(mm_data)

        # Entropía relativa
        eps = 1e-15
        ent = -np.sum(posteriors * np.log(posteriors + eps))
        rel_ent = 1.0 - (ent / (len(posteriors) * np.log(k)))

        return labels, posteriors, rel_ent
    except Exception as e:
        st.error(f"Error al ajustar StepMix: {e}")
        return None, None, None


# ─────────────────────────────────────────────
# CARGA Y PROCESAMIENTO
# ─────────────────────────────────────────────
df_orders, df_customers = load_data()
df_seg, cat_disc = build_promo_features(df_orders, df_customers)

# ── Sidebar ──
import os
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", use_container_width=True)

st.sidebar.markdown("""
<div style="text-align:center; padding: 16px 0;">
    <h2 style="margin:4px 0 0; font-weight:700;
       background: linear-gradient(135deg,#6366f1,#8b5cf6);
       -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    </h2>
    <p style="color:#94a3b8; font-size:.85rem; margin:0;">Segmentación de Mercado</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
n_classes = st.sidebar.slider("Número de clases latentes (K)", 2, 6, 4)

with st.sidebar.expander(" Acerca del modelo"):
    st.markdown("""
    **Modelo:** Latent Class Analysis (LCA) mixto con **StepMix**
    
    **Indicadores:**
    - Continuas: age, monetary, recency, frequency, descuentos
    - Categóricas: género, región, categoría, membresía, canal
    - Binarias: newsletter
    
    **Selección K:** Se elige por BIC mínimo
    """)

# ── Fit model ──
with st.spinner("Ajustando modelo LCA…"):
    labels, posteriors, rel_entropy = fit_lca_fixed_k(df_seg, k=n_classes)

if labels is not None:
    df_seg['segment'] = labels

    # Generar nombres descriptivos basados en perfil
    seg_profiles = df_seg.groupby('segment').agg({
        'monetary': 'mean',
        'pct_orders_with_discount': 'mean',
        'frequency': 'mean',
        'recency': 'mean',
    }).round(2)

    # Asignar nombres automáticos según perfil
    name_map = {}
    for seg in seg_profiles.index:
        row = seg_profiles.loc[seg]
        if row['pct_orders_with_discount'] > seg_profiles['pct_orders_with_discount'].median() \
                and row['monetary'] > seg_profiles['monetary'].median():
            name_map[seg] = f" Premium Promo-Activo"
        elif row['pct_orders_with_discount'] > seg_profiles['pct_orders_with_discount'].median():
            name_map[seg] = f" Cazador de Ofertas"
        elif row['monetary'] > seg_profiles['monetary'].median():
            name_map[seg] = f" Alto Valor"
        elif row['recency'] > seg_profiles['recency'].median():
            name_map[seg] = f" En Riesgo"
        else:
            name_map[seg] = f" Moderado"

    # Si hay duplicados de nombre, agregar índice
    seen = {}
    for k_seg, v in name_map.items():
        if v in seen:
            seen[v] += 1
            name_map[k_seg] = f"{v} {seen[v]}"
        else:
            seen[v] = 1

    SEGMENT_NAMES.update(name_map)
    df_seg['segment_name'] = df_seg['segment'].map(SEGMENT_NAMES)

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Trabajo 2: Segmentación de mercado</h1>
    <p>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_overview, tab_simulator, tab_categories, tab_model = st.tabs([
    "Vista General", "Simulador de Promociones",
    "Análisis por Categoría", "Modelo LCA"
])

# ═══════════════════════════════════════════════
# TAB 1: VISTA GENERAL
# ═══════════════════════════════════════════════
with tab_overview:
    if labels is None:
        st.error("No se pudo ajustar el modelo. Revise que StepMix esté instalado.")
        st.stop()

    # ── KPIs ──
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Clientes", f"{len(df_seg):,}")
    kpi2.metric("Segmentos", n_classes)
    kpi3.metric("Entropía Relativa", f"{rel_entropy:.3f}")
    kpi4.metric("Gasto Promedio", f"${df_seg['monetary'].mean():,.0f}")
    kpi5.metric("% Usan Descuento",
                f"{df_seg['pct_orders_with_discount'].mean():.0%}")

    st.markdown("---")

    # ── Perfil de segmentos ──
    st.subheader("Perfil de los Segmentos")

    profile_cols = ['age', 'monetary', 'recency', 'frequency',
                    'avg_order_value_usd', 'avg_discount_pct',
                    'pct_orders_with_discount', 'total_discount_saved',
                    'wishlist_items', 'reviews_given', 'returns_made',
                    'newsletter_subscribed', 'churned']

    profile = df_seg.groupby('segment_name')[profile_cols].mean().round(2)
    profile['n_clientes'] = df_seg.groupby('segment_name').size().values
    profile['% del total'] = (profile['n_clientes'] / len(df_seg) * 100).round(1)

    st.dataframe(
        profile.style.format({
            'monetary': '${:,.0f}',
            'avg_order_value_usd': '${:,.0f}',
            'total_discount_saved': '${:,.0f}',
            'avg_discount_pct': '{:.1f}%',
            'pct_orders_with_discount': '{:.0%}',
            'newsletter_subscribed': '{:.0%}',
            'churned': '{:.0%}',
            '% del total': '{:.1f}%',
        }).background_gradient(subset=['monetary'], cmap='Blues')
        .background_gradient(subset=['pct_orders_with_discount'], cmap='Oranges')
        .background_gradient(subset=['churned'], cmap='Reds'),
        use_container_width=True, height=250
    )

    # ── Charts ──
    col_a, col_b = st.columns(2)

    with col_a:
        # Radar chart de perfiles normalizados
        radar_cols = ['monetary', 'frequency', 'recency',
                      'avg_discount_pct', 'pct_orders_with_discount', 'wishlist_items']
        radar_data = df_seg.groupby('segment_name')[radar_cols].mean()
        # Normalizar 0-1
        radar_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min() + 1e-9)

        fig_radar = go.Figure()
        for seg_name in radar_norm.index:
            vals = radar_norm.loc[seg_name].tolist()
            vals.append(vals[0])  # cerrar el polígono
            cats = radar_cols + [radar_cols[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats, fill='toself',
                name=seg_name, opacity=0.6
            ))
        fig_radar.update_layout(
            title="<b>Perfil Normalizado por Segmento</b>",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            template='plotly_dark',
            height=420,
            font=dict(family='Inter'),
            legend=dict(orientation='h', y=-0.15)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_b:
        # Distribución de segmentos
        seg_counts = df_seg['segment_name'].value_counts().reset_index()
        seg_counts.columns = ['Segmento', 'Clientes']
        fig_pie = px.pie(
            seg_counts, values='Clientes', names='Segmento',
            title='<b>Distribución de Clientes por Segmento</b>',
            color_discrete_sequence=list(SEGMENT_COLORS.values()),
            hole=0.45,
        )
        fig_pie.update_layout(
            template='plotly_dark', height=420,
            font=dict(family='Inter'),
            legend=dict(orientation='h', y=-0.15)
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label',
                              textfont_size=12)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Scatter: Monetary vs Discount ──
    st.subheader("Gasto vs Receptividad a Descuentos")
    fig_scatter = px.scatter(
        df_seg, x='monetary', y='pct_orders_with_discount',
        color='segment_name', size='frequency',
        hover_data=['preferred_category', 'membership_tier', 'age'],
        labels={
            'monetary': 'Gasto Total (USD)',
            'pct_orders_with_discount': '% Pedidos con Descuento',
            'segment_name': 'Segmento',
            'frequency': 'Frecuencia',
        },
        title='<b>Relación Gasto – Uso de Descuentos por Segmento</b>',
        color_discrete_sequence=list(SEGMENT_COLORS.values()),
        opacity=0.55,
    )
    fig_scatter.update_layout(
        template='plotly_dark', height=500,
        font=dict(family='Inter'),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 2: SIMULADOR DE PROMOCIONES
# ═══════════════════════════════════════════════
with tab_simulator:
    if labels is None:
        st.error("Modelo no disponible."); st.stop()

    st.markdown("""
    <div class="glass-card">
        <h3> Simulador de Estrategia Promocional</h3>
        <p style="color:#94a3b8;">
            Selecciona un segmento objetivo, la categoría de producto y el nivel de descuento.
            El simulador proyecta el impacto estimado en ventas, ROI y alcance basándose en
            los patrones históricos de cada segmento.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_config, col_results = st.columns([1, 1.3])

    with col_config:
        st.markdown("####  Configuración de la Campaña")

        # Segmentos objetivo
        seg_options = sorted(df_seg['segment_name'].unique())
        target_segments = st.multiselect(
            "Segmentos objetivo",
            options=seg_options,
            default=[seg_options[0]] if seg_options else [],
            help="Selecciona uno o más segmentos a los que dirigir la promoción"
        )

        # Categorías
        target_categories = st.multiselect(
            "Categorías de producto",
            options=CATEGORY_LIST,
            default=['Electronics', 'Clothing & Apparel'],
            help="Categorías incluidas en la campaña"
        )

        st.markdown("---")

        # Nivel de descuento
        discount_level = st.slider(
            "Descuento ofrecido (%)", 5, 50, 15, step=5,
            help="Porcentaje de descuento a ofrecer en la campaña"
        )

        # Canal
        channels = st.multiselect(
            "Canales de adquisición a activar",
            ['Direct', 'Social Media', 'Organic Search', 'Email Campaign', 'Paid Ad', 'Referral'],
            default=['Email Campaign', 'Social Media']
        )

        st.markdown("---")
        run_sim = st.button(" Ejecutar Simulación", use_container_width=True,
                            type="primary")

    with col_results:
        if run_sim and target_segments and target_categories:
            st.markdown("####  Resultados Proyectados")

            # Filtrar clientes en segmentos objetivo
            df_target = df_seg[df_seg['segment_name'].isin(target_segments)]
            n_target = len(df_target)

            # Filtrar órdenes de esos clientes en las categorías seleccionadas
            target_ids = df_target['customer_id'].values
            df_orders_target = df_orders[
                (df_orders['customer_id'].isin(target_ids)) &
                (df_orders['category'].isin(target_categories))
            ]

            # ── Métricas históricas del segmento ──
            hist_avg_order_val = df_orders_target['total_amount_usd'].mean() if len(df_orders_target) > 0 else 0
            hist_discount_rate = (df_orders_target['discount_pct'] > 0).mean() if len(df_orders_target) > 0 else 0
            hist_avg_discount = df_orders_target['discount_pct'].mean() if len(df_orders_target) > 0 else 0
            hist_conversion = df_orders_target['is_repeat_customer'].mean() if len(df_orders_target) > 0 else 0

            # ── Simulación ──
            # Elasticidad estimada: aumento de descuento → aumento de tasa de compra
            discount_delta = max(0, discount_level - hist_avg_discount)
            # Elasticidad moderada: +1% descuento → +0.8% tasa de respuesta (ajustable)
            elasticity = 0.008
            base_response_rate = max(0.05, hist_discount_rate)
            projected_response_rate = min(0.95,
                base_response_rate + discount_delta * elasticity * base_response_rate)

            # Clientes que responderían
            n_respondents = int(n_target * projected_response_rate)

            # Ticket promedio ajustado (descuento reduce ticket pero aumenta volumen)
            projected_ticket = hist_avg_order_val * (1 - discount_level / 100)
            # Incremento de volumen por descuento
            volume_lift = 1.0 + (discount_level / 100) * 1.5  # Cada 1% desc → 1.5% más volumen
            projected_orders_per_customer = (df_target['frequency'].mean() / 12) * volume_lift

            # Revenue proyectado
            projected_revenue = n_respondents * projected_ticket * projected_orders_per_customer
            # Costo del descuento
            discount_cost = projected_revenue * (discount_level / (100 - discount_level))
            # Revenue bruto (antes del descuento)
            gross_revenue = projected_revenue + discount_cost
            # ROI de la campaña
            total_investment = campaign_budget + discount_cost
            roi = (projected_revenue - total_investment) / total_investment if total_investment > 0 else 0

            # ── Mostrar resultados ──
            r1, r2, r3 = st.columns(3)
            r1.metric(" Alcance", f"{n_target:,} clientes",
                      delta=f"{n_target/len(df_seg)*100:.1f}% del total")
            r2.metric(" Respuesta Estimada", f"{n_respondents:,}",
                      delta=f"{projected_response_rate:.0%} tasa")
            r3.metric(" Revenue Proyectado",
                      f"${projected_revenue:,.0f}",
                      delta=f"ROI {roi:.1%}")

            st.markdown("---")

            r4, r5, r6 = st.columns(3)
            r4.metric(" Ticket Promedio", f"${projected_ticket:,.2f}",
                      delta=f"-{discount_level}% vs normal")
            r5.metric(" Pedidos/cliente/mes", f"{projected_orders_per_customer:.2f}",
                      delta=f"+{(volume_lift-1)*100:.0f}% lift")
            r6.metric(" Costo Descuentos", f"${discount_cost:,.0f}")

            # ── Evaluación ──
            st.markdown("---")
            if roi > 0.5:
                st.markdown("""
                <div class="result-box">
                    <strong> Campaña Altamente Rentable</strong><br>
                    <span style="color:#94a3b8;">El ROI proyectado supera el 50%.
                    Se recomienda ejecutar la campaña e incluso considerar aumentar el presupuesto.</span>
                </div>
                """, unsafe_allow_html=True)
            elif roi > 0:
                st.markdown("""
                <div class="result-box warning">
                    <strong> Campaña Marginalmente Rentable</strong><br>
                    <span style="color:#94a3b8;">El ROI es positivo pero modesto.
                    Considere ajustar el descuento o focalizar en menos categorías para optimizar.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-box danger">
                    <strong> Campaña No Rentable</strong><br>
                    <span style="color:#94a3b8;">El costo de los descuentos supera el revenue incremental.
                    Reduzca el descuento o cambie de segmento objetivo.</span>
                </div>
                """, unsafe_allow_html=True)

            # ── Desglose por categoría ──
            st.markdown("####  Desglose por Categoría")
            cat_results = []
            for cat in target_categories:
                cat_orders = df_orders_target[df_orders_target['category'] == cat]
                if len(cat_orders) == 0:
                    continue
                cat_ticket = cat_orders['total_amount_usd'].mean()
                cat_disc_rate = (cat_orders['discount_pct'] > 0).mean()
                cat_n_customers = cat_orders['customer_id'].nunique()
                cat_projected = cat_n_customers * cat_ticket * (1 - discount_level/100) * volume_lift
                cat_results.append({
                    'Categoría': cat,
                    'Clientes Activos': cat_n_customers,
                    'Ticket Promedio': cat_ticket,
                    'Tasa Descuento Hist.': cat_disc_rate,
                    'Revenue Proyectado': cat_projected,
                })

            if cat_results:
                df_cat_res = pd.DataFrame(cat_results)
                fig_cat_bar = px.bar(
                    df_cat_res, x='Categoría', y='Revenue Proyectado',
                    color='Tasa Descuento Hist.',
                    color_continuous_scale='YlOrRd',
                    text_auto='$.2s',
                    title='<b>Revenue Proyectado por Categoría</b>',
                )
                fig_cat_bar.update_layout(
                    template='plotly_dark', height=400,
                    font=dict(family='Inter'),
                )
                st.plotly_chart(fig_cat_bar, use_container_width=True)

                st.dataframe(
                    df_cat_res.style.format({
                        'Ticket Promedio': '${:,.2f}',
                        'Tasa Descuento Hist.': '{:.0%}',
                        'Revenue Proyectado': '${:,.0f}',
                    }),
                    use_container_width=True
                )

            # ── Waterfall chart ──
            st.markdown("####  Cascada de Impacto Financiero")
            fig_waterfall = go.Figure(go.Waterfall(
                name="Impacto",
                orientation="v",
                measure=["absolute", "relative", "relative", "relative", "total"],
                x=["Revenue Bruto", "Descuento", "Presupuesto Campaña",
                   "Lift Volumen", "Resultado Neto"],
                y=[gross_revenue, -discount_cost, -campaign_budget,
                   projected_revenue * (volume_lift - 1) / volume_lift,
                   None],
                text=[f"${gross_revenue:,.0f}", f"-${discount_cost:,.0f}",
                      f"-${campaign_budget:,.0f}",
                      f"+${projected_revenue * (volume_lift-1)/volume_lift:,.0f}",
                      f"${projected_revenue - total_investment:,.0f}"],
                textposition="outside",
                connector=dict(line=dict(color="#475569")),
                increasing=dict(marker=dict(color="#10b981")),
                decreasing=dict(marker=dict(color="#ef4444")),
                totals=dict(marker=dict(color="#6366f1")),
            ))
            fig_waterfall.update_layout(
                template='plotly_dark', height=420,
                font=dict(family='Inter'),
                title='<b>Cascada de Impacto Financiero de la Campaña</b>',
                showlegend=False,
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)

        elif run_sim:
            st.warning("Selecciona al menos un segmento y una categoría para simular.")
        else:
            st.info(" Configura los parámetros de la campaña y haz clic en **Ejecutar Simulación**")

            # Mostrar preview de datos
            st.markdown("#### Vista previa: Receptividad Histórica por Segmento")
            prev_data = df_seg.groupby('segment_name').agg({
                'avg_discount_pct': 'mean',
                'pct_orders_with_discount': 'mean',
                'monetary': 'mean',
                'frequency': 'mean',
            }).round(2).reset_index()
            prev_data.columns = ['Segmento', 'Desc. Promedio (%)', '% con Descuento',
                                 'Gasto Promedio', 'Frecuencia']

            fig_preview = px.bar(
                prev_data, x='Segmento', y='% con Descuento',
                color='Gasto Promedio',
                color_continuous_scale='Viridis',
                text_auto='.0%',
                title='<b>Tasa de Uso de Descuentos por Segmento</b>',
            )
            fig_preview.update_layout(
                template='plotly_dark', height=400,
                font=dict(family='Inter'),
            )
            st.plotly_chart(fig_preview, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 3: ANÁLISIS POR CATEGORÍA
# ═══════════════════════════════════════════════
with tab_categories:
    if labels is None:
        st.error("Modelo no disponible."); st.stop()

    st.markdown("#### Receptividad a Descuentos: Segmento × Categoría")

    # Cruzar segmentos con categorías de órdenes
    df_orders_seg = df_orders.merge(
        df_seg[['customer_id', 'segment', 'segment_name']],
        on='customer_id', how='inner'
    )

    # Heatmap: tasa de descuento por segmento × categoría
    heat_data = df_orders_seg.groupby(['segment_name', 'category']).agg(
        avg_disc=('discount_pct', 'mean'),
        n_orders=('order_id', 'count'),
        pct_with_disc=('discount_pct', lambda x: (x > 0).mean()),
        avg_revenue=('total_amount_usd', 'mean'),
    ).reset_index()

    # Pivot para heatmap
    pivot_disc = heat_data.pivot_table(
        index='segment_name', columns='category', values='pct_with_disc'
    ).fillna(0)

    fig_heat = px.imshow(
        pivot_disc.values,
        x=pivot_disc.columns.tolist(),
        y=pivot_disc.index.tolist(),
        color_continuous_scale='YlOrRd',
        title='<b>% Pedidos con Descuento: Segmento × Categoría</b>',
        labels=dict(color='% con Desc.'),
        text_auto='.0%',
        aspect='auto',
    )
    fig_heat.update_layout(
        template='plotly_dark', height=400,
        font=dict(family='Inter'),
        xaxis=dict(tickangle=-45),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ── Selector de categoría para deep-dive ──
    sel_cat = st.selectbox("Selecciona una categoría para análisis detallado",
                           CATEGORY_LIST)

    cat_orders_sel = df_orders_seg[df_orders_seg['category'] == sel_cat]

    if len(cat_orders_sel) > 0:
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            # Revenue por segmento en esta categoría
            cat_seg_rev = cat_orders_sel.groupby('segment_name').agg(
                total_rev=('total_amount_usd', 'sum'),
                avg_rev=('total_amount_usd', 'mean'),
                n_orders=('order_id', 'count'),
            ).reset_index()

            fig_cat_seg = px.bar(
                cat_seg_rev, x='segment_name', y='total_rev',
                color='segment_name',
                color_discrete_sequence=list(SEGMENT_COLORS.values()),
                text_auto='$.2s',
                title=f'<b>Revenue Total en {sel_cat} por Segmento</b>',
                labels={'segment_name': 'Segmento', 'total_rev': 'Revenue Total'},
            )
            fig_cat_seg.update_layout(
                template='plotly_dark', height=400,
                font=dict(family='Inter'), showlegend=False,
            )
            st.plotly_chart(fig_cat_seg, use_container_width=True)

        with col_c2:
            # Distribución de descuentos por segmento
            fig_box = px.box(
                cat_orders_sel[cat_orders_sel['discount_pct'] > 0],
                x='segment_name', y='discount_pct',
                color='segment_name',
                color_discrete_sequence=list(SEGMENT_COLORS.values()),
                title=f'<b>Distribución de Descuentos en {sel_cat}</b>',
                labels={'segment_name': 'Segmento', 'discount_pct': 'Descuento (%)'},
            )
            fig_box.update_layout(
                template='plotly_dark', height=400,
                font=dict(family='Inter'), showlegend=False,
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # ── Tabla resumen ──
        cat_summary = cat_orders_sel.groupby('segment_name').agg(
            pedidos=('order_id', 'count'),
            clientes=('customer_id', 'nunique'),
            ticket_prom=('total_amount_usd', 'mean'),
            desc_prom=('discount_pct', 'mean'),
            tasa_desc=('discount_pct', lambda x: (x > 0).mean()),
            tasa_devol=('returned', 'mean'),
        ).round(3).reset_index()
        cat_summary.columns = ['Segmento', 'Pedidos', 'Clientes Únicos',
                               'Ticket Promedio', 'Desc. Promedio (%)',
                               'Tasa con Descuento', 'Tasa Devolución']

        st.dataframe(
            cat_summary.style.format({
                'Ticket Promedio': '${:,.2f}',
                'Desc. Promedio (%)': '{:.1f}%',
                'Tasa con Descuento': '{:.0%}',
                'Tasa Devolución': '{:.0%}',
            }).background_gradient(subset=['Tasa con Descuento'], cmap='YlOrRd'),
            use_container_width=True
        )
    else:
        st.info(f"No hay órdenes registradas para la categoría **{sel_cat}**.")


# ═══════════════════════════════════════════════
# TAB 4: MODELO LCA
# ═══════════════════════════════════════════════
with tab_model:
    if labels is None:
        st.error("Modelo no disponible."); st.stop()

    st.markdown("#### Selección del Número de Clases (BIC)")

    # Calcular BIC para rango de K
    with st.spinner("Calculando BIC para distintos K…"):
        lbl_full, post_full, bic_vals, best_k_bic, mm_data_full = fit_lca_model(df_seg)

    if bic_vals:
        bic_df = pd.DataFrame({
            'K': list(bic_vals.keys()),
            'BIC': list(bic_vals.values())
        })
        bic_df['Seleccionado'] = bic_df['K'] == n_classes

        fig_bic = px.line(
            bic_df, x='K', y='BIC', markers=True,
            title='<b>BIC por Número de Clases Latentes</b>',
            labels={'K': 'Número de Clases (K)', 'BIC': 'BIC'},
        )
        # Highlight selected K
        fig_bic.add_vline(x=n_classes, line_dash="dash",
                          line_color="#6366f1", opacity=0.7,
                          annotation_text=f"K={n_classes} (seleccionado)")
        # Highlight best K
        fig_bic.add_vline(x=best_k_bic, line_dash="dot",
                          line_color="#10b981", opacity=0.7,
                          annotation_text=f"K={best_k_bic} (mejor BIC)")
        fig_bic.update_layout(
            template='plotly_dark', height=400,
            font=dict(family='Inter'),
        )
        st.plotly_chart(fig_bic, use_container_width=True)

    # ── Métricas del modelo ──
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("K Seleccionado", n_classes)
    col_m2.metric("Entropía Relativa", f"{rel_entropy:.4f}",
                  delta="Bueno " if rel_entropy > 0.8 else "Mejorable ",
                  delta_color="normal" if rel_entropy > 0.8 else "inverse")
    min_class_pct = df_seg['segment'].value_counts(normalize=True).min()
    col_m3.metric("Clase más pequeña",
                  f"{min_class_pct:.1%}",
                  delta="OK " if min_class_pct > 0.05 else "Muy pequeña ",
                  delta_color="normal" if min_class_pct > 0.05 else "inverse")

    # ── Probabilidades posteriores ──
    st.markdown("---")
    st.subheader("Distribución de Probabilidades Posteriores")
    post_df = pd.DataFrame(posteriors,
                           columns=[f'P(Clase {i})' for i in range(n_classes)])
    post_df['max_prob'] = post_df.max(axis=1)

    fig_post = px.histogram(
        post_df, x='max_prob', nbins=50,
        title='<b>Certeza de Clasificación (máx probabilidad posterior)</b>',
        labels={'max_prob': 'Máxima Probabilidad Posterior', 'count': 'Frecuencia'},
        color_discrete_sequence=['#6366f1'],
    )
    fig_post.update_layout(
        template='plotly_dark', height=350,
        font=dict(family='Inter'),
    )
    st.plotly_chart(fig_post, use_container_width=True)

    with st.expander(" Fórmulas del Modelo"):
        st.markdown("#### Modelo LCA con datos mixtos")
        st.latex(r"""
        P(\mathbf{x}_i) = \sum_{c=1}^{K} \pi_c \prod_{j \in \text{cont}} 
        \phi(x_{ij}; \mu_{jc}, \sigma^2_{jc}) 
        \prod_{j \in \text{cat}} \prod_{r=1}^{R_j} \theta_{jrc}^{I(x_{ij}=r)}
        \prod_{j \in \text{bin}} p_{jc}^{x_{ij}}(1-p_{jc})^{1-x_{ij}}
        """)
        st.markdown("""
        Donde:
        - $\\pi_c$ : probabilidad a priori de la clase $c$
        - $\\phi$ : densidad normal para indicadores continuos
        - $\\theta_{jrc}$ : probabilidad de categoría $r$ en la variable $j$ para la clase $c$
        - $p_{jc}$ : probabilidad del indicador binario $j$ en la clase $c$
        - **BIC** = $-2 \\ln(L) + k \\ln(n)$, donde $k$ son los parámetros y $n$ las observaciones
        """)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.caption("Trabajo II – Marketing – DII UdeC 2026")
with col_f2:
    st.markdown(
        "<div style='text-align:right; color:#64748b; font-size:.8em;'>"
        "Grupo 29."
        "</div>",
        unsafe_allow_html=True
    )
