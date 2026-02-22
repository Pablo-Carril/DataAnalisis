import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="DataAnalisis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# --- 2. FUNCIONES DE LOG (FOOTER DERECHA) ---
def update_footer_log(message, type="info"):
    colors = {"info": "#007bff", "success": "#28a745", "error": "#dc3545", "warning": "#ffc107"}
    color = colors.get(type, "gray")
    ahora = datetime.now().strftime("%H:%M:%S")
    
    st.markdown(f"""
        <style>
            .footer {{
                position: fixed; left: 0; bottom: 0; width: 100%;
                background-color: rgba(0,0,0,0); color: {color};
                text-align: right; padding-right: 30px; padding-bottom: 10px;
                font-size: 0.85rem; z-index: 100; font-weight: bold;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 1);
            }}
            .block-container {{ padding-top: 2rem; padding-bottom: 5rem; }}
            [data-testid="stMetricValue"] {{ font-size: 1.8rem; color: #007bff; }}
        </style>
        <div class="footer"> [{ahora}] Estado: {message}</div>
    """, unsafe_allow_html=True)

# --- 3. LÓGICA DE DATOS (Simulando el Backend) ---
@st.cache_data(ttl=3600) # Caché de 1 hora para eficiencia en la nube
def load_data():
    """
    Aquí es donde cargarías tu archivo consolidado.
    Si usas Parquet, es extremadamente rápido.
    """
    path_archivo = os.path.join(os.path.dirname(__file__), "./api/Transacciones 506 julio.csv")
    
    if not os.path.exists(path_archivo):
        # Datos de prueba si el archivo no existe
        update_footer_log("Archivo no encontrado.", "warning")
       # df_demo = pd.DataFrame({
         #   'Ramal': ['Ramal A', 'Ramal B', 'Ramal C', 'Ramal D'],
         #   'Pasajeros': [4500, 3200, 5800, 2100],
         #   'Valor_Venta': [15000, 12000, 19000, 8500]
        #})
        #return df_demo
    
    try:
        df = pd.read_csv(path_archivo, delimiter=';', decimal=',', encoding='utf-8', parse_dates=['Fecha Hora'], dayfirst=True)
        # si es Parquet:  df = pd.read_parquet(path_archivo)
        update_footer_log("Base de datos sincronizada", "success")
        return df
    except Exception as e:
        update_footer_log(f"Error al leer datos: {e}", "error")
        return pd.DataFrame()

# --- 4. PROCESAMIENTO DE "MEDIDAS" ---
# Replicamos la lógica que antes hacía FastAPI
def get_metrics(df, ramales_seleccionados=None):
    df_filtered = df.copy()
    if ramales_seleccionados:
        df_filtered = df_filtered[df_filtered['Ramal'].isin(ramales_seleccionados)]
    
    total_ventas = df_filtered['Tarifa Cobrada'].sum() if 'Tarifa Cobrada' in df_filtered else 0
    total_pax = df_filtered['Tarifa Cobrada'].count() if 'Tarifa Cobrada' in df_filtered else 0
    
    return df_filtered, total_ventas, total_pax

# --- 5. INTERFAZ DE USUARIO ---
st.title("📊 Dashboard DataAnalisis")

# Sidebar
with st.sidebar:
    st.header("Filtros")
    df_raw = load_data()
    
    if not df_raw.empty:
        lista_ramales = df_raw['Ramal'].unique().tolist()
        seleccion = st.multiselect("Seleccionar Ramales:", lista_ramales, default=lista_ramales)
    else:
        seleccion = []

# Procesar datos según selección
df_filtrado, ventas, pasajeros = get_metrics(df_raw, seleccion)

# Organización por Pestañas
tab1, tab2, tab3 = st.tabs(["📌 Resumen", "📈 Gráficos", "📋 Tabla"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Ventas Totales", f"$ {ventas:,.2f}")
    c2.metric("Pasajeros Totales", f"{pasajeros:,}")
    c3.metric("Ramales Activos", len(seleccion))
    
    st.divider()
    st.subheader("Tendencia Rápida")
    df_agrupado = df_filtrado.groupby('Ramal', as_index=False)['Tarifa Cobrada'].sum()
    st.bar_chart(df_agrupado.set_index('Ramal')['Tarifa Cobrada'])

with tab2:
    if not df_filtrado.empty:
        df_agrupado = df_filtrado.groupby('Ramal', as_index=False)['Tarifa Cobrada'].sum()
        fig = px.bar(df_agrupado, x='Ramal', y='Tarifa Cobrada', color='Tarifa Cobrada',
                     color_continuous_scale='Viridis', title="Ventas por Ramal")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.dataframe(df_filtrado, use_container_width=True)
    # Botón de exportación directo
    csv = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar Datos Crudos (CSV)", csv, "data.csv", "text/csv")

# Inicialización Footer
if 'init' not in st.session_state:
    update_footer_log("Iniciado en modo 'Solo Streamlit'", "info")
    st.session_state['init'] = True