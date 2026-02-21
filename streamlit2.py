import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="DataAnalisis Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# --- 2. FUNCIONES DE LOG (FOOTER ALINEADO A LA DERECHA) ---
def update_footer_log(message, type="info"):
    colors = {
        "info": "#007bff",      # Azul
        "success": "#28a745",   # Verde
        "error": "#dc3545",     # Rojo
        "warning": "#ffc107"    # Amarillo/Naranja
    }
    color = colors.get(type, "gray")
    ahora = datetime.now().strftime("%H:%M:%S")
    
    st.markdown(f"""
        <style>
            .footer {{
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: rgba(0,0,0,0); /* Fondo transparente */
                color: {color};
                text-align: right;                /* Alineación a la derecha */
                padding-right: 30px;              /* Espacio desde el borde derecho */
                padding-bottom: 10px;
                font-size: 0.85rem;
                z-index: 100;
                font-weight: bold;
                /* Sombra negra para máxima legibilidad */
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 1); 
            }}
        </style>
        <div class="footer">
            📡 [{ahora}] Estado: {message}
        </div>
    """, unsafe_allow_html=True)

# --- 3. AJUSTE DE MARGEN INFERIOR ---
st.markdown("""
    <style>
        /* Aumentamos el padding inferior para que el contenido no choque con el footer */
        .block-container { 
            padding-top: 1rem; 
            padding-bottom: 4rem; 
        }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ESTILOS CSS ---
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 5rem; }
        [data-testid="stMetricValue"] { font-size: 1.8rem; color: #007bff; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. LÓGICA DE DATOS ---
API_URL_BASE = "http://localhost:8000/api/"

@st.cache_data(ttl=60)
def fetch_dashboard_data():
    try:
        response = requests.get(f"{API_URL_BASE}dashboard/data")
        if response.status_code == 200:
            update_footer_log("Sincronización completa", "success")
            return pd.DataFrame(response.json().items(), columns=['Ramal', 'Pasajeros'])
    except:
        update_footer_log("Error de enlace con FastAPI", "error")
    return pd.DataFrame()

# --- 5. CUERPO PRINCIPAL ---
st.title("📊 Dashboard de Control de Tráfico")

# Sidebar
with st.sidebar:
    st.title("Filtros")
    item_id = st.number_input("ID de Referencia:", min_value=1, value=1)

# Obtener datos globales
df_ventas = fetch_dashboard_data()

tab_resumen, tab_graficos, tab_raw = st.tabs(["📌 Resumen General", "📈 Análisis Visual", "📋 Datos Crudos"])

# --- TAB 1: RESUMEN GENERAL (Gráfico Restaurado) ---
with tab_resumen:
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    # Simulación de valor por ID
    try:
        res_item = requests.get(f"{API_URL_BASE}item/{item_id}")
        val_ventas = res_item.json()['valor'] if res_item.status_code == 200 else 0
    except: val_ventas = 0

    col_kpi1.metric("Ventas Totales", f"$ {val_ventas:,.2f}")
    col_kpi2.metric("Total Pasajeros", f"{df_ventas['Pasajeros'].sum():,}" if not df_ventas.empty else "0")
    col_kpi3.metric("Eficiencia", "88%")
    col_kpi4.metric("Ramales", len(df_ventas))

    st.divider()
    
    if not df_ventas.empty:
        st.subheader("Vista Rápida de Operaciones")
        # Restauramos el gráfico que faltaba aquí
        st.bar_chart(df_ventas.set_index('Ramal'), use_container_width=True)

# --- TAB 2: ANÁLISIS VISUAL ---
with tab_graficos:
    if not df_ventas.empty:
        fig = px.bar(df_ventas, x='Ramal', y='Pasajeros', color='Pasajeros', 
                     color_continuous_scale='Viridis', title='Tráfico por Ramal (Plotly)')
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: DATOS CRUDOS Y EXPORTACIÓN ---
with tab_raw:
    if not df_ventas.empty:
        st.subheader("Explorador de Datos")
        st.dataframe(df_ventas, use_container_width=True)
        
        st.divider()
        st.write("### 📥 Exportar Reporte")
        
        # Fila de botones de exportación
        c1, c2, c3, c4 = st.columns(4)
        
        # Exportar CSV (Nativo de Streamlit)
        csv = df_ventas.to_csv(index=False).encode('utf-8')
        c1.download_button("💾 CSV", data=csv, file_name="reporte.csv", mime="text/csv")
        
        # Nota sobre PDF/PNG/SVG:
        # Plotly permite descargar estos formatos directamente desde el icono de cámara en el gráfico.
        # Pero si los quieres como botones, aquí tienes la lógica base:
        c2.button("🖼️ PNG", on_click=lambda: update_footer_log("PNG generado (Simulado)", "info"))
        c3.button("🎨 SVG", on_click=lambda: update_footer_log("SVG generado (Simulado)", "info"))
        c4.button("📄 PDF", on_click=lambda: update_footer_log("Iniciando impresión PDF...", "warning"))
    else:
        st.error("No hay datos para mostrar.")

# Inicialización Footer
if 'init' not in st.session_state:
    update_footer_log("Sistema Listo", "info")
    st.session_state['init'] = True