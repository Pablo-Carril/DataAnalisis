# streamlit_app.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Configuración General ---
st.set_page_config(
    page_title="Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💻"
)

# FastAPI se ejecutará en el puerto 8000 por defecto
API_URL_BASE = "http://localhost:8000/api/dashboard/data"  #"http://localhost:8000/api/item/"
#API_URL_DATA = "http://localhost:8000/api/dashboard/data"

st.title("Dashboard")
st.markdown(
    """
    Probando Streamlit con llamadas a una API RESTful
    """
)

st.sidebar.header("Configuración de Búsqueda")
# --- Lógica de la Aplicación ---
item_id = st.sidebar.number_input(
    "Ingresa el ID del ítem a buscar:", 
    min_value=1, 
    value=1, 
    step=1
)

log_footer = st.sidebar.empty()
log_footer.info("Listo")

if st.sidebar.button("Buscar en Backend (FastAPI)"):
    try:
        with st.spinner(f"Buscando..."):
            # 1. Hacemos la llamada HTTP a la API
            response = requests.get(f"{API_URL_BASE}")   #{item_id}") lo saco por ahora
        
        # 2. Verificamos el estado de la respuesta
        if response.status_code == 200:
            data = response.json()
            log_footer.success(f"✅ ¡Datos recibidos de FastAPI para el ítem: ")  #{data['id']}!")
            
            # 3. Mostramos los datos
            col1, col2 = st.columns(2)
            
            with col1:
                #st.header(data['nombre'])
                #st.info(f"ID: {data['id']}")
                #st.markdown(f"**Descripción:** {data['descripcion']}")
                #st.title("💸 Total de Ventas Calculadas por FastAPI")

                # Muestra el resultado usando el formato st.metric
                st.metric(
                label=f"Suma Total de Ventas", # (Top {num_rows} Filas)",
                value=f"2"     #${total_ventas_obtenidas:,.2f}" # Formato de moneda
                )
                
            with col2:
                st.subheader("Respuesta JSON Cruda")
                st.json(data)
                
        else:
            log_footer.error(f"❌ Error al conectar o recibir datos de la API. Código de estado: {response.status_code}")
            log_footer.warning("Asegúrate de que el servidor FastAPI esté corriendo en http://localhost:8000.")
            
    except requests.exceptions.ConnectionError:
        log_footer.error("❌ Error de Conexión: No se pudo conectar al servidor FastAPI. Asegúrate de que esté ejecutándose.")
        
#st.markdown("---")
#st.caption("Recuerda: FastAPI corre en 8000, Streamlit en 8501.")


# Cargamos los datos desde el server
@st.cache_data(ttl=60)  # Caching para evitar recargar la API con cada interacción, en segundos.

def fetch_data_from_api():
    """Función para obtener datos de la API de FastAPI."""
    try:
        response = requests.get(API_URL_BASE)
        
        if response.status_code == 200:
            data_list = response.json()
            # Convertimos la lista de diccionarios (JSON) a un DataFrame de Pandas
            #df = pd.DataFrame(data_list)
            #return df
            df_recuento = pd.DataFrame(
                data_list.items(), # Convierte claves y valores a filas
                columns=['Ramal', 'Recuento de Tarifas']
            )
            return df_recuento
        else:
            log_footer.error(f"Error al obtener datos. Código: {response.status_code}")
            return pd.DataFrame() # Devuelve un DF vacío en caso de error
            
    except requests.exceptions.ConnectionError:
        log_footer.error("❌ Error de Conexión: Asegúrate de que el servidor FastAPI esté corriendo en http://localhost:8000.")
        return pd.DataFrame()

# --- Cargar y Mostrar Datos ---

df_ventas = fetch_data_from_api()

if not df_ventas.empty:
    log_footer.success(f"✅ Datos cargados correctamente. Se recibieron {len(df_ventas)} filas.")
    
    # 1. Mostrar la tabla de datos
    st.subheader("Datos Crudos del CSV (Desde FastAPI)")
    st.dataframe(df_ventas, use_container_width=True)
    
    # 2. Visualización y Análisis (Usando Pandas en Streamlit para el gráfico)
    #st.subheader("Ventas por Región")
    
    # Agrupamos los datos localmente en Streamlit para el gráfico
    #ventas_por_region = df_ventas.groupby('Region')['Ventas'].sum().reset_index()
    
    #st.bar_chart(ventas_por_region, x='Region', y='Ventas')
    
else:
    log_footer.info("Esperando que el servidor FastAPI esté disponible para cargar los datos.")