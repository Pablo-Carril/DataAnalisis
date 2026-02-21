# streamlit_app.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Configuración General ---
st.set_page_config(
    page_title="DataAnalisis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="💻"
)

# Inyectar CSS para ocultar el menú 
hide_menu_style = """
        <style>
       /* #MainMenu {visibility: hidden;} */
       /* header {visibility: hidden;} */
        footer {visibility: hidden;}
        /* Quitar el espacio superior del contenedor principal */
            .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
                padding-left: 3rem;
                padding-right: 3rem;
            }
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# FastAPI se ejecutará en el puerto 8000 por defecto
API_URL_BASE = "http://localhost:8000/api/"  #"http://localhost:8000/api/item/"
#API_URL_DATA = "http://localhost:8000/api/dashboard/data"

st.title("Dashboard - Informe Mensual")
st.markdown(
    """
    Streamlit y FastAPI  -  Línea: 
    """
)
st.sidebar.header("Configuración de Búsqueda")

log_footer = st.sidebar.empty()
log_footer.info("Listo")


# --- Lógica de la Aplicación ---
item_id = st.sidebar.number_input(
    "Ingresa el ID del ítem a buscar:", 
    min_value=1, 
    value=1, 
    step=1
)




if st.sidebar.button("Ventas totales"):
    try:
        with st.spinner(f"Buscando..."):
            # 1. Hacemos la llamada HTTP a la API
            response = requests.get(f"{API_URL_BASE}item/{item_id}")
        
        # 2. Verificamos el estado de la respuesta
        if response.status_code == 200:
            data = response.json()
            log_footer.info(data)
            
            # 3. Mostramos los datos
            col1, col2 = st.columns(2)
            
            with col1:
                #st.header(data['nombre'])
                #st.info(f"ID: {data['id']}")
                #st.markdown(f"**Descripción:** {data['descripcion']}")
                #st.title("💸 Total de Ventas Calculadas por FastAPI")

                st.metric(
                label=f"Suma Total de Ventas", # (Top {num_rows} Filas)",
                value=f"${data['valor']:,.2f}" # Formato de moneda
                )
                
            with col2:
                st.subheader("Respuesta JSON:")
                st.json(data)
                
        else:
            log_footer.error(f"❌ Error al conectar o recibir datos de la API. Código de estado: {response.status_code}")
            log_footer.warning("Asegúrate de que el servidor FastAPI esté corriendo en http://localhost:8000.")
            
    except requests.exceptions.ConnectionError:
        log_footer.error("❌ Error de Conexión: No se pudo conectar al servidor FastAPI. Asegúrate de que esté ejecutándose.")


#decorador, se coloca antes de una función:
@st.cache_data(ttl=60)  # Caching para evitar recargar la API con cada interacción, en segundos.
def fetch_df():
    """Función para obtener DataFrames de la API de FastAPI."""
    try:
        response = requests.get(f"{API_URL_BASE}dashboard/data")
        
        if response.status_code == 200:
            data_list = response.json()
            # Convertimos la lista de diccionarios (JSON) a un DataFrame de Pandas
            #df = pd.DataFrame(data_list)
            #return df
            datos = pd.DataFrame(data_list.items(), columns=['Ramal', 'Recuento de Tarifas']) # Convierte claves y valores a filas
            return datos
        
        else:
            log_footer.error(f"Error al obtener datos. Código: {response.status_code}")
            return pd.DataFrame() # Devuelve un DF vacío en caso de error
            
    except requests.exceptions.ConnectionError:
        log_footer.error("❌ Error de Conexión: Asegúrate de que el servidor FastAPI esté corriendo")
        return pd.DataFrame()

# --- Pasajeros por Ramal ---
if st.sidebar.button("Pasajeros por Ramal"):
    df_ventas = fetch_df()
    if not df_ventas.empty:
        log_footer.success(f"✅ Datos cargados correctamente. Se recibieron {len(df_ventas)} filas.")
        
        # 1. Mostrar la tabla de datos
        st.subheader("Tabla Pasajeros por Ramal")
        st.dataframe(df_ventas, use_container_width=True)
        
        # 2. Visualización - gráfico de Streamlit integrado bar_chart
        st.subheader("Gráficos Pasajeros por Ramal")
        # Agrupamos los datos localmente en Streamlit para el gráfico
        st.bar_chart(df_ventas, x='Ramal', y='Recuento de Tarifas')

        #probando plotly:
        st.subheader("Mediante Plotly:")
        fig = px.bar(
            df_ventas, 
            x='Ramal', 
            y='Recuento de Tarifas',
            title='Pasajeros por Ramal', # Título del gráfico Plotly
            color='Ramal', # Opcional: colorea las barras según el ramal
            labels={'Recuento de Trx': 'Trx Contadas'} # Etiquetas más limpias
        )
        # Opcional: Mejorar la interactividad y el diseño
        fig.update_layout(xaxis_title="Ramal", yaxis_title="Pasajeros")
        fig.update_traces(marker_line_width=1.5, marker_line_color='black') # Añadir bordes a las barras
        
        st.plotly_chart(fig, use_container_width=True)
        
        
    else:
        log_footer.error("Esperando que el servidor FastAPI esté disponible para cargar los datos.")



