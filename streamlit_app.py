# streamlit_app.py
import streamlit as st
import requests

# --- Configuración General ---
st.set_page_config(
    page_title="Streamlit + FastAPI Demo",
    layout="wide"
)

# FastAPI se ejecutará en el puerto 8000 por defecto
API_URL_BASE = "http://localhost:8000/api/v1/item/"

st.title("🐍 Streamlit + FastAPI: Aplicación Full-Stack")
st.markdown(
    """
    Esta es una demo de Streamlit que hace llamadas a una API RESTful 
    ejecutada en **FastAPI** para obtener datos.
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

if st.sidebar.button("Buscar en Backend (FastAPI)"):
    try:
        with st.spinner(f"Buscando ítem #{item_id} en FastAPI..."):
            # 1. Hacemos la llamada HTTP a la API
            response = requests.get(f"{API_URL_BASE}{item_id}")
        
        # 2. Verificamos el estado de la respuesta
        if response.status_code == 200:
            data = response.json()
            st.success(f"✅ ¡Datos recibidos de FastAPI para el ítem {data['id']}!")
            
            # 3. Mostramos los datos
            col1, col2 = st.columns(2)
            
            with col1:
                st.header(data['nombre'])
                st.info(f"ID: {data['id']}")
                st.markdown(f"**Descripción:** {data['descripcion']}")
                
            with col2:
                st.subheader("Respuesta JSON Cruda")
                st.json(data)
                
        else:
            st.error(f"❌ Error al conectar o recibir datos de la API. Código de estado: {response.status_code}")
            st.warning("Asegúrate de que el servidor FastAPI esté corriendo en http://localhost:8000.")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Error de Conexión: No se pudo conectar al servidor FastAPI. Asegúrate de que esté ejecutándose.")
        
st.markdown("---")
st.caption("Recuerda: FastAPI corre en 8000, Streamlit en 8501.")