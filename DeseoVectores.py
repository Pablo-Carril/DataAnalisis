import streamlit as st
import pandas as pd
import pydeck as pdk
import folium
from streamlit_folium import st_folium

# Configuración de página
st.set_page_config(page_title="Análisis de Flujos de Transporte", layout="wide")

st.title("📊 Mapa de Flujos de Transacciones")
st.markdown("""
Esta herramienta estima el destino de los pasajeros buscando su próximo viaje en sentido contrario el mismo día.
""")

# --- 1. CARGA DE DATOS ---
@st.cache_data
def cargar_datos(archivo):
    # Cargar Parquet
    df = pd.read_parquet(archivo)
    
    # Asegurar tipos de datos
    df['Fecha Hora'] = pd.to_datetime(df['Fecha Hora'])
    if 'Fecha' in df.columns:
        df = df.drop(columns=['Fecha'])  #elimina la columa Fecha original. sólo usaré Fecha Hora.
    df['Fecha'] = df['Fecha Hora'].dt.date
    
    # Limpieza básica: quitar filas sin coordenadas
    df = df.dropna(subset=['Latitud', 'Longitud'])
    return df

# --- 2. LÓGICA DE PROCESAMIENTO (ORIGEN-DESTINO) ---
@st.cache_data
def calcular_vectores_flujo(df):
    # Ordenar cronológicamente por tarjeta
    df = df.sort_values(['Tarjeta', 'Fecha Hora'])
    
    # INVERTIMOS Latitud y Longitud porque en el CSV/Parquet vienen invertidos.
    df = df.rename(columns={
        'Latitud': 'Long_Original', # Latitud es Longitud y
        'Longitud': 'Lat_Original'  # Longitud es Latitud
    })
    df = df.rename(columns={        # volvemos a los nombres
        'Long_Original': 'Longitud', 
        'Lat_Original': 'Latitud'
    })
    
    # Creamos columnas desplazadas (shift) dentro de cada grupo de tarjeta
    # Esto busca el 'Siguiente Evento' de esa misma persona
    df_sorted = df.groupby('Tarjeta').apply(lambda x: x.assign(
        Lat_Destino=x['Latitud'].shift(-1),
        Lon_Destino=x['Longitud'].shift(-1),
        Sentido_Siguiente=x['Sentido'].shift(-1),
        Fecha_Siguiente=x['Fecha'].shift(-1)
    )).reset_index(drop=True)

    # CRITERIOS PARA VALIDAR UN VECTOR:
    # 1. El sentido debe haber cambiado (Ida -> Vuelta o Vuelta -> Ida)
    # 2. Debe ser el mismo día
    # 3. No deben ser nulos los destinos
    mask = (
        (df_sorted['Sentido'] != df_sorted['Sentido_Siguiente']) & 
        (df_sorted['Fecha'] == df_sorted['Fecha_Siguiente']) &
        (df_sorted['Lat_Destino'].notna())
    )
    
    return df_sorted[mask]

# --- 3. INTERFAZ DE USUARIO ---
archivo_subido = st.sidebar.file_uploader("Cargar archivo Parquet", type=["parquet"])

if archivo_subido:
    df_raw = cargar_datos(archivo_subido)
    
    # Filtros en el Sidebar
    st.sidebar.header("Filtros")
    
    # Filtro de Fecha
    fechas = sorted(df_raw['Fecha'].unique())
    fecha_sel = st.sidebar.selectbox("Seleccionar Fecha", fechas)
    
    # Filtro de Ramal
    ramales = ["Todos"] + sorted(df_raw['Ramal'].unique().tolist())
    ramal_sel = st.sidebar.selectbox("Seleccionar Ramal", ramales)
    
    # Aplicar filtros antes de procesar flujos
    df_filtrado = df_raw[df_raw['Fecha'] == fecha_sel]
    if ramal_sel != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Ramal'] == ramal_sel]
    
    # Procesar vectores
    with st.spinner('Calculando vectores de origen-destino...'):
        df_flujos = calcular_vectores_flujo(df_filtrado)

    if not df_flujos.empty:
        # Métricas rápidas (sin cambios)
        col1, col2, col3 = st.columns(3)
        col1.metric("Transacciones totales", len(df_filtrado))
        col2.metric("Vectores detectados", len(df_flujos))
        col3.metric("Ramal actual", ramal_sel)

        # --- LIMPIEZA AGRESIVA PARA EVITAR TYPEERROR ---
        # Convertimos a lista de diccionarios. Esto es lo más seguro para Pydeck
        # porque elimina cualquier dependencia de índices o tipos complejos de Pandas.
        data_arcos = df_flujos.reset_index(drop=True).copy()
        for c in data_arcos.columns:
            if data_arcos[c].dtype == 'datetime64[ns]' or data_arcos[c].dtype == 'object':
                data_arcos[c] = data_arcos[c].astype(str)
        dict_arcos = data_arcos.to_dict(orient='records')

        data_puntos = df_filtrado.reset_index(drop=True).copy()
        for c in data_puntos.columns:
            if data_puntos[c].dtype == 'datetime64[ns]' or data_puntos[c].dtype == 'object':
                data_puntos[c] = data_puntos[c].astype(str)
        dict_puntos = data_puntos.to_dict(orient='records')
        # -----------------------------------------------

        # --- 4. MAPA DE PYDECK ---
        layer_arcos = pdk.Layer(
            "ArcLayer",
            dict_arcos, # Usamos la lista de diccionarios
            get_source_position=["Longitud", "Latitud"],
            get_target_position=["Lon_Destino", "Lat_Destino"],
            get_source_color=[240, 100, 0, 180],
            get_target_color=[0, 200, 255, 180],
            get_width=2.5,
            pickable=True,
        )

        layer_puntos = pdk.Layer(
            "ScatterplotLayer",
            dict_puntos, # Usamos la lista de diccionarios
            get_position=["Longitud", "Latitud"],
            get_color=[255, 255, 255, 20],
            get_radius=30,
        )

        view_state = pdk.ViewState(
            latitude=df_flujos["Latitud"].mean(),
            longitude=df_flujos["Longitud"].mean(),
            zoom=13,
            pitch=45,
        )

        # Usamos una estructura más simple para el Deck
        Config_pydeck = pdk.Deck(
            map_provider="Mapbox",
            map_style='light',
            initial_view_state=view_state,
            layers=[ layer_arcos],
            tooltip={
                "html": "<b>Tarjeta:</b> {Tarjeta}<br/>"
                        "<b>Subida:</b> {Sentido}<br/>"
                        "<b>Próximo Viaje:</b> {Sentido_Siguiente}",
                "style": {"color": "white"}
            }
        )
        
        st.subheader("Primero:")
        st.pydeck_chart(Config_pydeck)

        st.subheader("Segundo:")
        st.pydeck_chart(pdk.Deck(
            map_provider="carto", # Cambiamos de Mapbox a Carto (Gratis)
            map_style="light",    # O "dark"
            initial_view_state=view_state,
            layers=[layer_arcos]  # Prueba SIN la capa de puntos primero
            ))
        

        # FOLIUM:
           # if not df_flujos.empty:
           # st.subheader("Visualización de Flujos (Calles Reales)")
            
           # # 1. Preparar el mapa base
           # centro_lat = df_flujos["Latitud"].mean()
          #  centro_lon = df_flujos["Longitud"].mean()
            
           # # Creamos el mapa con OpenStreetMap (Infalible)
           # m = folium.Map(location=[centro_lat, centro_lon], zoom_start=13, tiles="OpenStreetMap")

           # # 2. Agregar los vectores
           # # Usamos los datos ya procesados en dict_arcos
           #     # Puntos
           #     punto_subida = [flow["Latitud"], flow["Longitud"]]
           #     punto_bajada = [flow["Lat_Destino"], flow["Lon_Destino"]]
           #     
          # #     # Línea de flujo
           #     folium.PolyLine(
           #         locations=[punto_subida, punto_bajada],
           #         color="#FF4500", # Naranja fuerte
           #         weight=2,
           #         opacity=0.5,
           #         tooltip=f"Tarjeta: {flow['Tarjeta']} | {flow['Sentido']} -> {flow['Sentido_Siguiente']}"
           #     ).add_to(m)
#
           #     # Marcador circular en la subida para dar contexto
           #     folium.CircleMarker(
           #         location=punto_subida,
           #         radius=2,
           #         color="blue",
           #         fill=True,
           #         opacity=0.4
           #     ).add_to(m)

           # # 3. Renderizar el mapa en Streamlit
           # st_folium(m, width=1000, height=600)

        # OTRA PRUEBA
        if not df_flujos.empty:
            st.subheader("Mapa Nativo de Streamlit")
            
            # 1. Ajustamos nombres para que st.map no proteste
            df_map_native = df_flujos[['Latitud', 'Longitud']].copy()
            df_map_native.columns = ['lat', 'lon'] # Renombrado rápido
            
            # st.map es lo más básico, debería forzar la carga de la capa base
            st.map(df_map_native)
            
            # 2. Intento con Folium usando un servidor de mapas alternativo
           # st.subheader("Mapa con Folium (Servidor Stamen/Carto)")
           # 
           # centro_lat = df_flujos["Latitud"].mean()
           # centro_lon = df_flujos["Longitud"].mean()
           # 
           # # Usamos 'cartodbpositron' que es muy liviano y rara vez falla
           # m = folium.Map(location=[centro_lat, centro_lon], zoom_start=12, tiles='cartodbpositron')

           # for flow in dict_arcos:
           #     folium.PolyLine(
           #         locations=[[flow["Latitud"], flow["Longitud"]], [flow["Lat_Destino"], flow["Lon_Destino"]]],
           #         color="red",
           #         weight=2,
           #         opacity=0.4
           #     ).add_to(m)

           # st_folium(m, width=1000, height=500)



        # Opcional: Mostrar tabla de datos
        if st.checkbox("Ver datos procesados"):
            st.dataframe(df_flujos[['Fecha Hora', 'Tarjeta', 'Latitud', 'Longitud', 'Lat_Destino', 'Lon_Destino', 'Sentido']])
            
    else:
        st.warning("No se encontraron viajes de ida y vuelta para los mismos pasajeros en los filtros seleccionados.")
else:
    st.info("Por favor, carga un archivo .parquet en la barra lateral para comenzar.")