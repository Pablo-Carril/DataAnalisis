import streamlit as st
import pandas as pd
import pydeck as pdk

# Configuración de página
st.set_page_config(page_title="Análisis de Flujos de Transporte", layout="wide")

# --- 0. MEMORIA DE MAPA ---
if 'map_view' not in st.session_state:
    st.session_state.map_view = {
        "latitude": -34.921,
        "longitude": -57.954,
        "zoom": 12,
        "pitch": 45,
        "bearing": 0
    }

st.title("📊 Mapa de Flujos de Pasajeros")
st.markdown("""
Esta herramienta agrupa viajes cercanos para visualizar los **corredores de mayor demanda**. 
Mapa de Deseo 
""")

# --- 1. CARGA DE DATOS ---
@st.cache_data
def cargar_datos(archivo):
    df = pd.read_parquet(archivo)
    df['Fecha Hora'] = pd.to_datetime(df['Fecha Hora'])
    if 'Fecha' in df.columns:
        df = df.drop(columns=['Fecha'])
    df['Fecha'] = df['Fecha Hora'].dt.date
    df['Hora_Int'] = df['Fecha Hora'].dt.hour
    
    # --- LIMPIEZA DE COORDENADAS CERO (ÁFRICA) ---
    # Filtramos ceros tanto en Latitud como en Longitud
    df = df[(df['Latitud'] != 0) & (df['Longitud'] != 0)]
    
    df = df.dropna(subset=['Latitud', 'Longitud'])
    return df

# --- 2. LÓGICA DE PROCESAMIENTO (OPTIMIZADA) ---
@st.cache_data
def calcular_vectores_flujo(df):
    df = df.sort_values(['Tarjeta', 'Fecha Hora'])
    
    # Inversión de coordenadas (Lógica original)
    df = df.rename(columns={'Latitud': 'Long_Original', 'Longitud': 'Lat_Original'})
    df = df.rename(columns={'Long_Original': 'Longitud', 'Lat_Original': 'Latitud'})
    
    # Aseguramos coordenadas negativas para Argentina
    df['Latitud'] = df['Latitud'].apply(lambda x: -abs(x) if x != 0 else x)
    df['Longitud'] = df['Longitud'].apply(lambda x: -abs(x) if x != 0 else x)

    # Verificamos que el desplazamiento sea sobre la misma tarjeta
    mask_misma_tarjeta = df['Tarjeta'] == df['Tarjeta'].shift(-1)
    
    df['Lat_Destino'] = df['Latitud'].shift(-1)
    df['Lon_Destino'] = df['Longitud'].shift(-1)
    df['Sentido_Siguiente'] = df['Sentido'].shift(-1)
    df['Fecha_Siguiente'] = df['Fecha'].shift(-1)

    # También filtramos si el DESTINO es (0,0) por si acaso
    mask_cero_destino = (df['Lat_Destino'] != 0) & (df['Lon_Destino'] != 0)

    mask = (
        mask_misma_tarjeta &
        mask_cero_destino &
        (df['Sentido'] != df['Sentido_Siguiente']) & 
        (df['Fecha'] == df['Fecha_Siguiente']) &
        (df['Lat_Destino'].notna())
    )
    return df[mask].copy()

# --- 3. AGRUPACIÓN POR CERCANÍA ---
@st.cache_data
def agrupar_por_zonas(df, precision=3):
    df_agg = df.copy()
    df_agg['lat_ori'] = df_agg['Latitud'].round(precision)
    df_agg['lon_ori'] = df_agg['Longitud'].round(precision)
    df_agg['lat_des'] = df_agg['Lat_Destino'].round(precision)
    df_agg['lon_des'] = df_agg['Lon_Destino'].round(precision)
    
    df_zonas = df_agg.groupby(['lat_ori', 'lon_ori', 'lat_des', 'lon_des', 'Sentido']).size().reset_index(name='Pasajeros')
    return df_zonas

# --- 4. INTERFAZ DE USUARIO ---
archivo_subido = st.sidebar.file_uploader("Cargar archivo Parquet", type=["parquet"])

if archivo_subido:
    df_raw = cargar_datos(archivo_subido)
    
    st.sidebar.header("Filtros de Datos")
    fechas_disponibles = sorted(df_raw['Fecha'].unique())
    opciones_fecha = ["Todo el mes"] + [str(f) for f in fechas_disponibles]
    fecha_sel = st.sidebar.selectbox("Seleccionar Período", opciones_fecha)
    
    ramales = ["Todos"] + sorted(df_raw['Ramal'].unique().tolist())
    ramal_sel = st.sidebar.selectbox("Seleccionar Ramal", ramales)

    st.sidebar.markdown("---")
    st.sidebar.header("Filtros de Visualización")
    
    hora_rango = st.sidebar.slider("Rango Horario (Subida)", 0, 23, (0, 23))
    sentido_sel = st.sidebar.radio("Sentido de Subida", ["Ambos", "Ida", "Vuelta"])
    prec_sel = st.sidebar.slider("Precisión de agrupación", 2, 4, 3)
    mostrar_puntos = st.sidebar.toggle("Mostrar Transacciones Únicas (Puntos)", value=False)

    df_filtrado = df_raw.copy()
    if fecha_sel != "Todo el mes":
        fecha_obj = pd.to_datetime(fecha_sel).date()
        df_filtrado = df_filtrado[df_filtrado['Fecha'] == fecha_obj]
    if ramal_sel != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Ramal'] == ramal_sel]
    
    with st.spinner('Procesando vectores de flujo...'):
        df_flujos = calcular_vectores_flujo(df_filtrado)

    if not df_flujos.empty:
        df_mapa = df_flujos.copy()
        df_mapa = df_mapa[(df_mapa['Hora_Int'] >= hora_rango[0]) & (df_mapa['Hora_Int'] <= hora_rango[1])]
        if sentido_sel != "Ambos":
            df_mapa = df_mapa[df_mapa['Sentido'] == sentido_sel]

        if not df_mapa.empty:
            df_zonas = agrupar_por_zonas(df_mapa, precision=prec_sel)
            
            st.sidebar.markdown("---")
            st.sidebar.header("Filtro de Intensidad")
            min_pasajeros = st.sidebar.slider(
                "Ocultar flujos menores a:", 
                1, int(df_zonas['Pasajeros'].max()), 2
            )
            
            if st.sidebar.button("Centrar Mapa en Datos"):
                st.session_state.map_view["latitude"] = float(df_zonas["lat_ori"].mean())
                st.session_state.map_view["longitude"] = float(df_zonas["lon_ori"].mean())
                st.session_state.map_view["zoom"] = 12
            
            df_zonas_filtradas = df_zonas[df_zonas['Pasajeros'] >= min_pasajeros].copy()

            if not df_zonas_filtradas.empty:
                max_p = df_zonas_filtradas['Pasajeros'].max()
                
                # --- RECUPERAMOS EL COLOR ROJO ---
                def get_red_color(x):
                    ratio = x / max_p if max_p > 0 else 0
                    # Naranja [255, 165, 0] -> Rojo [255, 0, 0]
                    return [255, int(165 * (1 - ratio)), 0, 200]

                df_zonas_filtradas['color_ori'] = df_zonas_filtradas['Pasajeros'].apply(get_red_color)
                df_zonas_filtradas['color_des'] = [[0, 200, 255, 200]] * len(df_zonas_filtradas)
                df_zonas_filtradas['grosor_final'] = df_zonas_filtradas['Pasajeros'].clip(upper=40)

                capas = [
                    pdk.Layer(
                        "ArcLayer",
                        df_zonas_filtradas.to_dict(orient='records'),
                        get_source_position=["lon_ori", "lat_ori"],
                        get_target_position=["lon_des", "lat_des"],
                        get_source_color="color_ori",
                        get_target_color="color_des",
                        get_width="grosor_final",
                        pickable=True,
                        auto_highlight=True, # <--- RECUPERAMOS EL RESALTADO VIOLETA
                    )
                ]

                if mostrar_puntos:
                    df_pts = df_mapa[['Latitud', 'Longitud']].copy()
                    capas.append(pdk.Layer(
                        "ScatterplotLayer",
                        df_pts.to_dict(orient='records'),
                        get_position=["Longitud", "Latitud"],
                        get_color=[255, 255, 255, 60],
                        get_radius=20,
                    ))

                st.subheader(f"Análisis: {ramal_sel} | Umbral: >= {min_pasajeros} pasajeros")
                
                view_state = pdk.ViewState(
                    latitude=st.session_state.map_view["latitude"],
                    longitude=st.session_state.map_view["longitude"],
                    zoom=st.session_state.map_view["zoom"],
                    pitch=st.session_state.map_view["pitch"],
                    bearing=st.session_state.map_view["bearing"]
                )

                st.pydeck_chart(pdk.Deck(
                    map_provider="carto", map_style="light",
                    initial_view_state=view_state,
                    layers=capas,
                    tooltip={"html": "<b>Sentido:</b> {Sentido}<br><b>Pasajeros:</b> {Pasajeros}"}
                ))
            else:
                st.warning("No hay flujos para el umbral seleccionado.")
    else:
        st.warning("No se encontraron viajes.")
else:
    st.info("Carga un archivo .parquet para comenzar.")