import streamlit as st
import pandas as pd
import pydeck as pdk

# Configuración de página
st.set_page_config(page_title="Análisis de Flujos de Transporte", layout="wide")

st.title("📊 Mapa de Flujos de Transacciones (Agrupado)")
st.markdown("""
Esta herramienta agrupa viajes cercanos para visualizar los **corredores de mayor demanda**. 
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
    df = df.dropna(subset=['Latitud', 'Longitud'])
    return df

# --- 2. LÓGICA DE PROCESAMIENTO ---
@st.cache_data
def calcular_vectores_flujo(df):
    df = df.sort_values(['Tarjeta', 'Fecha Hora'])
    
    # Inversión de coordenadas (Lógica original)
    df = df.rename(columns={'Latitud': 'Long_Original', 'Longitud': 'Lat_Original'})
    df = df.rename(columns={'Long_Original': 'Longitud', 'Lat_Original': 'Latitud'})
    
    # Aseguramos coordenadas negativas para Argentina
    df['Latitud'] = df['Latitud'].apply(lambda x: -abs(x) if x != 0 else x)
    df['Longitud'] = df['Longitud'].apply(lambda x: -abs(x) if x != 0 else x)

    df_sorted = df.groupby('Tarjeta', group_keys=False).apply(lambda x: x.assign(
        Lat_Destino=x['Latitud'].shift(-1),
        Lon_Destino=x['Longitud'].shift(-1),
        Sentido_Siguiente=x['Sentido'].shift(-1),
        Fecha_Siguiente=x['Fecha'].shift(-1)
    )).reset_index(drop=True)

    mask = (
        (df_sorted['Sentido'] != df_sorted['Sentido_Siguiente']) & 
        (df_sorted['Fecha'] == df_sorted['Fecha_Siguiente']) &
        (df_sorted['Lat_Destino'].notna())
    )
    return df_sorted[mask]

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

    # NUEVO SELECTOR: Mostrar/Ocultar transacciones únicas
    mostrar_puntos = st.sidebar.toggle("Mostrar Transacciones Únicas (Puntos)", value=False)

    # Filtrado inicial
    df_filtrado = df_raw.copy()
    if fecha_sel != "Todo el mes":
        fecha_obj = pd.to_datetime(fecha_sel).date()
        df_filtrado = df_filtrado[df_filtrado['Fecha'] == fecha_obj]
    if ramal_sel != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Ramal'] == ramal_sel]
    
    with st.spinner('Procesando...'):
        df_flujos = calcular_vectores_flujo(df_filtrado)

if not df_flujos.empty:
        # 1. Filtros de visualización (Hora y Sentido)
        df_mapa = df_flujos.copy()
        df_mapa = df_mapa[(df_mapa['Hora_Int'] >= hora_rango[0]) & (df_mapa['Hora_Int'] <= hora_rango[1])]
        if sentido_sel != "Ambos":
            df_mapa = df_mapa[df_mapa['Sentido'] == sentido_sel]

        if not df_mapa.empty:
            # 2. Agrupación base
            df_zonas = agrupar_por_zonas(df_mapa, precision=prec_sel)
            
            # --- NUEVO FILTRO DE UMBRAL DE PASAJEROS ---
            st.sidebar.markdown("---")
            st.sidebar.header("Filtro de Intensidad")
            min_pasajeros = st.sidebar.slider(
                "Ocultar flujos menores a:", 
                min_value=1, 
                max_value=int(df_zonas['Pasajeros'].max()), 
                value=2, # Por defecto ocultamos las únicas (valor 1)
                help="Filtra arcos según la cantidad de pasajeros acumulados."
            )
            
            # Aplicamos el filtro al DataFrame de las zonas
            df_zonas_filtradas = df_zonas[df_zonas['Pasajeros'] >= min_pasajeros].copy()
            # -------------------------------------------

            if not df_zonas_filtradas.empty:
                max_p = df_zonas_filtradas['Pasajeros'].max()
                
                # Lógica de colores y grosor (Tope 40) sobre el set filtrado
                df_zonas_filtradas['color_ori'] = df_zonas_filtradas['Pasajeros'].apply(
                    lambda x: [255, int(165 * (1 - (x/max_p))), 0, 200]
                )
                df_zonas_filtradas['color_des'] = [[0, 200, 255, 200]] * len(df_zonas_filtradas)
                df_zonas_filtradas['grosor_final'] = df_zonas_filtradas['Pasajeros'].clip(upper=40)

                capas = []

                # 1. Capa de Arcos (Corredores filtrados)
                capas.append(pdk.Layer(
                    "ArcLayer",
                    df_zonas_filtradas.to_dict(orient='records'),
                    get_source_position=["lon_ori", "lat_ori"],
                    get_target_position=["lon_des", "lat_des"],
                    get_source_color="color_ori",
                    get_target_color="color_des",
                    get_width="grosor_final",
                    pickable=True,
                ))

                # 2. Capa de Puntos (Opcional)
                if mostrar_puntos:
                    df_puntos_limpios = df_mapa[['Latitud', 'Longitud']].copy()
                    capas.append(pdk.Layer(
                        "ScatterplotLayer",
                        df_puntos_limpios.to_dict(orient='records'),
                        get_position=["Longitud", "Latitud"],
                        get_color=[255, 255, 255, 60],
                        get_radius=20,
                    ))

                # Renderizado
                st.subheader(f"Análisis: {ramal_sel} | Umbral: >= {min_pasajeros} pasajeros")
                st.pydeck_chart(pdk.Deck(
                    map_provider="carto", map_style="light",
                    initial_view_state=pdk.ViewState(
                        latitude=df_zonas_filtradas["lat_ori"].mean(),
                        longitude=df_zonas_filtradas["lon_ori"].mean(),
                        zoom=12, pitch=45
                    ),
                    layers=capas,
                    tooltip={"html": "<b>Pasajeros:</b> {Pasajeros}"}
                ))
            else:
                st.warning("No hay flujos que superen el umbral seleccionado.")
else:
    st.info("Carga un archivo .parquet para comenzar.")