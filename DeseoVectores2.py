import streamlit as st
import pandas as pd
import pydeck as pdk
import folium
from streamlit_folium import st_folium

# Configuración de página
st.set_page_config(page_title="Análisis de Flujos de Transporte", layout="wide")

st.title("📊 Mapa de Flujos de Transacciones (Agrupado)")
st.markdown("""
Esta herramienta agrupa viajes cercanos para visualizar los **corredores de mayor demanda**. 
El grosor del arco indica la cantidad de pasajeros que realizaron ese trayecto.
""")

# --- 1. CARGA DE DATOS ---
@st.cache_data
def cargar_datos(archivo):
    df = pd.read_parquet(archivo)
    df['Fecha Hora'] = pd.to_datetime(df['Fecha Hora'])
    if 'Fecha' in df.columns:
        df = df.drop(columns=['Fecha'])
    df['Fecha'] = df['Fecha Hora'].dt.date
    # Creamos columna de hora para el filtro
    df['Hora_Int'] = df['Fecha Hora'].dt.hour
    df = df.dropna(subset=['Latitud', 'Longitud'])
    return df

# --- 2. LÓGICA DE PROCESAMIENTO (ORIGEN-DESTINO) ---
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

# --- 3. NUEVA FUNCIÓN: AGRUPACIÓN POR CERCANÍA ---
@st.cache_data
def agrupar_por_zonas(df, precision=3):
    df_agg = df.copy()
    df_agg['lat_ori'] = df_agg['Latitud'].round(precision)
    df_agg['lon_ori'] = df_agg['Longitud'].round(precision)
    df_agg['lat_des'] = df_agg['Lat_Destino'].round(precision)
    df_agg['lon_des'] = df_agg['Lon_Destino'].round(precision)
    
    # Incluimos Sentido en la agrupación para el tooltip
    df_zonas = df_agg.groupby(['lat_ori', 'lon_ori', 'lat_des', 'lon_des', 'Sentido']).size().reset_index(name='Pasajeros')
    return df_zonas

# --- 4. INTERFAZ DE USUARIO ---
archivo_subido = st.sidebar.file_uploader("Cargar archivo Parquet", type=["parquet"])

if archivo_subido:
    df_raw = cargar_datos(archivo_subido)
    
    st.sidebar.header("Filtros de Datos")
    
    # Filtro de Período
    fechas_disponibles = sorted(df_raw['Fecha'].unique())
    opciones_fecha = ["Todo el mes"] + [str(f) for f in fechas_disponibles]
    fecha_sel = st.sidebar.selectbox("Seleccionar Período", opciones_fecha)
    
    # Filtro de Ramal
    ramales = ["Todos"] + sorted(df_raw['Ramal'].unique().tolist())
    ramal_sel = st.sidebar.selectbox("Seleccionar Ramal", ramales)

    st.sidebar.markdown("---")
    st.sidebar.header("Filtros de Visualización")
    
    # Filtro de Hora
    hora_rango = st.sidebar.slider("Rango Horario (Subida)", 0, 23, (0, 23))
    
    # Filtro de Sentido
    sentido_sel = st.sidebar.radio("Sentido de Subida", ["Ambos", "Ida", "Vuelta"])

    # Slider de agrupación
    prec_sel = st.sidebar.slider("Precisión de agrupación", 2, 4, 3, 
                                 help="2: Barrios (~1km), 3: Cuadras (~100m), 4: Esquinas")

    mostrar_puntos = st.sidebar.toggle("Mostrar Transacciones Únicas (Puntos)", value=False)

    # Aplicación de filtros iniciales
    df_filtrado = df_raw.copy()
    if fecha_sel != "Todo el mes":
        fecha_obj = pd.to_datetime(fecha_sel).date()
        df_filtrado = df_filtrado[df_filtrado['Fecha'] == fecha_obj]
    
    if ramal_sel != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Ramal'] == ramal_sel]
    
    with st.spinner(f'Procesando flujos...'):
        df_flujos = calcular_vectores_flujo(df_filtrado)

    if not df_flujos.empty:
        # Aplicamos filtros de visualización (Hora y Sentido)
        df_mapa = df_flujos.copy()
        df_mapa = df_mapa[(df_mapa['Hora_Int'] >= hora_rango[0]) & (df_mapa['Hora_Int'] <= hora_rango[1])]
        
        if sentido_sel != "Ambos":
            df_mapa = df_mapa[df_mapa['Sentido'] == sentido_sel]

        if not df_mapa.empty:
            df_zonas = agrupar_por_zonas(df_mapa, precision=prec_sel)
            max_p = df_zonas['Pasajeros'].max()
            
            # Lógica de colores (Rojo/Naranja -> Celeste)
            def color_origen(cantidad):
                ratio = cantidad / max_p if max_p > 0 else 0
                return [255, int(165 * (1 - ratio)), 0, 200]

            df_zonas['color_ori'] = df_zonas['Pasajeros'].apply(color_origen)
            df_zonas['color_des'] = df_zonas['Pasajeros'].apply(lambda x: [0, 200, 255, 200]) # Celeste para todos los destinos
            
            # Grosor visual relativo
            df_zonas['grosor_real'] = df_zonas['Pasajeros'].clip(upper=40)

            # Conversión para Pydeck
            dict_zonas = df_zonas.to_dict(orient='records')

            layer_arcos_final = pdk.Layer(
                "ArcLayer",
                dict_zonas,
                get_source_position=["lon_ori", "lat_ori"],
                get_target_position=["lon_des", "lat_des"],
                get_source_color="color_ori", 
                get_target_color="color_des",
                get_width="grosor_real",
                pickable=True,
                auto_highlight=True
            )

            view_state = pdk.ViewState(
                latitude=df_zonas["lat_ori"].mean(),
                longitude=df_zonas["lon_ori"].mean(),
                zoom=12,
                pitch=45,
            )

            st.subheader(f"Análisis de Corredores: {ramal_sel} ({sentido_sel})")
            st.caption(f"Período: {fecha_sel} | Horario: {hora_rango[0]}hs a {hora_rango[1]}hs")

            st.pydeck_chart(pdk.Deck(
                map_provider="carto",
                map_style="light",
                initial_view_state=view_state,
                layers=[layer_arcos_final],
                tooltip={"html": "<b>Sentido:</b> {Sentido}<br><b>Flujo:</b> {Pasajeros} pasajeros"}
            ))

            if st.checkbox("Ver tabla de flujos agrupados"):
                st.dataframe(df_zonas.sort_values('Pasajeros', ascending=False))
        else:
            st.warning("No hay datos para el rango horario o sentido seleccionado.")
    else:
        st.warning("No se encontraron viajes de ida y vuelta con los filtros de Fecha/Ramal.")
else:
    st.info("Por favor, carga un archivo .parquet en la barra lateral.")