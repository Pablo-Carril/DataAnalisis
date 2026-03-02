import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np

# Configuración de página
st.set_page_config(page_title="Análisis de Flujos de Transporte", layout="wide")

# --- 0. INICIALIZAR MEMORIA ---
# Guardamos el estado de la vista del mapa para que no se reinicie con cada filtro
if 'view_state' not in st.session_state: # Solo se inicializa una vez por sesión
    st.session_state.view_state = None
if 'last_major_filters' not in st.session_state:
    st.session_state.last_major_filters = None

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
    
    #Optimización de memoria: float64 -> float32
    for col in ['Latitud', 'Longitud']:
        if col in df.columns:
            df[col] = df[col].astype('float32')
            
    df['Hora_Int'] = df['Fecha Hora'].dt.hour
    df = df[(df['Latitud'] != 0) & (df['Longitud'] != 0)]
    df = df.dropna(subset=['Latitud', 'Longitud'])
    return df

# --- 2. LÓGICA DE PROCESAMIENTO ---
@st.cache_data
def calcular_vectores_flujo(df):
    df = df.sort_values(['Tarjeta', 'Fecha Hora'])
    df = df.rename(columns={'Latitud': 'Long_Original', 'Longitud': 'Lat_Original'})
    df = df.rename(columns={'Long_Original': 'Longitud', 'Lat_Original': 'Latitud'})
    df['Latitud'] = df['Latitud'].apply(lambda x: -abs(x) if x != 0 else x)
    df['Longitud'] = df['Longitud'].apply(lambda x: -abs(x) if x != 0 else x)

    mask_misma_tarjeta = df['Tarjeta'] == df['Tarjeta'].shift(-1)
    df['Lat_Destino'] = df['Latitud'].shift(-1)
    df['Lon_Destino'] = df['Longitud'].shift(-1)
    df['Sentido_Siguiente'] = df['Sentido'].shift(-1)
    df['Fecha_Siguiente'] = df['Fecha'].shift(-1)

    mask_cero_destino = (df['Lat_Destino'] != 0) & (df['Lon_Destino'] != 0)
    mask = (
        mask_misma_tarjeta &
        mask_cero_destino &
        (df['Sentido'] != df['Sentido_Siguiente']) & 
        (df['Fecha'] == df['Fecha_Siguiente']) &
        (df['Lat_Destino'].notna())
    )
    return df[mask].copy()

# --- 3. AGRUPACIÓN ---
@st.cache_data
def agrupar_por_zonas(df, precision=3):
    # Optimización: No copiamos el DF entero. Calculamos series al vuelo.
    factor = 10 ** precision
    
    # Redondeo vectorizado directo
    lat_ori = (df['Latitud'] * factor).round() / factor
    lon_ori = (df['Longitud'] * factor).round() / factor
    lat_des = (df['Lat_Destino'] * factor).round() / factor
    lon_des = (df['Lon_Destino'] * factor).round() / factor
    
    df_zonas = df.groupby([
        lat_ori.rename('lat_ori'), 
        lon_ori.rename('lon_ori'), 
        lat_des.rename('lat_des'), 
        lon_des.rename('lon_des'), 
        'Sentido'
    ]).size().reset_index(name='Pasajeros')
    
    return df_zonas

@st.cache_data
def calcular_estadisticas_nodos(df, precision=3):
    factor = 10 ** precision
    
    # Redondeo de coordenadas
    lat_ori = (df['Latitud'] * factor).round() / factor
    lon_ori = (df['Longitud'] * factor).round() / factor
    lat_des = (df['Lat_Destino'] * factor).round() / factor
    lon_des = (df['Lon_Destino'] * factor).round() / factor
    
    # Agrupación
    df_sub = pd.DataFrame({'lat': lat_ori, 'lon': lon_ori})
    df_baj = pd.DataFrame({'lat': lat_des, 'lon': lon_des})
    
    sub = df_sub.groupby(['lat', 'lon']).size().reset_index(name='Subieron')
    baj = df_baj.groupby(['lat', 'lon']).size().reset_index(name='Bajaron')
    
    # Merge
    nodos = pd.merge(sub, baj, on=['lat', 'lon'], how='outer').fillna(0)
    nodos['Subieron'] = nodos['Subieron'].astype(int)
    nodos['Bajaron'] = nodos['Bajaron'].astype(int)
    return nodos

# --- 4. INTERFAZ DE USUARIO ---
#archivo_subido = st.sidebar.file_uploader("Cargar archivo Parquet", type=["parquet"])
archivo_subido = "Transacciones saes octubre.parquet"

if archivo_subido:
    df_raw = cargar_datos(archivo_subido)
    
    st.sidebar.header("Filtros de Datos")
    fechas_disponibles = sorted(df_raw['Fecha'].unique())
    opciones_fecha = ["Todo el mes"] + [str(f) for f in fechas_disponibles]
    fecha_sel = st.sidebar.selectbox("Seleccionar Período", opciones_fecha)
    
    ramales = ["Todos"] + sorted(df_raw['Ramal'].unique().tolist())
    ramal_sel = st.sidebar.selectbox("Seleccionar Ramal", ramales)

    st.sidebar.markdown("---")
    # Botón para resetear la vista del mapa
    if st.sidebar.button("📍 Resetear Vista del Mapa"):
        st.session_state.view_state = None
        # El script se re-ejecutará naturalmente, aplicando el reseteo.

    st.sidebar.header("Filtros de Visualización")
    
    # Consolidación de filtros para evitar DuplicateElementId y NameError
    hora_rango = st.sidebar.slider("Rango Horario (Subida)", 0, 23, (0, 23), key="slider_h")
    sentido_sel = st.sidebar.radio("Sentido de Subida", ["Ambos", "Ida", "Vuelta"], index=1, key="radio_s")
    
    # Cambio: Slider en Metros en lugar de precisión abstracta
    metros_sel = st.sidebar.select_slider("Tamaño de zona (metros aprox)", options=[50, 100, 200, 300, 400, 500, 800, 1000, 1500, 2000], value=100, key="slider_m")
    # Cálculo inverso: Convertir metros a precisión decimal (1 grado lat ~ 111,111 metros)
    prec_sel = np.log10(111111 / metros_sel)
    
    mostrar_puntos = st.sidebar.toggle("Mostrar Puntos", value=True, key="toggle_ptos")
    mostrar_grilla = st.sidebar.toggle("Mostrar Grilla", value=False, key="toggle_grid")
    min_pasajeros = st.sidebar.number_input("Ocultar flujos menores a:", 1, 1000, value=2, key="num_i")

    # Aplicación de filtros
    df_filtrado = df_raw.copy()
    if fecha_sel != "Todo el mes":
        fecha_obj = pd.to_datetime(fecha_sel).date()
        df_filtrado = df_filtrado[df_filtrado['Fecha'] == fecha_obj]
    if ramal_sel != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Ramal'] == ramal_sel]
    
    with st.spinner('Procesando vectores de flujo...'):
        df_flujos = calcular_vectores_flujo(df_filtrado)

    # --- LÓGICA DE VISTA DE MAPA ESTABLE ---
    # Se define qué filtros fuerzan un reseteo del centro del mapa.
    current_major_filters = (fecha_sel, ramal_sel)

    # Se recalcula el centro SÓLO si:
    # 1. Es la primera carga o se ha pulsado el botón de reseteo (view_state is None).
    # 2. Han cambiado los filtros principales (fecha o ramal).
    if st.session_state.view_state is None or st.session_state.last_major_filters != current_major_filters:
        if not df_flujos.empty:
            lat_centro = float(df_flujos["Latitud"].mean())
            lon_centro = float(df_flujos["Longitud"].mean())
            st.session_state.view_state = pdk.ViewState(
                latitude=lat_centro, longitude=lon_centro, zoom=12, pitch=45, bearing=0
            )
            # Se guarda la configuración de filtros con la que se calculó este centro.
            st.session_state.last_major_filters = current_major_filters
        elif st.session_state.view_state is None: # Fallback solo para la primera carga si no hay datos
             st.session_state.view_state = pdk.ViewState(latitude=-34.921, longitude=-57.954, zoom=12, pitch=45, bearing=0)

    if not df_flujos.empty:
        # Filtros de Mapa aplicados sobre los vectores
        # Optimización: Filtrado directo sin copia inicial
        mask_hora = (df_flujos['Hora_Int'] >= hora_rango[0]) & (df_flujos['Hora_Int'] <= hora_rango[1])
        df_mapa = df_flujos[mask_hora]
        
        if sentido_sel != "Ambos":
            df_mapa = df_mapa[df_mapa['Sentido'] == sentido_sel]

        if not df_mapa.empty:
            capas = []
            df_zonas = agrupar_por_zonas(df_mapa, precision=prec_sel)
            df_zonas = df_zonas[df_zonas['Pasajeros'] >= min_pasajeros].reset_index(drop=True)

            # Calcular estadísticas de nodos (Subidas/Bajadas)
            df_nodos = calcular_estadisticas_nodos(df_mapa, precision=prec_sel)

            # 0. Capa de Grilla (Fondo)
            if mostrar_grilla:
                capas.append(pdk.Layer(
                    "GridLayer",
                    df_mapa[['Latitud', 'Longitud']],
                    get_position=["Longitud", "Latitud"],
                    cell_size=metros_sel,
                    extruded=False,
                    pickable=False,
                    color_range=[
                        [255, 255, 0, 20],
                        [255, 255, 0, 100],
                        [255, 255, 0, 200],
                        [255, 255, 0, 255]
                    ],
                    coverage=0.9,
                ))

            # 1. Capa de Arcos (Flujos)
            if not df_zonas.empty:
                max_p = int(df_zonas['Pasajeros'].max())
                
                def color_log(x):
                    ratio = np.log1p(x) / np.log1p(max_p) if max_p > 1 else 0
                    return [255, int(165 * (1 - ratio)), 0, 200]

                df_zonas['color_ori'] = df_zonas['Pasajeros'].apply(color_log)
                # Eliminamos la columna 'color_des' para ahorrar memoria, usamos constante en el Layer
                df_zonas['grosor_final'] = df_zonas['Pasajeros'].clip(upper=40).astype(float)

                # Campos vacíos para tooltip consistente
                df_zonas['Subieron'] = ""
                df_zonas['Bajaron'] = ""

                capas.append(pdk.Layer(
                    "ArcLayer",
                    df_zonas, # Pasamos el DataFrame DIRECTAMENTE
                    id="arcos",
                    get_source_position=["lon_ori", "lat_ori"],
                    get_target_position=["lon_des", "lat_des"],
                    get_source_color="color_ori",
                    get_target_color=[0, 150, 255, 200], # Color constante
                    get_width="grosor_final",
                    pickable=True,
                    auto_highlight=True,
                ))

            # 2. Capa de Puntos (Transacciones)
            if mostrar_puntos:
                capas.append(pdk.Layer(
                    "ScatterplotLayer",
                    df_mapa[['Latitud', 'Longitud']], # Pasamos DataFrame directo
                    get_position=["Longitud", "Latitud"],
                    get_color=[140, 140, 140, 100], # Gris
                    get_radius=20,
                ))

                # Puntos agrupados con estadísticas
                if not df_nodos.empty:
                    # Campos vacíos para tooltip consistente
                    df_nodos['Pasajeros'] = ""

                    capas.append(pdk.Layer(
                        "ScatterplotLayer",
                        df_nodos,
                        get_position=["lon", "lat"],
                        get_color=[20, 150, 0, 120], # verde 
                        get_radius=30,
                        pickable=True,
                    ))

            # 3. Renderizado del mapa si hay capas que mostrar
            if capas:
                max_p_display = int(df_zonas['Pasajeros'].max()) if not df_zonas.empty else 0

                st.subheader(f"Análisis: {ramal_sel} | Máx: {max_p_display} pasajeros en un corredor")
                st.pydeck_chart(pdk.Deck(
                    map_provider="carto",
                    map_style="light",
                    initial_view_state=st.session_state.view_state, # Usamos SIEMPRE la vista guardada
                    layers=capas,
                    tooltip={
                        "html": "<b>Pasajeros:</b> {Pasajeros}<br/>"
                                "<b>Subieron:</b> {Subieron}<br/>"
                                "<b>Bajaron:</b> {Bajaron}"
                    }
                ), key="deck_map")

                # La funcionalidad de selección se ha desactivado temporalmente para estabilizar el zoom.
            else:
                st.warning("No se encontraron flujos ni transacciones para los filtros aplicados.")
        else:
            st.warning("No hay viajes que coincidan con los filtros de hora y sentido.")
    else:
        st.warning("No se encontraron viajes.")
else:
    st.info("Carga un archivo .parquet para comenzar.")