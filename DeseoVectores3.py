import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import os

# Inyectar CSS para ocultar el menú 
css_style = """
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
st.markdown(css_style, unsafe_allow_html=True)


# Configuración de página
st.set_page_config(page_title="Análisis de Flujos", layout="wide")

# --- 0. INICIALIZAR MEMORIA ---
# Guardamos el estado de la vista del mapa para que no se reinicie con cada filtro
if 'view_state' not in st.session_state: # Solo se inicializa una vez por sesión
    st.session_state.view_state = None
if 'last_major_filters' not in st.session_state:
    st.session_state.last_major_filters = None

st.title("📊 Mapa de Flujos de Pasajeros")
st.markdown("""
Mapa de Deseo. Esta herramienta agrupa viajes cercanos para visualizar los **corredores de mayor demanda**.
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
def agrupar_por_zonas(df, df_ruta, metros_sel=100):
    # Si no hay ruta, no se puede agrupar.
    if df_ruta.empty:
        return pd.DataFrame()

    # Coordenadas de la ruta de referencia
    ruta_lats = df_ruta['Latitud'].values
    ruta_lons = df_ruta['Longitud'].values
    ruta_cum = df_ruta['Dist_Acum'].values # Distancia acumulada en KM

    # Coordenadas de los viajes
    lats_ori = df['Latitud'].values
    lons_ori = df['Longitud'].values
    lats_des = df['Lat_Destino'].values
    lons_des = df['Lon_Destino'].values

    # 1. Encontrar los índices de los puntos más cercanos en la ruta
    idx_ori = np.argmin((lats_ori[:, None] - ruta_lats[None, :])**2 + (lons_ori[:, None] - ruta_lons[None, :])**2, axis=1)
    idx_des = np.argmin((lats_des[:, None] - ruta_lats[None, :])**2 + (lons_des[:, None] - ruta_lons[None, :])**2, axis=1)

    # 2. Obtener la distancia acumulada para cada punto
    dist_acum_ori = ruta_cum[idx_ori]
    dist_acum_des = ruta_cum[idx_des]

    # 3. Agrupar por distancia (binning)
    km_bin = metros_sel / 1000.0
    dist_binned_ori = (dist_acum_ori / km_bin).round() * km_bin
    dist_binned_des = (dist_acum_des / km_bin).round() * km_bin

    # 4. Encontrar los puntos de la ruta que corresponden a las distancias agrupadas
    idx_binned_ori = np.argmin(np.abs(dist_binned_ori[:, None] - ruta_cum[None, :]), axis=1)
    idx_binned_des = np.argmin(np.abs(dist_binned_des[:, None] - ruta_cum[None, :]), axis=1)

    # Crear un DataFrame temporal con las coordenadas "snapped" y "binned" a la ruta
    df_snapped = pd.DataFrame({
        'lat_ori': ruta_lats[idx_binned_ori],
        'lon_ori': ruta_lons[idx_binned_ori],
        'lat_des': ruta_lats[idx_binned_des],
        'lon_des': ruta_lons[idx_binned_des],
        'Sentido': df['Sentido'].values
    })
    
    # Agrupar por las coordenadas de la ruta
    df_zonas = df_snapped.groupby([
        'lat_ori', 'lon_ori', 'lat_des', 'lon_des', 'Sentido'
    ]).size().reset_index(name='Pasajeros')
    
    return df_zonas

@st.cache_data
def calcular_estadisticas_nodos(df, df_ruta, metros_sel=100):
    if df_ruta.empty:
        return pd.DataFrame()

    # Coordenadas de la ruta de referencia
    ruta_lats = df_ruta['Latitud'].values
    ruta_lons = df_ruta['Longitud'].values
    ruta_cum = df_ruta['Dist_Acum'].values

    # Coordenadas de los viajes
    lats_ori = df['Latitud'].values
    lons_ori = df['Longitud'].values
    lats_des = df['Lat_Destino'].values
    lons_des = df['Lon_Destino'].values

    # 1. Encontrar los índices de los puntos más cercanos en la ruta
    idx_ori = np.argmin((lats_ori[:, None] - ruta_lats[None, :])**2 + (lons_ori[:, None] - ruta_lons[None, :])**2, axis=1)
    idx_des = np.argmin((lats_des[:, None] - ruta_lats[None, :])**2 + (lons_des[:, None] - ruta_lons[None, :])**2, axis=1)

    # 2. Obtener la distancia acumulada para cada punto
    dist_acum_ori = ruta_cum[idx_ori]
    dist_acum_des = ruta_cum[idx_des]

    # 3. Agrupar por distancia (binning)
    km_bin = metros_sel / 1000.0
    dist_binned_ori = (dist_acum_ori / km_bin).round() * km_bin
    dist_binned_des = (dist_acum_des / km_bin).round() * km_bin

    # 4. Encontrar los puntos de la ruta que corresponden a las distancias agrupadas
    idx_binned_ori = np.argmin(np.abs(dist_binned_ori[:, None] - ruta_cum[None, :]), axis=1)
    idx_binned_des = np.argmin(np.abs(dist_binned_des[:, None] - ruta_cum[None, :]), axis=1)
    
    # Crear DataFrames con coordenadas "snapped" y "binned"
    df_sub = pd.DataFrame({'lat': ruta_lats[idx_binned_ori], 'lon': ruta_lons[idx_binned_ori]})
    df_baj = pd.DataFrame({'lat': ruta_lats[idx_binned_des], 'lon': ruta_lons[idx_binned_des]})

    sub = df_sub.groupby(['lat', 'lon']).size().reset_index(name='Subieron')
    baj = df_baj.groupby(['lat', 'lon']).size().reset_index(name='Bajaron')
    
    # Merge
    nodos = pd.merge(sub, baj, on=['lat', 'lon'], how='outer').fillna(0)
    nodos['Subieron'] = nodos['Subieron'].astype(int)
    nodos['Bajaron'] = nodos['Bajaron'].astype(int)
    return nodos

# --- 5. FUNCIONES DE DISTANCIA (RUTA) ---
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

@st.cache_data
def cargar_ruta_referencia(archivo):
    try:
        df = pd.read_csv(archivo, sep=';', decimal=',', names=['Ramal', 'Sentido', 'Latitud', 'Longitud', 'Orden'])
        df = df.sort_values('Orden').reset_index(drop=True)
        # Calcular distancias acumuladas a lo largo de la ruta
        lats = df['Latitud'].values
        lons = df['Longitud'].values
        dists = haversine_np(lons[:-1], lats[:-1], lons[1:], lats[1:])
        df['Dist_Acum'] = np.concatenate(([0], np.cumsum(dists)))
        return df
    except Exception as e:
        return pd.DataFrame()

def calcular_distancia_traza_vectorizado(lats_ori, lons_ori, lats_des, lons_des, df_ruta):
    ruta_lats = df_ruta['Latitud'].values
    ruta_lons = df_ruta['Longitud'].values
    ruta_cum = df_ruta['Dist_Acum'].values
    
    # Encontrar índices más cercanos en la ruta (Broadcasting: Zonas x Ruta)
    idx_ori = np.argmin((lats_ori[:, None] - ruta_lats[None, :])**2 + (lons_ori[:, None] - ruta_lons[None, :])**2, axis=1)
    idx_des = np.argmin((lats_des[:, None] - ruta_lats[None, :])**2 + (lons_des[:, None] - ruta_lons[None, :])**2, axis=1)
    
    return np.abs(ruta_cum[idx_des] - ruta_cum[idx_ori])

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

    #st.sidebar.markdown("---")
    # Botón para resetear la vista del mapa
    #if st.sidebar.button("📍 Resetear Vista del Mapa"):
       # st.session_state.view_state = None
        # El script se re-ejecutará naturalmente, aplicando el reseteo.

    st.sidebar.header("Filtros de Visualización")
    
    # Consolidación de filtros para evitar DuplicateElementId y NameError
    hora_rango = st.sidebar.slider("Rango Horario (Subida)", 0, 23, (0, 23), key="slider_h")
    sentido_sel = st.sidebar.radio("Sentido de Subida", ["Ambos", "Ida", "Vuelta"], index=1, key="radio_s")
    
    metros_sel = st.sidebar.select_slider("Tamaño de Agrupación (metros)", options=[50, 100, 200, 300, 400, 500], value=100)

    mostrar_puntos = st.sidebar.toggle("Mostrar Puntos", value=True, key="toggle_ptos")
    min_pasajeros = st.sidebar.number_input("Ocultar flujos menores a:", 1, 1000, value=2, key="num_i")

    # --- Carga dinámica de la ruta de referencia ---
    df_ruta = pd.DataFrame()
    nombre_archivo_ruta = ""
    # Solo intentamos cargar una ruta si se ha seleccionado un ramal y un sentido específicos.
    if ramal_sel != "Todos" and sentido_sel != "Ambos":
        # Intentamos detectar el archivo con o sin el ID de línea (9433)
        posibles_nombres = [
           # f"Ramal_{ramal_sel}_{sentido_sel}_9433.ACTrec",
            f"Ramal_{ramal_sel}_{sentido_sel}.ACTrec"
        ]
        
        for nombre in posibles_nombres:
            if os.path.exists(nombre):
                nombre_archivo_ruta = nombre
                break
        
        if nombre_archivo_ruta:
            df_ruta = cargar_ruta_referencia(nombre_archivo_ruta)
            if df_ruta.empty:
                st.sidebar.error(f"El archivo existe pero falló la lectura: {nombre_archivo_ruta}")
        else:
            st.sidebar.warning(f"No se encontró archivo de ruta (ej. {posibles_nombres[0]})")
            st.sidebar.info(f"Carpeta actual: {os.getcwd()}")
            
    elif ramal_sel == "Todos" or sentido_sel == "Ambos":
        st.sidebar.info("Seleccione un Ramal y Sentido para visualizar la ruta y agrupar los flujos sobre ella.")

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
            df_zonas = agrupar_por_zonas(df_mapa, df_ruta, metros_sel)
            
            # Verificamos que se hayan generado zonas antes de filtrar para evitar KeyError
            if not df_zonas.empty and 'Pasajeros' in df_zonas.columns:
                df_zonas = df_zonas[df_zonas['Pasajeros'] >= min_pasajeros].reset_index(drop=True)

            if not df_ruta.empty:
                path_data = pd.DataFrame({
                    'path': [df_ruta[['Longitud', 'Latitud']].values.tolist()]
                })
                capas.append(pdk.Layer(
                    'PathLayer',
                    data=path_data,
                    get_path='path',
                    get_color=[200, 40, 40, 160], # color
                    get_width=15,
                    width_min_pixels=3,
                    id='ruta_referencia_layer'
                ))

            # --- CÁLCULO DE DISTANCIAS ---
            if not df_ruta.empty and not df_zonas.empty:
                df_zonas['Km_Recorridos'] = calcular_distancia_traza_vectorizado(
                    df_zonas['lat_ori'].values, df_zonas['lon_ori'].values,
                    df_zonas['lat_des'].values, df_zonas['lon_des'].values,
                    df_ruta
                )
                
                total_pax = df_zonas['Pasajeros'].sum()
                if total_pax > 0:
                    dist_media = (df_zonas['Km_Recorridos'] * df_zonas['Pasajeros']).sum() / total_pax
                    st.metric("Distancia Media Ponderada (km)", f"{dist_media:.2f}")

            # Calcular estadísticas de nodos (Subidas/Bajadas)
            df_nodos = calcular_estadisticas_nodos(df_mapa, df_ruta, metros_sel)

            # 0. Capa de Grilla (Fondo)

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

                st.subheader(f"Ramal: {ramal_sel} - {sentido_sel} - Suben: en Rojo - Bajan: en Azul")#| Máx: {max_p_display} pasajeros en un corredor")
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

                # --- EXPORTACIÓN DE DATOS ---
                if not df_zonas.empty:
                    st.markdown("---")
                    df_export = df_zonas.copy()
                    df_export['Ramal'] = ramal_sel
                    
                    # Renombrar para coincidir con solicitud
                    if 'Km_Recorridos' in df_export.columns:
                        df_export = df_export.rename(columns={'Km_Recorridos': 'distancia'})
                    
                    cols_export = ['Ramal', 'Sentido', 'lat_ori', 'lon_ori', 'lat_des', 'lon_des', 'distancia', 'Pasajeros']
                    cols_final = [c for c in cols_export if c in df_export.columns]
                    
                    csv = df_export[cols_final].to_csv(index=False, sep=';', decimal=',')
                    
                    st.download_button(
                        label="📥 Descargar CSV (Flujos)",
                        data=csv,
                        file_name=f"flujos_{ramal_sel}_{sentido_sel}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No se encontraron flujos ni transacciones para los filtros aplicados.")
        else:
            st.warning("No hay viajes que coincidan con los filtros de hora y sentido.")
    else:
        st.warning("No se encontraron viajes.")
else:
    st.info("Carga un archivo .parquet para comenzar.")