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
    dist_rango = st.sidebar.slider("Filtrar por Distancia (km)", 0.0, 50.0, (0.0, 50.0), step=0.5, key="slider_dist")

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

            # --- CÁLCULO DE DISTANCIAS Y FILTRO ---
            if not df_ruta.empty and not df_zonas.empty:
                df_zonas['Km_Recorridos'] = calcular_distancia_traza_vectorizado(
                    df_zonas['lat_ori'].values, df_zonas['lon_ori'].values,
                    df_zonas['lat_des'].values, df_zonas['lon_des'].values,
                    df_ruta
                )

                # Aplicar filtro de distancia del slider
                df_zonas = df_zonas[
                    (df_zonas['Km_Recorridos'] >= dist_rango[0]) &
                    (df_zonas['Km_Recorridos'] <= dist_rango[1])
                ].reset_index(drop=True)
                
                # Recalcular métricas con los datos ya filtrados por distancia
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
                log_max_p = np.log1p(max_p) if max_p > 0 else 1
                
                def color_log(x):
                    ratio = np.log1p(x) / log_max_p if log_max_p > 0 else 0
                    return [255, int(165 * (1 - ratio)), 0, 200]

                df_zonas['color_ori'] = df_zonas['Pasajeros'].apply(color_log)
                
                # ANCHO DE LÍNEA LOGARÍTMICO: Más sutil y no satura al alejar el zoom.
                df_zonas['grosor_final'] = (1 + (np.log1p(df_zonas['Pasajeros']) / log_max_p) * 14) if log_max_p > 0 else 1

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

                # --- CREACIÓN DE PESTAÑAS DE VISUALIZACIÓN ---
                tab1, tab2 = st.tabs(["🗺️ Mapa 3D (Arcos)", "🛰️ Vista 2D (Vectores)"])

                with tab1:
                    st.subheader(f"Ramal: {ramal_sel} - {sentido_sel} | Suben: Naranja/Rojo - Bajan: Azul")
                    st.pydeck_chart(pdk.Deck(
                        map_provider="carto",
                        map_style="light",
                        initial_view_state=st.session_state.view_state, # Usamos SIEMPRE la vista guardada
                        layers=capas,
                        tooltip={
                            "html": "<b>Pasajeros:</b> {Pasajeros}<br/>"
                                    "<b>Subieron:</b> {Subieron}<br/>"
                                    "<b>Bajaron:</b> {Bajaron}"
                        },
                        height=900
                    ), key="deck_map_3d")

                with tab2:
                    if not df_zonas.empty:
                        # --- PREPARACIÓN PARA VISTA 2D ---
                        # Se crea una copia para la Tab 2, se ordena para el Z-Index y se trabaja sobre ella.
                        # Esto evita modificar el df_zonas original que se usa en la Tab 1 y en la exportación.
                        df_2d = df_zonas.copy().sort_values('Pasajeros', ascending=True).reset_index(drop=True)
                        
                        # 1. Cálculo de Colores (Gradiente de Matiz y Transparencia)
                        max_p_2d = df_2d['Pasajeros'].max()
                        
                        def get_color_2d(p):
                            ratio = p / max_p_2d if max_p_2d > 0 else 0
                            # Alpha: 40 (muy transparente) a 255 (opaco)
                            alpha = int(40 + (215 * ratio))
                            
                            # Matiz: Cian -> Azul -> Verde -> Amarillo -> Naranja -> Rojo
                            if ratio < 0.2: # Cian a Azul
                                r, g, b = 0, int(255 * (1 - (ratio / 0.2))), 255
                            elif ratio < 0.4: # Azul a Verde
                                r, g, b = 0, int(255 * ((ratio - 0.2) / 0.2)), int(255 * (1 - ((ratio - 0.2) / 0.2)))
                            elif ratio < 0.6: # Verde a Amarillo
                                r, g, b = int(255 * ((ratio - 0.4) / 0.2)), 255, 0
                            elif ratio < 0.8: # Amarillo a Naranja
                                r, g, b = 255, int(255 - (90 * ((ratio - 0.6) / 0.2))), 0
                            else: # Naranja a Rojo
                                r, g, b = 255, int(165 * (1 - ((ratio - 0.8) / 0.2))), 0
                            return [r, g, b, alpha]

                        df_2d['color_2d'] = df_2d['Pasajeros'].apply(get_color_2d)
                        
                        # 2. Tamaño de Iconos (Aumentado)
                        # Aseguramos un tamaño base mínimo (ej. 5) + escalado
                        df_2d['size_icon'] = df_2d['grosor_final'].apply(lambda x: max(x, 5))

                        # --- LEYENDA DE COLORES ---
                        def generar_leyenda_html(max_val):
                            color_stops = "cyan, blue, lime, yellow, orange, red"
                            html_string = f"""
                            <div style="
                                font-family: 'Source Sans Pro', sans-serif; 
                                font-size: 0.8rem; 
                                color: #31333F;
                                margin-bottom: 10px;
                                border: 1px solid #EAEAEA;
                                padding: 5px 10px;
                                border-radius: 5px;
                            ">
                                <strong>Leyenda de Pasajeros</strong>
                                <div style="display: flex; align-items: center; justify-content: space-between;">
                                    <span>1</span>
                                    <div style="
                                        background: linear-gradient(to right, {color_stops});
                                        flex-grow: 1; margin: 0 10px;
                                        height: 15px; border: 1px solid #CCC; border-radius: 5px;
                                    "></div>
                                    <span>{int(max_val)}</span>
                                </div>
                                <div style="text-align: center; font-size: 0.7rem; color: #555;">
                                    (El grosor y la opacidad también aumentan con la cantidad)
                                </div>
                            </div>
                            """
                            return html_string

                        capas_2d = []

                        # AÑADIR RUTA DE REFERENCIA AL MAPA 2D
                        if not df_ruta.empty:
                            path_data_2d = pd.DataFrame({
                                'path': [df_ruta[['Longitud', 'Latitud']].values.tolist()]
                            })
                            capas_2d.append(pdk.Layer(
                                'PathLayer',
                                data=path_data_2d,
                                get_path='path',
                                get_color=[100, 100, 100, 120], # Gris para no competir
                                get_width=15,
                                width_min_pixels=2,
                                id='ruta_referencia_layer_2d'
                            ))
                        
                        # Capa de Líneas (vectores principales)
                        capas_2d.append(pdk.Layer(
                            "LineLayer",
                            df_2d,
                            get_source_position=["lon_ori", "lat_ori"],
                            get_target_position=["lon_des", "lat_des"],
                            get_color="color_2d", 
                            get_width="grosor_final",
                            pickable=True,
                            auto_highlight=True,
                        ))

                        def crear_puntas_de_flecha(df):
                            lats_ori = df['lat_ori'].values
                            lons_ori = df['lon_ori'].values
                            lats_des = df['lat_des'].values
                            lons_des = df['lon_des'].values
                            
                            dx = lons_des - lons_ori
                            dy = lats_des - lats_ori
                            norm = np.sqrt(dx**2 + dy**2)
                            norm[norm == 0] = 1 # Evitar división por cero
                            
                            # El tamaño de la flecha es 3% del largo del vector, con un mínimo y máximo para que no se dispare
                            # Estos valores (0.0002, 0.002) están en grados de lat/lon y funcionan bien para escalas de ciudad.
                            arrow_size = np.clip(norm * 0.03, 0.0002, 0.002)
                            
                            # Vectores unitarios de la dirección del flujo
                            ux = dx / norm
                            uy = dy / norm
                            
                            # Vector perpendicular para las "alas" de la flecha
                            px = -uy
                            py = ux
                            
                            # Calcular los dos puntos que forman la cabeza de la flecha
                            punta1_lon = lons_des - ux * arrow_size + px * arrow_size * 0.6
                            punta1_lat = lats_des - uy * arrow_size + py * arrow_size * 0.6
                            
                            punta2_lon = lons_des - ux * arrow_size - px * arrow_size * 0.6
                            punta2_lat = lats_des - uy * arrow_size - py * arrow_size * 0.6
                            
                            # Crear un nuevo DataFrame con los segmentos de línea para las flechas
                            return pd.DataFrame({
                                "start_lon": np.concatenate([lons_des, lons_des]),
                                "start_lat": np.concatenate([lats_des, lats_des]),
                                "end_lon": np.concatenate([punta1_lon, punta2_lon]),
                                "end_lat": np.concatenate([punta1_lat, punta2_lat]),
                                "color": np.concatenate([df['color_2d'].values, df['color_2d'].values]),
                                "width": np.concatenate([df['grosor_final'].values, df['grosor_final'].values]) # Ancho proporcional
                            })

                        df_puntas = crear_puntas_de_flecha(df_2d)
                        
                        # Capa para las puntas de flecha
                        capas_2d.append(pdk.Layer(
                            "LineLayer",
                            df_puntas,
                            get_source_position=["start_lon", "start_lat"],
                            get_target_position=["end_lon", "end_lat"],
                            get_color="color",
                            get_width="width",
                            id="arrow_heads_layer"
                        ))
                        
                        # 3. Vista cenital (desde arriba) y bloqueada
                        view_state_2d = pdk.ViewState(
                            latitude=st.session_state.view_state.latitude, longitude=st.session_state.view_state.longitude,
                            zoom=st.session_state.view_state.zoom, 
                            pitch=0, 
                            bearing=0,
                            max_pitch=0, # Bloquea la inclinación
                            min_pitch=0  # Bloquea la inclinación
                        )
                        
                        st.subheader(f"Vectores de flujo en 2D (Origen-Destino)")
                        st.pydeck_chart(
                            pdk.Deck(
                                map_provider="carto", map_style="light", 
                                initial_view_state=view_state_2d,
                                layers=capas_2d, 
                                tooltip={"html": "<b>Pasajeros:</b> {Pasajeros}"},
                                height=900
                            ), key="deck_map_2d"
                        )
                        
                        # Mostrar leyenda debajo del mapa
                        st.markdown(generar_leyenda_html(max_p_2d), unsafe_allow_html=True)
                    else:
                        st.info("No hay datos de flujos para mostrar en la vista 2D.")

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