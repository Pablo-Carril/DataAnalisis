import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import os

from spatial_utils import calcular_distancia_traza_vectorizado
from data_processing import (cargar_datos, cargar_informacion_paradas, calcular_vectores_flujo, 
                             agrupar_por_zonas, calcular_estadisticas_nodos, cargar_recorridos_todos, procesar_ruta_filtrada)

# Inyectar CSS para ocultar el menú 
css_style = """
        <style>
       /* #MainMenu {visibility: hidden;} */
       /* header {visibility: hidden;} */
        footer {visibility: hidden;}
        /* Quitar el espacio superior del contenedor principal */
            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 0rem;
                padding-left: 3rem;
                padding-right: 3rem;
            }
            /* FORZAR ALTURA DEL MAPA */
            div[data-testid="stDeckGlJsonChart"] {
                height: 750px !important;
            }
            /* Modificamos el botón de la pestaña */
            div[data-baseweb="tab-list"] button div {
                font-size: 20px !important;
                font-weight: 600 !important;
            }
            
            /* Simulamos alejar zoom */
            .block-container {
                transform: scale(0.85);
                /*transform-origin: top left;*/
                width: 118%;
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
#st.markdown("""
#Mapa de Deseo. Esta herramienta agrupa viajes cercanos para visualizar los **corredores de mayor demanda**.
#""")

# --- 4. INTERFAZ DE USUARIO ---
#archivo_subido = st.sidebar.file_uploader("Cargar archivo Parquet", type=["parquet"])

st.sidebar.header("Selección de Línea")
linea_seleccionada = st.sidebar.selectbox("Seleccionar Línea", ["Línea 98", "Línea 85"], index=0)

if linea_seleccionada == "Línea 85":
    archivo_subido = "Transacciones saes octubre.parquet"
else:
    archivo_subido = "Transacciones expreso octubre.parquet"

if archivo_subido:
    df_raw = cargar_datos(archivo_subido)
    
    st.sidebar.header("Filtros de Datos")
    fechas_disponibles = sorted(df_raw['Fecha'].unique())
    opciones_fecha = ["Todo el mes"] + [str(f) for f in fechas_disponibles]
    fecha_sel = st.sidebar.selectbox("Seleccionar Período", opciones_fecha)

    ramales_unicos = sorted(df_raw['Ramal'].unique().tolist())
    ramales = ["Todos"] + ramales_unicos
    default_index = 1 if ramales_unicos else 0
    ramal_sel = st.sidebar.selectbox("Seleccionar Ramal", ramales, index=default_index)
    sentido_sel = st.sidebar.radio("Sentido de Subida", ["Ambos", "Ida", "Vuelta"], index=1, key="radio_s")

    st.sidebar.markdown("---")
    # Nuevo control para seleccionar el tipo de agrupación - Por Distancia o por Clusters DBSCAN
    criterio_agrupacion = st.sidebar.radio("Agrupar Por:", ["Distancia", "Clusters", "KDE", "Por Parada", "Por Sección"], index=0, key="criterio_agrupacion")
    
    # Ajustamos las opciones del slider según el criterio
    label_slider = "Tamaño"
    val_default = 100
    kde_mode = "Unidos" # Valor por defecto para evitar NameError
    n_secciones_agrupar = 1 # Inicialización para evitar errores
    n_paradas_agrupar = 1
    if criterio_agrupacion == 'Distancia':
        label_slider = "Agrupación (mts):"
        metros_sel = st.sidebar.select_slider(f"{label_slider}", options=[100, 150, 200, 300, 400, 500, 800, 1000, 1500,3100], value=val_default)
    elif criterio_agrupacion == 'KDE':
        label_slider = "Area (mts):"
        val_default = 200
        metros_sel = st.sidebar.select_slider(f"{label_slider}", options=[100, 150, 160, 170, 180, 190, 200, 250, 300, 400, 500, 800, 1000, 1500], value=val_default)
        kde_mode = st.sidebar.radio(
            "Modo Detección KDE:", ["Unidos", "Separados"], index=0, 
            help="**Unidos**: Detecta 'hubs' de actividad general (subidas+bajadas). **Separados**: Detecta hubs de subida y bajada de forma independiente y luego los combina."
        )
    elif criterio_agrupacion == 'Por Parada':
        n_paradas_agrupar = st.sidebar.slider("Paradas a agrupar:", 1, 10, value=1)
        metros_sel = 100 # Valor dummy para evitar errores
    elif criterio_agrupacion == 'Por Sección':
        n_secciones_agrupar = 1
        metros_sel = 100 
    
    else:
        label_slider = "Radio del Cluster (mts):"
        val_default = 200
        metros_sel = st.sidebar.select_slider(f"{label_slider}", options=[100, 150, 170, 180, 190, 200, 210, 220, 230, 240, 260, 300], value=val_default)

    st.sidebar.markdown("---")
    # Nuevo Toggle para activar la clasificación por rangos en cualquier modo
    color_por_rango = st.sidebar.checkbox("Clasificar por Rangos de Demanda", value=False, help="Activa la paleta cian-violeta y permite filtrar por nivel de carga.")
    
    rango_demanda = "Todos"
    if color_por_rango:
        rango_demanda = st.sidebar.selectbox(
            "Nivel a Visualizar:",
            ["Todos", "Muy Baja (0-20%)", "Baja (20-40%)", "Media (40-60%)", "Alta (60-80%)", "Muy Alta (80-100%)"],
            index=0
        )
    
    #st.sidebar.markdown("---")
    # Botón para resetear la vista del mapa
    #if st.sidebar.button("📍 Resetear Vista del Mapa"):
       # st.session_state.view_state = None
        # El script se re-ejecutará naturalmente, aplicando el reseteo.

    #st.sidebar.header("Filtros de Visualización")
    min_pasajeros = st.sidebar.number_input("Ocultar flujos menores a:", 1, 1000, value=1, key="num_i")
    
    # Consolidación de filtros para evitar DuplicateElementId y NameError
    dist_rango = st.sidebar.slider("Filtrar por Distancia (km)", 0.0, 50.0, (0.0, 50.0), step=0.5, key="slider_dist")

    hora_rango = st.sidebar.slider("Rango Horario (Subida)", 0, 23, (0, 23), key="slider_h")
    
    mostrar_puntos = st.sidebar.toggle("Mostrar Puntos Originales", value=False, key="toggle_ptos")
    ocultar_retrocesos = st.sidebar.checkbox("Ocultar retrocesos (Ida < 0km)", value=True)

    # Aplicación de filtros a datos base
    df_filtrado = df_raw.copy()
    if fecha_sel != "Todo el mes":
        fecha_obj = pd.to_datetime(fecha_sel).date()
        df_filtrado = df_filtrado[df_filtrado['Fecha'] == fecha_obj]
    if ramal_sel != "Todos":
        df_filtrado = df_filtrado[df_filtrado['Ramal'] == ramal_sel]
        
    # --- CARGA Y PREPARACIÓN DE PARADAS Y RECORRIDOS ---
    df_paradas_all, dict_ramales = cargar_informacion_paradas()
    df_recorridos_all = cargar_recorridos_todos()
    
    df_paradas_sel = pd.DataFrame()
    df_ruta = pd.DataFrame()
    
    if ramal_sel != "Todos" and sentido_sel != "Ambos":
        if ramal_sel in dict_ramales:
            codigo_buscado = dict_ramales[ramal_sel]
            
            # Filtro Paradas
            df_paradas_sel = df_paradas_all[df_paradas_all['Ramal_Cod'] == codigo_buscado].copy()
            df_paradas_sel = df_paradas_sel[df_paradas_sel['Sentido'] == sentido_sel]
            
            # Filtro Recorridos
            if not df_recorridos_all.empty:
                df_ruta_sel = df_recorridos_all[(df_recorridos_all['Ramal_Cod'] == codigo_buscado) & (df_recorridos_all['Sentido'] == sentido_sel)].copy()
                if not df_ruta_sel.empty:
                    df_ruta = procesar_ruta_filtrada(df_ruta_sel)
                else:
                    st.sidebar.warning(f"No se encontraron recorridos unificados para el Ramal '{ramal_sel}' y Sentido '{sentido_sel}'.")
            else:
                st.sidebar.error("No se pudo cargar el archivo unificado de recorridos (Recorridos_Todos_con_punto.csv).")
        else:
            st.sidebar.warning(f"El ramal '{ramal_sel}' no se encuentra en Ramales.csv")
    elif ramal_sel == "Todos" or sentido_sel == "Ambos":
        st.sidebar.info("Seleccione un Ramal y Sentido para visualizar la ruta y agrupar los flujos sobre ella.")
        
    if not df_paradas_sel.empty:
        df_paradas_sel['Seccion'] = pd.to_numeric(df_paradas_sel['Seccion'], errors='coerce')
        df_paradas_sel = df_paradas_sel.dropna(subset=['Seccion'])

    # Sincronizar paradas con la ruta para obtener Km_Posicion
    if not df_ruta.empty and not df_paradas_sel.empty:
        r_lats, r_lons, r_cum = df_ruta['Latitud'].values, df_ruta['Longitud'].values, df_ruta['Dist_Acum'].values
        p_lats, p_lons = df_paradas_sel['Latitud'].values, df_paradas_sel['Longitud'].values
        d_sq = (p_lats[:, None] - r_lats[None, :])**2 + (p_lons[:, None] - r_lons[None, :])**2
        df_paradas_sel['Km_Posicion'] = r_cum[np.argmin(d_sq, axis=1)]

    with st.spinner('Procesando vectores de flujo...'):
        # Pasamos df_paradas_sel para que el proyecto haga snap a ellas desde la inferencia de destino
        df_flujos = calcular_vectores_flujo(df_filtrado, df_ruta=df_ruta, df_paradas=df_paradas_sel)

    # --- LÓGICA DE VISTA DE MAPA ESTABLE ---
    # Se define qué filtros fuerzan un reseteo del centro del mapa.
    current_major_filters = (archivo_subido, fecha_sel, ramal_sel)

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

    layer_etiquetas_seccion = None
    if not df_flujos.empty:
        if not df_paradas_sel.empty:
            # --- CAPA DE ETIQUETAS DE SECCIÓN ---
            layer_etiquetas_seccion = None
            if not df_paradas_sel.empty and 'Seccion' in df_paradas_sel.columns:
                # Si estamos agrupando por sección, las etiquetas deben coincidir con los grupos
                df_labels = df_paradas_sel.dropna(subset=['Seccion']).copy()
                if criterio_agrupacion == "Por Sección":
                    df_labels['Seccion'] = (df_labels['Seccion'] // n_secciones_agrupar) * n_secciones_agrupar
                
                df_secciones_label = df_labels.sort_values('Orden').groupby('Seccion').first().reset_index()
                df_secciones_label['Seccion_Str'] = df_secciones_label['Seccion'].astype(int).astype(str)
                layer_etiquetas_seccion = pdk.Layer(
                    "TextLayer",
                    df_secciones_label,
                    get_position=["Longitud", "Latitud"],
                    get_text="Seccion_Str",
                    get_size=22,
                    get_color=[255, 255, 255],
                    get_alignment_baseline="'bottom'",
                    background_color=[0, 0, 0, 160],
                    id="section_labels_layer"
                )

        # Filtros de Mapa aplicados sobre los vectores
        # Optimización: Filtrado directo sin copia inicial
        mask_hora = (df_flujos['Hora_Int'] >= hora_rango[0]) & (df_flujos['Hora_Int'] <= hora_rango[1])
        df_mapa = df_flujos[mask_hora]
        
        if sentido_sel != "Ambos":
            df_mapa = df_mapa[df_mapa['Sentido'] == sentido_sel]

        if not df_mapa.empty:
            capas = []
            
            # Determine parameters to pass to agrupar_por_zonas based on criterion
            n_param_for_grouping = 1 # Default value for n_paradas argument
            if criterio_agrupacion == 'Por Parada':
                n_param_for_grouping = n_paradas_agrupar
            elif criterio_agrupacion == 'Por Sección':
                n_param_for_grouping = n_secciones_agrupar # Use n_secciones_agrupar for n_paradas argument

            df_zonas = agrupar_por_zonas(df_mapa, df_ruta, metros_sel, criterio_agrupacion,
                                         kde_mode=kde_mode, df_paradas=df_paradas_sel, n_paradas=n_param_for_grouping)
            
            # Verificamos que se hayan generado zonas antes de filtrar para evitar KeyError
            if not df_zonas.empty and 'Pasajeros' in df_zonas.columns:
                df_zonas = df_zonas[df_zonas['Pasajeros'] >= min_pasajeros].reset_index(drop=True)

            # --- FILTRO POR RANGO DE DEMANDA ---
            # Se aplica para que el selector 'Nivel a Visualizar' tenga efecto en el mapa y métricas.
            if color_por_rango and rango_demanda != "Todos" and not df_zonas.empty:
                max_p_ref_filter = df_zonas['Pasajeros'].max()
                if max_p_ref_filter > 0:
                    df_zonas['ratio_tmp'] = df_zonas['Pasajeros'] / max_p_ref_filter
                    if rango_demanda == "Muy Baja (0-20%)":
                        df_zonas = df_zonas[df_zonas['ratio_tmp'] < 0.2]
                    elif rango_demanda == "Baja (20-40%)":
                        df_zonas = df_zonas[(df_zonas['ratio_tmp'] >= 0.2) & (df_zonas['ratio_tmp'] < 0.4)]
                    elif rango_demanda == "Media (40-60%)":
                        df_zonas = df_zonas[(df_zonas['ratio_tmp'] >= 0.4) & (df_zonas['ratio_tmp'] < 0.6)]
                    elif rango_demanda == "Alta (60-80%)":
                        df_zonas = df_zonas[(df_zonas['ratio_tmp'] >= 0.6) & (df_zonas['ratio_tmp'] < 0.8)]
                    elif rango_demanda == "Muy Alta (80-100%)":
                        df_zonas = df_zonas[df_zonas['ratio_tmp'] >= 0.8]
                    
                    if 'ratio_tmp' in df_zonas.columns:
                        df_zonas = df_zonas.drop(columns=['ratio_tmp'])
                    df_zonas = df_zonas.reset_index(drop=True)

            if not df_ruta.empty:
                path_data = pd.DataFrame({ # This is for the reference route path
                    'path': [df_ruta[['Longitud', 'Latitud']].values.tolist()]
                })
                capas.append(pdk.Layer(
                    'PathLayer',
                    data=path_data,
                    get_path='path',
                    get_color=[250, 40, 40, 160], # color
                    get_width=15,
                    width_min_pixels=3,
                    id='ruta_referencia_layer'
                ))
            
            # Añadir etiquetas de sección si están disponibles
            if layer_etiquetas_seccion:
                capas.append(layer_etiquetas_seccion)

            # --- CÁLCULO DE DISTANCIAS Y FILTRO ---
            if not df_ruta.empty and not df_zonas.empty:
                # Calculamos distancia con signo (positiva = avanza, negativa = retrocede)
                dist_signed = calcular_distancia_traza_vectorizado(
                    df_zonas['lat_ori'].values, df_zonas['lon_ori'].values,
                    df_zonas['lat_des'].values, df_zonas['lon_des'].values,
                    df_ruta
                )
                df_zonas['Km_Recorridos'] = np.abs(dist_signed)
                df_zonas['Diff_Km'] = dist_signed

                # Filtro de retrocesos (si el usuario lo activa)
                if ocultar_retrocesos:
                    df_zonas = df_zonas[df_zonas['Diff_Km'] >= 0].reset_index(drop=True)

                # Aplicar filtro de distancia del slider
                df_zonas = df_zonas[
                    (df_zonas['Km_Recorridos'] >= dist_rango[0]) &
                    (df_zonas['Km_Recorridos'] <= dist_rango[1])
                ].reset_index(drop=True)
                
                # Recalcular métricas con los datos ya filtrados por distancia
                total_pax = df_zonas['Pasajeros'].sum()
                if total_pax > 0:
                    total_km_recorridos = (df_zonas['Km_Recorridos'] * df_zonas['Pasajeros']).sum()
                    dist_media = total_km_recorridos / total_pax
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    with col2:
                        st.metric("Distancia Media (km): ", f"{dist_media:.2f}")
                    with col1:
                        st.metric(f"Ramal: ", f"{ramal_sel} - {sentido_sel}")
                    with col6:
                         with st.expander("❓ Ayuda"):
                            st.markdown("""
                                        **Uso de la aplicación**

                                        1. Seleccione el **Ramal** en el panel izquierdo  
                                        2. Modifique el **Tamaño de Agrupación** para agrupar pasajeros en Flujos.
                                        (100 metros: simula paradas - más grandes: simula barrios)
                                        3. Ajuste **Ocultar flujos menores a** para poder ver flujos ocultos.
                                        4. Ajuste el **rango horario**  
                                        5. Use las Pestañas **Vista 3D** y **Vista 2D**
                                        """)
                    with col3:
                        cant_flujos = len(df_zonas)
                        st.metric("Cantidad de Flujos", f"{cant_flujos:,}")
                    with col4:
                        st.metric("Pasajeros Totales", f"{total_pax:,}")
                    with col5:
                        st.metric("Kilómetros Totales", f"{total_km_recorridos:,.0f}")

            # Calcular estadísticas de nodos (Subidas/Bajadas)
            # Pass df_ruta to calcular_estadisticas_nodos to enable Km_Posicion calculation
            df_nodos = calcular_estadisticas_nodos(df_zonas)
            
            # Calculate Km_Posicion for nodes here, as df_ruta is available
            if not df_nodos.empty and not df_ruta.empty:
                ruta_lats = df_ruta['Latitud'].values
                ruta_lons = df_ruta['Longitud'].values
                ruta_cum = df_ruta['Dist_Acum'].values
                node_lats = df_nodos['lat'].values
                node_lons = df_nodos['lon'].values
                dists_sq = (node_lats[:, None] - ruta_lats[None, :])**2 + (node_lons[:, None] - ruta_lons[None, :])**2
                min_idx = np.argmin(dists_sq, axis=1)
                df_nodos['Km_Posicion'] = ruta_cum[min_idx]
            
            # 0. Capa de Grilla (Fondo)
            
            # 1. Capa de Arcos (Flujos)
            if not df_zonas.empty:
                max_p_display = int(df_zonas['Pasajeros'].max())
                log_max_p = np.log1p(max_p_display) if max_p_display > 0 else 1

                # --- FUNCIÓN DE COLOR COMPARTIDA (Consolidada) ---
                def get_color_shared(p):
                    ratio = p / max_p_display if max_p_display > 0 else 0
                    alpha = int(40 + (215 * ratio))
                    
                    if color_por_rango:
                        # 5 Grupos discretos: Cian -> Azul Claro -> Azul -> Púrpura -> Violeta
                        if ratio < 0.2: return [0, 255, 255, alpha]   # Cian
                        elif ratio < 0.4: return [0, 128, 255, alpha] # Azul Claro
                        elif ratio < 0.6: return [0, 0, 255, alpha]   # Azul
                        elif ratio < 0.8: return [138, 43, 226, alpha] # AzulVioleta
                        else: return [148, 0, 211, alpha]             # Violeta Oscuro

                    # Gradiente continuo original
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

                def color_log(x):
                    if color_por_rango: return get_color_shared(x)
                    ratio = np.log1p(x) / log_max_p if log_max_p > 0 else 0
                    return [255, int(165 * (1 - ratio)), 0, 200]

                df_zonas['color_ori'] = df_zonas['Pasajeros'].apply(color_log)
                
                # ANCHO DE LÍNEA LOGARÍTMICO: Más sutil y no satura al alejar el zoom.
                df_zonas['grosor_final'] = (1 + (np.log1p(df_zonas['Pasajeros']) / log_max_p) * 14) if log_max_p > 0 else 1

                # Campos vacíos para tooltip consistente
                df_zonas['Subieron'] = ""
                df_zonas['Bajaron'] = ""
                
                # Change to store float instead of formatted string
                # Calcular porcentaje del flujo respecto al total visible
                total_visible = df_zonas['Pasajeros'].sum()
                if total_visible > 0:
                    df_zonas['Porcentaje'] = (df_zonas['Pasajeros'] / total_visible) * 100
                else:
                    df_zonas['Porcentaje'] = 0.0

                # Columnas de texto para tooltip consistente sin errores de formato
                df_zonas['Porcentaje_Actividad_Str'] = df_zonas['Porcentaje'].map('{:.1f}%'.format)
                df_zonas['Porcentaje_Subieron_Str'] = ""
                df_zonas['Porcentaje_Bajaron_Str'] = ""
                df_zonas['Distancia_Str'] = df_zonas['Km_Recorridos'].map('{:.2f} km'.format) if 'Km_Recorridos' in df_zonas.columns else "N/A"

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
                    get_color=[140, 140, 140, 50], # Gris
                    get_radius=20,
                ))

            # Puntos agrupados con estadísticas
            if not df_nodos.empty:
                # Capa de Paradas (Puntos Azules) para Tab 1
                if not df_paradas_sel.empty:
                    df_p_3d = df_paradas_sel.copy()
                    # Preparar campos vacíos para evitar errores en el tooltip global
                    for c in ['Pasajeros','Subieron','Bajaron','Porcentaje_Actividad_Str','Porcentaje_Subieron_Str','Porcentaje_Bajaron_Str','Distancia_Str']:
                        df_p_3d[c] = ""
                    df_p_3d['Pasajeros'] = "Parada: " + df_p_3d['Nombre parada']
                    capas.append(pdk.Layer("ScatterplotLayer", df_p_3d, get_position=["Longitud", "Latitud"],
                                         get_color=[0, 100, 255, 200], get_radius=20, pickable=True, id="stops_3d"))

                # Aseguramos que 'Pasajeros' esté vacío para el tooltip de nodos,
                # ya que los nodos tienen 'Subieron' y 'Bajaron'
                df_nodos['Pasajeros'] = ""
                
                # Columnas de texto pre-formateadas para el tooltip
                df_nodos['Porcentaje_Actividad_Str'] = df_nodos['Porcentaje_Actividad'].map('{:.1f}%'.format)
                df_nodos['Porcentaje_Subieron_Str'] = df_nodos['Porcentaje_Subieron'].map('{:.1f}%'.format)
                df_nodos['Porcentaje_Bajaron_Str'] = df_nodos['Porcentaje_Bajaron'].map('{:.1f}%'.format)
                df_nodos['Distancia_Str'] = ""
                
                # 'Porcentaje' ya se calcula dentro de calcular_estadisticas_nodos
                # 'Subieron' y 'Bajaron' ya están en df_nodos

                # --- RADIO DINÁMICO PARA NODOS 3D ---
                max_act_3d = df_nodos['Total_Actividad'].max()
                if max_act_3d > 0:
                    # Radio dinámico: Mínimo 20m, Máximo 250m (según actividad)
                    df_nodos['radius_3d'] = 20 + (df_nodos['Total_Actividad'] / max_act_3d) * 230
                else:
                    df_nodos['radius_3d'] = 20

                # --- COLOR DINÁMICO DE NODOS (Subida vs Bajada) ---
                def get_node_color(row):
                    # Si bajan más que suben -> Amarillo Claro, sino Verde
                    if row['Bajaron'] > row['Subieron']:
                        return [255, 235, 60, 120] # Amarillo Claro
                    return [20, 150, 0, 100] # Verde

                df_nodos['color_nodo'] = df_nodos.apply(get_node_color, axis=1)

                capas.append(pdk.Layer(
                    "ScatterplotLayer",
                    df_nodos,
                    get_position=["lon", "lat"],
                    get_color="color_nodo",
                    get_radius='radius_3d',
                    pickable=True,
                ))

            # 3. Renderizado del mapa si hay capas que mostrar
            if capas:
                # --- CREACIÓN DE PESTAÑAS DE VISUALIZACIÓN ---
                tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Mapa 3D (Arcos)", "🛰️ Vista 2D (Vectores)", "📈 Vista Lineal 1D", "📊 Matriz OD"])

                with tab1:
                    
                    st.pydeck_chart(pdk.Deck(
                        map_provider="carto",
                        map_style="road",
                        initial_view_state=st.session_state.view_state, # Usamos SIEMPRE la vista guardada
                        layers=capas,
                        tooltip={
                            "html": "<b>Pasajeros:</b> {Pasajeros}<br/>"
                                    "<b>Distancia:</b> {Distancia_Str}<br/>"
                                    "<b>Subieron:</b> {Subieron}<br/>"
                                    "<b>Bajaron:</b> {Bajaron}<br/>"
                                    "<b>% Actividad:</b> {Porcentaje_Actividad_Str}<br/>"
                                    "<b>% Subieron:</b> {Porcentaje_Subieron_Str}<br/>"
                                    "<b>% Bajaron:</b> {Porcentaje_Bajaron_Str}"
                        },
                        # height=700 se maneja por CSS ahora, pero dejamos un valor base
                    ), key="deck_map_3d", use_container_width=True)
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write(" **Arcos**: Suben (Naranja/Rojo) -> Bajan (Azul) | **Nodos**: Verde (Subida) - Amarillo (Bajada)")
                
                with tab2:
                    if not df_zonas.empty:
                        # --- PREPARACIÓN PARA VISTA 2D ---
                        # Se crea una copia para la Tab 2, se ordena para el Z-Index y se trabaja sobre ella.
                        # Esto evita modificar el df_zonas original que se usa en la Tab 1 y en la exportación.
                        df_2d = df_zonas.copy().sort_values('Pasajeros', ascending=True).reset_index(drop=True)
                        
                        # 1. Cálculo de Colores (Gradiente de Matiz y Transparencia)
                        max_p_2d = df_2d['Pasajeros'].max()
                        
                        df_2d['color_2d'] = df_2d['Pasajeros'].apply(get_color_shared)
                        
                        # 2. Tamaño de Iconos (Aumentado)
                        # Aseguramos un tamaño base mínimo (ej. 5) + escalado
                        df_2d['size_icon'] = df_2d['grosor_final'].apply(lambda x: max(x, 5))

                        # --- LEYENDA DE COLORES ---
                        def generar_leyenda_html(max_val):
                            if color_por_rango:
                                step = max_val / 5
                                html_string = f"""
                                <div style="font-family: 'Source Sans Pro', sans-serif; font-size: 0.8rem; color: #FFF; margin-bottom: 10px; border: 1px solid #EAEAEA; padding: 5px 10px; border-radius: 5px;">
                                    <strong style="color: #FFF;">Rangos de Pasajeros (Cian a Violeta):</strong>
                                    <div style="display: flex; gap: 5px; margin-top: 5px;">
                                        <div style="flex: 1; background: #00FFFF; height: 12px; border-radius: 2px;"></div>
                                        <div style="flex: 1; background: #0080FF; height: 12px; border-radius: 2px;"></div>
                                        <div style="flex: 1; background: #0000FF; height: 12px; border-radius: 2px;"></div>
                                        <div style="flex: 1; background: #8A2BE2; height: 12px; border-radius: 2px;"></div>
                                        <div style="flex: 1; background: #9400D3; height: 12px; border-radius: 2px;"></div>
                                    </div>
                                    <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: #AAA; margin-top: 2px;">
                                        <span>0</span><span>{int(step)}</span><span>{int(step*2)}</span><span>{int(step*3)}</span><span>{int(step*4)}</span><span>{int(max_val)}</span>
                                    </div>
                                </div>"""
                                return html_string

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
                                <strong style="color: #FFF;">Leyenda de Pasajeros:</strong>
                                <div style="display: flex; align-items: center; justify-content: space-between;">
                                    <span>1</span>
                                    <div style="
                                        background: linear-gradient(to right, {color_stops});
                                        flex-grow: 1; margin: 0 10px;
                                        height: 15px; border: 1px solid #CCC; border-radius: 5px;
                                    "></div>
                                    <span style="color: #FFF;">{int(max_val)}</span>
                                </div>
                                <div style="text-align: center; font-size: 0.7rem; color: #AAA;">
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
                                get_color=[250, 100, 100, 120], # Gris para no competir?
                                get_width=15,
                                width_min_pixels=2,
                                id='ruta_referencia_layer_2d'
                            ))
                        
                        if layer_etiquetas_seccion:
                            capas_2d.append(layer_etiquetas_seccion)
                        
                        # Capa de Líneas (vectores principales)
                        if not df_paradas_sel.empty:
                            df_p_2d = df_paradas_sel.copy()
                            for c in ['Pasajeros','Subieron','Bajaron','Porcentaje_Actividad_Str','Porcentaje_Subieron_Str','Porcentaje_Bajaron_Str','Distancia_Str']:
                                df_p_2d[c] = ""
                            df_p_2d['Pasajeros'] = "Parada: " + df_p_2d['Nombre parada']
                            capas_2d.append(pdk.Layer("ScatterplotLayer", df_p_2d, get_position=["Longitud", "Latitud"],
                                                  get_color=[0, 100, 255, 200], get_radius=20, pickable=True, id="stops_2d"))

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
                            arrow_size = np.clip(norm * 0.02, 0.0002, 0.002)
                            
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
                                "width": np.concatenate([df['grosor_final'].values, df['grosor_final'].values]), # Ancho proporcional
                                # Datos extra para el tooltip
                                "Pasajeros": np.concatenate([df['Pasajeros'].values, df['Pasajeros'].values]),
                                "Subieron": np.concatenate([df['Subieron'].values, df['Subieron'].values]),
                                "Bajaron": np.concatenate([df['Bajaron'].values, df['Bajaron'].values]),
                                "Porcentaje_Actividad_Str": np.concatenate([df['Porcentaje_Actividad_Str'].values, df['Porcentaje_Actividad_Str'].values]),
                                "Porcentaje_Subieron_Str": np.concatenate([df['Porcentaje_Subieron_Str'].values, df['Porcentaje_Subieron_Str'].values]),
                                "Porcentaje_Bajaron_Str": np.concatenate([df['Porcentaje_Bajaron_Str'].values, df['Porcentaje_Bajaron_Str'].values]),
                                "Distancia_Str": np.concatenate([df['Distancia_Str'].values, df['Distancia_Str'].values])
                            })

                        df_puntas = crear_puntas_de_flecha(df_2d)
                        
                        # Capa para las puntas de flecha
                        capas_2d.append(pdk.Layer(
                            "LineLayer",
                            df_puntas,
                            get_source_position=["start_lon", "start_lat"],
                            get_target_position=["end_lon", "end_lat"],
                            get_color="color",
                            get_width= 4,  #"width",
                            id="arrow_heads_layer"
                        ))
                        
                        # --- PUNTOS ESTADÍSTICOS (NODOS) EN 2D CON RADIO DINÁMICO ---
                        if not df_nodos.empty:
                            df_nodos_2d = df_nodos.copy()
                            max_act = df_nodos_2d['Total_Actividad'].max()
                            
                            # Radio dinámico: Mínimo 20m, Máximo 250m (según actividad)
                            if max_act > 0:
                                df_nodos_2d['radius_2d'] = 20 + (df_nodos_2d['Total_Actividad'] / max_act) * 230
                            else:
                                df_nodos_2d['radius_2d'] = 20 # Use color_nodo for 2D nodes as well

                            capas_2d.append(pdk.Layer(
                                "ScatterplotLayer",
                                df_nodos_2d,
                                get_position=["lon", "lat"],
                                get_color="color_nodo",
                                get_radius="radius_2d",
                                pickable=True,
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
                        
                        # st.subheader(f"Vectores de flujo en 2D (Origen-Destino)")
                        st.pydeck_chart(
                            pdk.Deck(
                                map_provider="carto", map_style="road", 
                                initial_view_state=view_state_2d,
                                layers=capas_2d, 
                                tooltip={
                                    "html": "<b>Pasajeros:</b> {Pasajeros}<br/>"
                                            "<b>Distancia:</b> {Distancia_Str}<br/>"
                                            "<b>Subieron:</b> {Subieron}<br/>" # Add new percentages to tooltip
                                            "<b>Bajaron:</b> {Bajaron}<br/>"
                                            "<b>% Actividad:</b> {Porcentaje_Actividad_Str}<br/>"
                                            "<b>% Subieron:</b> {Porcentaje_Subieron_Str}<br/>"
                                            "<b>% Bajaron:</b> {Porcentaje_Bajaron_Str}"
                                },
                                # height=700 se maneja por CSS
                            ), key="deck_map_2d", use_container_width=True
                        )
                        
                        # Mostrar leyenda debajo del mapa
                        st.write("") # dejo un espacio
                        st.write("") # dejo un espacio
                        st.write("")
                        st.write("")
                        st.write("")
                        st.write("")
                        st.write("")
                        st.write("")
                        st.write("")
                        st.write("")
                        st.write("")
                        st.write("")
                        st.write("")
                        st.markdown(generar_leyenda_html(max_p_2d), unsafe_allow_html=True)
                    else:
                        st.info("No hay datos de flujos para mostrar en la vista 2D.")

                with tab3:
                    if not df_ruta.empty and not df_zonas.empty:
                        with st.spinner("Generando diagrama lineal de flujos..."):
                            # 1. Mapear coordenadas a Kilómetros de Ruta (Broadcasting)
                            # Esto nos da la posición exacta en el eje X para el gráfico 1D
                            ruta_lats = df_ruta['Latitud'].values
                            ruta_lons = df_ruta['Longitud'].values
                            ruta_cum = df_ruta['Dist_Acum'].values
                            
                            # Orígenes
                            dists_ori = (df_zonas['lat_ori'].values[:, None] - ruta_lats[None, :])**2 + (df_zonas['lon_ori'].values[:, None] - ruta_lons[None, :])**2
                            km_ori = ruta_cum[np.argmin(dists_ori, axis=1)]
                            
                            # Destinos
                            dists_des = (df_zonas['lat_des'].values[:, None] - ruta_lats[None, :])**2 + (df_zonas['lon_des'].values[:, None] - ruta_lons[None, :])**2
                            km_des = ruta_cum[np.argmin(dists_des, axis=1)]

                            df_1d = df_zonas.copy()
                            df_1d['km_ori'] = km_ori
                            df_1d['km_des'] = km_des
                            df_1d['dist_abs'] = (df_1d['km_des'] - df_1d['km_ori']).abs()
                            
                            # Calcular mediana para separar cortos de largos
                            median_dist = df_1d['dist_abs'].median()
                            
                            # 2. Ordenar por Pasajeros (Los "más chicos" primero, para estar cerca de la línea)
                            df_1d = df_1d.sort_values('Pasajeros', ascending=True).reset_index(drop=True)
                            
                            # 3. Algoritmo de Apilamiento (Stacking) BIDIRECCIONAL
                            levels_top = []    # Para viajes cortos (Arriba)
                            levels_bottom = [] # Para viajes largos (Abajo)
                            row_levels = []
                            row_dirs = [] # 1 para arriba, -1 para abajo
                            
                            for idx, row in df_1d.iterrows():
                                s, e = min(row['km_ori'], row['km_des']), max(row['km_ori'], row['km_des'])
                                # Pequeño margen para evitar toques visuales entre líneas verticales
                                s, e = s - 0.05, e + 0.05
                                
                                # Decidir si va arriba o abajo
                                is_top = row['dist_abs'] <= median_dist
                                target_levels = levels_top if is_top else levels_bottom
                                direction = 1 if is_top else -1
                                
                                assigned_lvl = -1
                                # Buscamos el nivel más bajo disponible donde no choque
                                for i, lvl_intervals in enumerate(target_levels):
                                    collision = False
                                    for (occ_s, occ_e) in lvl_intervals:
                                        # Chequeo de intersección de intervalos
                                        if max(s, occ_s) < min(e, occ_e):
                                            collision = True
                                            break
                                    if not collision:
                                        lvl_intervals.append((s, e))
                                        assigned_lvl = i
                                        break
                                
                                if assigned_lvl == -1:
                                    assigned_lvl = len(target_levels)
                                    target_levels.append([(s, e)])
                                
                                row_levels.append(assigned_lvl)
                                row_dirs.append(direction)
                            
                            # --- LÓGICA DE COLOR CORREGIDA (Espectro Completo) ---
                            max_p_1d = df_1d['Pasajeros'].max()
                            def get_color_1d(p):
                                ratio = p / max_p_1d if max_p_1d > 0 else 0
                                alpha = 200
                                if ratio < 0.2: r, g, b = 0, int(255 * (1 - (ratio / 0.2))), 255
                                elif ratio < 0.4: r, g, b = 0, int(255 * ((ratio - 0.2) / 0.2)), int(255 * (1 - ((ratio - 0.2) / 0.2)))
                                elif ratio < 0.6: r, g, b = int(255 * ((ratio - 0.4) / 0.2)), 255, 0
                                elif ratio < 0.8: r, g, b = 255, int(255 - (90 * ((ratio - 0.6) / 0.2))), 0
                                else: r, g, b = 255, int(165 * (1 - ((ratio - 0.8) / 0.2))), 0
                                return [r, g, b, alpha]
                            
                            # 4. Construcción de Geometrías para PyDeck (Simulación de Coordenadas)
                            # Mapeamos: Km -> Longitud | Nivel -> Latitud
                            SCALE_X = 0.01   # 1 km = 0.01 grados Longitud
                            SCALE_Y = 0.0015 # 1 nivel = 0.0015 grados Latitud
                            BASE_LAT, BASE_LON = 0, 0
                            
                            # Detectar orientación geográfica de la ruta
                            lat_start = df_ruta['Latitud'].iloc[0]
                            lat_end = df_ruta['Latitud'].iloc[-1]
                            # Sur: Latitud disminuye (más negativa). Norte: Latitud aumenta (menos negativa).
                            is_southbound = lat_end < lat_start
                            max_km_ruta = ruta_cum.max()

                            paths_data = []
                            arrows_data = []

                            for i, row in df_1d.iterrows():
                                lvl = row_levels[i]
                                direction = row_dirs[i]
                                
                                # Altura basada en el nivel asignado (apilado)
                                y_base = BASE_LAT
                                y_top = BASE_LAT + (lvl + 1) * SCALE_Y * direction
                                
                                if is_southbound:
                                    # Sur: Izquierda a Derecha (Km 0 -> 0)
                                    x_ori = BASE_LON + row['km_ori'] * SCALE_X
                                    x_des = BASE_LON + row['km_des'] * SCALE_X
                                else:
                                    # Norte: Derecha a Izquierda (Km 0 -> Max)
                                    x_ori = BASE_LON + (max_km_ruta - row['km_ori']) * SCALE_X
                                    x_des = BASE_LON + (max_km_ruta - row['km_des']) * SCALE_X
                                
                                # --- GENERAR FORMA CUADRADA CON CURVA SUAVE ---
                                # Puntos clave
                                # Radio de giro (en grados)
                                r_x = 0.3 * SCALE_X # 300m radio X
                                r_y = abs(y_top - y_base) * 0.2 # 20% altura radio Y
                                
                                # Ajustar radio si el viaje es muy corto
                                dist_x = abs(x_des - x_ori)
                                if r_x * 2 > dist_x:
                                    r_x = dist_x / 2.5
                                
                                # Definimos la geometría base (bracket)
                                # p0(start) -> p1(sube) -> p2(curva) -> p3(curva_end) -> p4(baja) -> p5(end)
                                sign_dir = 1 if x_des > x_ori else -1
                                sign_y = 1 if y_top > y_base else -1 # 1 para Arriba, -1 para Abajo
                                
                                p0 = [x_ori, y_base]
                                p1 = [x_ori, y_top - (r_y * sign_y)]
                                p2 = [x_ori + (r_x * sign_dir), y_top]
                                p3 = [x_des - (r_x * sign_dir), y_top]
                                p4 = [x_des, y_top - (r_y * sign_y)]
                                p5 = [x_des, y_base]
                                
                                path_coords = [p0, p1, p2, p3, p4, p5]
                                
                                # Color y Grosor
                                color = get_color_1d(row['Pasajeros']) # Usamos la nueva función de color
                                width = row['grosor_final']

                                paths_data.append({
                                    'path': path_coords,
                                    'color': color,
                                    'width': width,
                                    'Pasajeros': row['Pasajeros'],
                                    'Distancia': f"{row['dist_abs']:.2f} km",
                                    'Origen_km': f"{row['km_ori']:.1f}",
                                    'Destino_km': f"{row['km_des']:.1f}"
                                })

                                # --- PUNTA DE FLECHA (Optimizado para evitar cruces) ---
                                # Solo dibujamos flecha si el viaje es mayor a 500m para evitar manchas
                                if row['dist_abs'] > 0.5:
                                    # La colocamos en el medio del tramo horizontal (p2 -> p3)
                                    mid_x = (p2[0] + p3[0]) / 2
                                    mid_y = (p2[1] + p3[1]) / 2
                                    
                                    # Vector dirección unitario (horizontal)
                                    u_x = 1 if x_des > x_ori else -1
                                    
                                    # Tamaño ajustado: aprox 400m en escala X (aprox 6px en pantalla)
                                    arrow_size = 0.004 
                                    
                                    # Puntos de las alas (formando una V acostada)
                                    # a1 (arriba), a2 (abajo), mid es el vértice
                                    a1 = [mid_x - u_x * arrow_size, mid_y + arrow_size * 0.7]
                                    a2 = [mid_x - u_x * arrow_size, mid_y - arrow_size * 0.7]
                                    
                                    # Guardamos como path continuo a1 -> mid -> a2
                                    arrows_data.append({'path': [a1, [mid_x, mid_y], a2], 'color': color, 'width': width})

                            # Capa de Paradas (Puntos Azules) - Tab 3
                            stops_1d_layer = None
                            if not df_paradas_sel.empty:
                                df_p_linear = df_paradas_sel.copy()
                                def get_x_linear(km):
                                    return BASE_LON + (km if is_southbound else (max_km_ruta - km)) * SCALE_X
                                df_p_linear['pos'] = df_p_linear['Km_Posicion'].apply(lambda k: [get_x_linear(k), BASE_LAT])
                                # Tooltip simplificado para paradas en 1D
                                for c in ['Pasajeros', 'Distancia', 'Origen_km', 'Destino_km']: df_p_linear[c] = ""
                                df_p_linear['Pasajeros'] = "Parada: " + df_p_linear['Nombre parada']
                                
                                stops_1d_layer = pdk.Layer(
                                    "ScatterplotLayer", df_p_linear, get_position="pos",
                                    get_color=[0, 100, 255, 255], get_radius=6, radius_units='pixels', pickable=True
                                )

                            # --- RENDERIZADO PYDECK ---
                            
                            # Capa de Ruta (Línea Base negra)
                            # Extendemos un poco la línea base para que se vea bien el final
                            linea_base_data = pd.DataFrame({
                                'path': [[[BASE_LON, BASE_LAT], [BASE_LON + ruta_cum.max() * SCALE_X, BASE_LAT]]]
                            })
                            
                            # Capa de Fondo (Rectángulo Blanco detrás del gráfico)
                            max_lvl_top = len(levels_top) if levels_top else 0
                            max_lvl_bot = len(levels_bottom) if levels_bottom else 0
                            min_x, max_x = BASE_LON - 0.05, BASE_LON + ruta_cum.max() * SCALE_X + 0.05
                            min_y = BASE_LAT - (max_lvl_bot + 20) * SCALE_Y
                            max_y = BASE_LAT + (max_lvl_top + 20) * SCALE_Y
                            
                            bg_data = pd.DataFrame({'polygon': [[[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]]})

                            layers_1d = [
                                # Fondo primero (z-index bajo)
                                pdk.Layer("PolygonLayer", bg_data, get_polygon="polygon", get_fill_color=[255, 255, 255], get_line_color=[255,255,255], pickable=False),
                                # Línea Base
                                pdk.Layer("PathLayer", linea_base_data, get_path="path", get_color=[0,0,0,255], get_width=3, width_units='pixels'),
                                # Arcos (Brackets)
                                pdk.Layer("PathLayer", pd.DataFrame(paths_data), get_path="path", get_color="color", get_width="width", width_units='pixels', pickable=True, rounded=True),
                                # Flechas (Ahora PathLayer para uniones limpias)
                                pdk.Layer("PathLayer", pd.DataFrame(arrows_data), get_path="path", get_color="color", get_width="width", width_units='pixels', rounded=True),
                            ]

                            # Capa de Secciones (Etiquetas) - Tab 3
                            if not df_paradas_sel.empty and 'Seccion' in df_paradas_sel.columns:
                                df_l_1d = df_paradas_sel.dropna(subset=['Seccion']).copy()
                                if criterio_agrupacion == "Por Sección":
                                    df_l_1d['Seccion'] = (df_l_1d['Seccion'] // n_secciones_agrupar) * n_secciones_agrupar
                                
                                df_s_linear = df_l_1d.sort_values('Orden').groupby('Seccion').first().reset_index()
                                def get_x_linear_local(km): return BASE_LON + (km if is_southbound else (max_km_ruta - km)) * SCALE_X
                                # Posicionar arriba de los flujos superiores (dinámico según el apilamiento)
                                label_y = BASE_LAT + (max_lvl_top + 6) * SCALE_Y
                                df_s_linear['pos'] = df_s_linear['Km_Posicion'].apply(lambda k: [get_x_linear_local(k), label_y])
                                df_s_linear['text'] = df_s_linear['Seccion'].astype(int).astype(str)
                                
                                layers_1d.append(pdk.Layer(
                                    "TextLayer", df_s_linear, get_position="pos", get_text="text",
                                    get_size=22, get_color=[255, 255, 255], get_alignment_baseline="'bottom'", get_text_anchor="'middle'", background_color=[0, 0, 0, 160]
                                ))

                            if stops_1d_layer:
                                layers_1d.append(stops_1d_layer)

                            # Vista Centrada en el medio del recorrido
                            cx = BASE_LON + (ruta_cum.max() * SCALE_X) / 2
                            view_1d = pdk.ViewState(latitude=BASE_LAT, longitude=cx, zoom=11, pitch=0, bearing=0, max_pitch=0, min_pitch=0, max_rotation=0, min_rotation=0)

                            st.pydeck_chart(pdk.Deck(
                                map_provider=None, # Sin mapa base, solo canvas blanco
                                map_style=None,
                                initial_view_state=view_1d,
                                layers=layers_1d,
                                tooltip={
                                    "html": "<b>Pax: {Pasajeros}</b><br>Dist: {Distancia}<br>Km: {Origen_km} -> {Destino_km}",
                                    "style": {"color": "white", "backgroundColor": "#333"}
                                }
                            ), use_container_width=True)
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")

                    else:
                        st.info("Se requiere cargar una ruta de referencia (ACTrec) para visualizar el gráfico 1D.")

                with tab4:
                    if not df_ruta.empty and not df_zonas.empty:
                        with st.spinner("Generando Matriz Origen-Destino..."):
                            # 1. Mapear flujos a Kilómetros de Ruta (Lógica OD)
                            ruta_lats, ruta_lons, ruta_cum = df_ruta['Latitud'].values, df_ruta['Longitud'].values, df_ruta['Dist_Acum'].values
                            max_km_od = ruta_cum.max()
                            
                            d_ori_od = (df_zonas['lat_ori'].values[:, None] - ruta_lats[None, :])**2 + (df_zonas['lon_ori'].values[:, None] - ruta_lons[None, :])**2
                            km_ori_od = ruta_cum[np.argmin(d_ori_od, axis=1)]
                            d_des_od = (df_zonas['lat_des'].values[:, None] - ruta_lats[None, :])**2 + (df_zonas['lon_des'].values[:, None] - ruta_lons[None, :])**2
                            km_des_od = ruta_cum[np.argmin(d_des_od, axis=1)]
                            
                            df_od_matrix = df_zonas.copy()
                            df_od_matrix['km_ori'], df_od_matrix['km_des'] = km_ori_od, km_des_od
                            
                            # 2. Configuración Visual de la Matriz
                            SCALE_OD = 0.001 
                            
                            # Preparar orden y offsets para desplazar flujos paralelos
                            df_od_matrix = df_od_matrix.sort_values(['km_ori', 'km_des'])
                            df_od_matrix['ori_idx'] = df_od_matrix.groupby('km_ori').cumcount()
                            df_od_matrix['ori_tot'] = df_od_matrix.groupby('km_ori')['km_des'].transform('count')
                            df_od_matrix['des_idx'] = df_od_matrix.groupby('km_des').cumcount()
                            df_od_matrix['des_tot'] = df_od_matrix.groupby('km_des')['km_ori'].transform('count')

                            OFFSET_VAL = 0.00015  # Espaciado entre líneas para evitar superposición
                            CURVE_VAL = 0.0005    # Radio de la curva (tramo a 45°)
                            FIXED_WIDTH = 3.5     # Ancho fijo pedido
                            
                            paths_od = []
                            for _, row in df_od_matrix.iterrows():
                                # Calculamos los desplazamientos para que no se pisen los flujos
                                off_y = (row['ori_idx'] - (row['ori_tot'] - 1) / 2) * OFFSET_VAL
                                off_x = (row['des_idx'] - (row['des_tot'] - 1) / 2) * OFFSET_VAL
                                
                                y_s = -row['km_ori'] * SCALE_OD + off_y
                                x_d = row['km_des'] * SCALE_OD + off_x
                                
                                # Geometría: Recto -> Curva 45° -> Recto final
                                p1 = [0, y_s]
                                p2 = [x_d - CURVE_VAL, y_s]
                                p3 = [x_d, y_s + CURVE_VAL]
                                p4 = [x_d, 0]
                                
                                paths_od.append({
                                    'path': [p1, p2, p3, p4],
                                    'color': get_color_shared(row['Pasajeros']),
                                    'width': FIXED_WIDTH,
                                    'Pasajeros': row['Pasajeros'],
                                    'tooltip_od': f"<b>Pasajeros:</b> {row['Pasajeros']}"
                                })
                            
                            # Capa de Paradas (Puntos Azules) - Tab 4 (En Ejes X e Y)
                            stops_od_layers = []
                            if not df_paradas_sel.empty:
                                df_ps_y = df_paradas_sel.copy()
                                df_ps_y['lon_od'], df_ps_y['lat_od'] = 0, -df_ps_y['Km_Posicion'] * SCALE_OD
                                df_ps_y['tooltip_od'] = "<b>Parada (Origen):</b> " + df_ps_y['Nombre parada']
                                df_ps_x = df_paradas_sel.copy()
                                df_ps_x['lon_od'], df_ps_x['lat_od'] = df_ps_x['Km_Posicion'] * SCALE_OD, 0
                                df_ps_x['tooltip_od'] = "<b>Parada (Destino):</b> " + df_ps_x['Nombre parada']
                                stops_od_layers.append(pdk.Layer("ScatterplotLayer", df_ps_y, get_position=["lon_od", "lat_od"], get_color=[0, 100, 255, 180], get_radius=4, radius_units='pixels', pickable=True))
                                stops_od_layers.append(pdk.Layer("ScatterplotLayer", df_ps_x, get_position=["lon_od", "lat_od"], get_color=[0, 100, 255, 180], get_radius=4, radius_units='pixels', pickable=True))
                            
                            # 3. Preparación de Nodos en Ejes (Independientes por Subida/Bajada)
                            max_sub_od = df_nodos['Subieron'].max()
                            max_baj_od = df_nodos['Bajaron'].max()
                            
                            # Nodos Origen (Eje Y - Izquierda)
                            nodes_y = df_nodos.copy()
                            nodes_y['lon_od'], nodes_y['lat_od'] = 0, -nodes_y['Km_Posicion'] * SCALE_OD
                            # Radio según quienes SUBIERON
                            nodes_y['radius_px'] = 6 + (nodes_y['Subieron'] / max_sub_od * 20) if max_sub_od > 0 else 6
                            # Tooltip específico para Origen
                            nodes_y['tooltip_od'] = nodes_y.apply(lambda r: f"<b>Origen (Subidas):</b> {r['Subieron']}<br/><b>%:</b> {r['Porcentaje_Subieron_Str']}", axis=1)
                            
                            # Nodos Destino (Eje X - Arriba)
                            nodes_x = df_nodos.copy()
                            nodes_x['lon_od'], nodes_x['lat_od'] = nodes_x['Km_Posicion'] * SCALE_OD, 0
                            # Radio según quienes BAJARON
                            nodes_x['radius_px'] = 6 + (nodes_x['Bajaron'] / max_baj_od * 20) if max_baj_od > 0 else 6
                            # Tooltip específico para Destino
                            nodes_x['tooltip_od'] = nodes_x.apply(lambda r: f"<b>Destino (Bajadas):</b> {r['Bajaron']}<br/><b>%:</b> {r['Porcentaje_Bajaron_Str']}", axis=1)
                            
                            layers_od = [
                                # Fondo blanco para contraste (debe ir primero para estar al fondo)
                                pdk.Layer("PolygonLayer", pd.DataFrame({'poly': [[[-0.05, 0.05], [max_km_od*SCALE_OD + 0.05, 0.05], [max_km_od*SCALE_OD + 0.05, -max_km_od*SCALE_OD - 0.05], [-0.05, -max_km_od*SCALE_OD - 0.05]]]}), get_polygon="poly", get_fill_color=[255, 255, 255]),
                                # Ejes de referencia (Y y X)
                                pdk.Layer("PathLayer", pd.DataFrame({'path': [[[0, 0.01], [0, -max_km_od*SCALE_OD - 0.01]], [[-0.01, 0], [max_km_od*SCALE_OD + 0.01, 0]] ]}), get_path="path", get_color=[50, 50, 50], get_width=2, width_units='pixels'),
                                # Ejes con números de sección
                                *(
                                    [pdk.Layer("TextLayer",
                                              (df_paradas_sel.dropna(subset=['Seccion']).copy().assign(
                                                  Seccion=lambda x: (x['Seccion'] // n_secciones_agrupar) * n_secciones_agrupar if criterio_agrupacion == "Por Sección" else x['Seccion']
                                              ).sort_values('Orden').groupby('Seccion').first().reset_index()).assign(
                                                  pos_y=lambda d: d['Km_Posicion'].apply(lambda km: [-0.001, -km * SCALE_OD]), # Desplazar a la izquierda del eje
                                                  text=lambda d: d['Seccion'].astype(int).astype(str)
                                              ),
                          get_position="pos_y", get_text="text", get_size=22, get_color=[255, 255, 255], get_text_anchor="'end'", get_alignment_baseline="'center'", background_color=[0, 0, 0, 160]), 
                                     pdk.Layer("TextLayer",
                                              (df_paradas_sel.dropna(subset=['Seccion']).copy().assign(
                                                  Seccion=lambda x: (x['Seccion'] // n_secciones_agrupar) * n_secciones_agrupar if criterio_agrupacion == "Por Sección" else x['Seccion']
                                              ).sort_values('Orden').groupby('Seccion').first().reset_index()).assign(
                                                  pos_x=lambda d: d['Km_Posicion'].apply(lambda km: [km * SCALE_OD, 0.001]), # Desplazar hacia arriba (fuera del cuadrante)
                                                  text=lambda d: d['Seccion'].astype(int).astype(str)
                                              ),
                          get_position="pos_x", get_text="text", get_size=22, get_color=[255, 255, 255], get_alignment_baseline="'bottom'", get_text_anchor="'middle'", background_color=[0, 0, 0, 160])] 
                                    if not df_paradas_sel.empty and 'Seccion' in df_paradas_sel.columns else []
                                ),
                                # Flujos OD
                                pdk.Layer("PathLayer", pd.DataFrame(paths_od), get_path="path", get_color="color", get_width="width", width_units='pixels', pickable=True),
                                # Nodos Origen (Y)
                                pdk.Layer("ScatterplotLayer", nodes_y, get_position=["lon_od", "lat_od"], get_color="color_nodo", get_radius="radius_px", radius_units='pixels', pickable=True),
                                # Nodos Destino (X)
                                pdk.Layer("ScatterplotLayer", nodes_x, get_position=["lon_od", "lat_od"], get_color="color_nodo", get_radius="radius_px", radius_units='pixels', pickable=True)
                            ] + stops_od_layers
                            
                            # Centrar vista en el cuadrante inferior derecho del punto (0,0)
                            v_lat, v_lon = -(max_km_od * SCALE_OD)/2, (max_km_od * SCALE_OD)/2
                            st.pydeck_chart(pdk.Deck(
                                map_provider=None, map_style=None, # Sin mapa base
                                initial_view_state=pdk.ViewState(latitude=v_lat, longitude=v_lon, zoom=10, pitch=0, bearing=0, max_pitch=0, min_pitch=0, max_rotation=0, min_rotation=0), # Bloquear rotación y 3D
                                layers=layers_od,
                                tooltip={"html": "{tooltip_od}"}
                            ), use_container_width=True)
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            st.write("")
                            
                            st.info("Eje Y (Vertical): Orígenes por Km de Ruta | Eje X (Horizontal): Destinos por Km de Ruta. (0,0) en esquina superior izquierda.")
                    else:
                        st.info("Se requiere una ruta de referencia y datos de flujos para generar la matriz.")

                # --- EXPORTACIÓN DE DATOS ---
                if not df_zonas.empty:
                    st.markdown("---")
                    col1_dl, col2_dl, col3_dl = st.columns(3)

                    # 1. Auditoría de Inferencia (Individual)
                    # Utilizamos df_mapa en lugar de df_flujos para exportar sólo los datos filtrados por UI
                    if 'df_mapa' in locals() and not df_mapa.empty:
                        df_audit_export = df_mapa.copy()
                        if 'distancia' in df_audit_export.columns:
                            df_audit_export = df_audit_export[
                                (df_audit_export['distancia'] >= dist_rango[0]) & 
                                (df_audit_export['distancia'] <= dist_rango[1])
                            ]
                        cols_audit = ['Tarjeta', 'Fecha Hora', 'Sentido', 'Fecha Hora_Siguiente', 'Sentido_Siguiente', 'distancia', 'distancia lineal']
                        cols_audit_final = [c for c in cols_audit if c in df_audit_export.columns]
                        df_audit = df_audit_export[cols_audit_final].copy()
                        csv_audit = df_audit.to_csv(index=False, sep=';', decimal=',')
                        with col1_dl: # Usamos la primera columna para el botón de auditoría
                            st.download_button(
                                label="📥 Auditoría (Individual)",
                                data=csv_audit,
                                file_name=f"auditoria_{ramal_sel}_{sentido_sel}.csv",
                                mime="text/csv"
                            )

                    df_export = df_zonas.copy()
                    df_export['Ramal'] = ramal_sel
                    
                    # Renombrar para coincidir con solicitud
                    if 'Km_Recorridos' in df_export.columns:
                        df_export = df_export.rename(columns={'Km_Recorridos': 'distancia'})
                    
                    # Añadimos dinámicamente nombres y secciones a la exportación
                    extras = [c for c in ['nombre_ori', 'nombre_des', 'seccion_ori', 'seccion_des'] if c in df_export.columns]
                    cols_export = ['Ramal', 'Sentido'] + extras + ['lat_ori', 'lon_ori', 'lat_des', 'lon_des', 'distancia', 'Pasajeros', 'Porcentaje']
                    cols_final = [c for c in cols_export if c in df_export.columns]
                    
                    csv_flujos = df_export[cols_final].to_csv(index=False, sep=';', decimal=',')
                    
                    with col2_dl: # Ahora en la segunda columna
                        st.download_button( # Export flows
                            label="📥 Descargar CSV (Flujos)",
                            data=csv_flujos,
                            file_name=f"flujos_{ramal_sel}_{sentido_sel}.csv",
                            mime="text/csv"
                        )

                    # Exportar estadísticas de Nodos (Subidas y Bajadas por ubicación)
                    if not df_nodos.empty:
                        extras_n = [c for c in ['nombre', 'seccion'] if c in df_nodos.columns]
                        cols_nodos = ['lat', 'lon'] + extras_n + ['Km_Posicion', 'Subieron', 'Bajaron', 'Porcentaje_Actividad', 'Porcentaje_Subieron', 'Porcentaje_Bajaron']
                        cols_nodos_final = [c for c in cols_nodos if c in df_nodos.columns]
                        csv_nodos = df_nodos[cols_nodos_final].to_csv(index=False, sep=';', decimal=',')
                        # Export nodes
                        with col3_dl:
                            st.download_button(
                                label="📥 Descargar CSV (Estadísticas Nodos)",
                                data=csv_nodos,
                                file_name=f"nodos_{ramal_sel}_{sentido_sel}.csv",
                                mime="text/csv"
                            )
            else:
                st.warning("No se encontraron flujos ni transacciones para los filtros aplicados.")
        else:
            st.warning("No hay viajes que coincidan con los filtros de hora y sentido.")
    else:
        st.warning("No se encontraron viajes. Seleccione otro Ramal y sentido.")
else:
    st.info("Carga un archivo .parquet para comenzar.")