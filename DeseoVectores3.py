import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import os
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from scipy.spatial import KDTree

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
                height: 550px !important;
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

# --- 1. CARGA DE DATOS ---
@st.cache_data
def cargar_datos(archivo):
    df = pd.read_parquet(archivo)
    df['Fecha Hora'] = pd.to_datetime(df['Fecha Hora'])
    if 'Fecha' in df.columns:
        df = df.drop(columns=['Fecha'])
    df['Fecha'] = df['Fecha Hora'].dt.date
    
    #Optimización de memoria: float64 -> float32
    #for col in ['Latitud', 'Longitud']:
    #    if col in df.columns:
    #        df[col] = df[col].astype('float32')
            
    df['Hora_Int'] = df['Fecha Hora'].dt.hour
    df = df[(df['Latitud'] != 0) & (df['Longitud'] != 0)]
    df = df.dropna(subset=['Latitud', 'Longitud'])
    return df

# --- 2. LÓGICA DE PROCESAMIENTO ---
@st.cache_data
def calcular_vectores_flujo(df, df_ruta=None):

    # ------------------------------------------------------------
    # 1. PREPARACIÓN DEL DATAFRAME
    # ------------------------------------------------------------

    # Ordenar por tarjeta y tiempo.
    # Esto es fundamental para reconstruir la secuencia real de viajes
    # de cada usuario (trip chaining).
    df = df.sort_values(['Tarjeta', 'Fecha Hora'])

    # Intercambiar columnas Latitud/Longitud.
    # Algunos datasets vienen invertidos y esto corrige el orden.
    df[['Latitud','Longitud']] = df[['Longitud','Latitud']].to_numpy()

    # Convertir columnas a arrays numpy para acelerar cálculos.
    lat = df['Latitud'].to_numpy()
    lon = df['Longitud'].to_numpy()

    # Corregir signo de coordenadas.
    # En Argentina latitudes y longitudes deben ser negativas.
    # Esto evita errores de ubicación en el mapa.
    lat[lat != 0] = -np.abs(lat[lat != 0])
    lon[lon != 0] = -np.abs(lon[lon != 0])

    df['Latitud'] = lat
    df['Longitud'] = lon

    # Si hay ruta, calcular la distancia acumulada de cada transacción (Snapping)
    if df_ruta is not None and not df_ruta.empty:
        r_lats = df_ruta['Latitud'].values
        r_lons = df_ruta['Longitud'].values
        r_cum = df_ruta['Dist_Acum'].values

        tx_lats = df['Latitud'].values
        tx_lons = df['Longitud'].values

        # Broadcasting para encontrar el punto más cercano en la ruta
        dists_sq = (tx_lats[:, None] - r_lats[None, :])**2 + (tx_lons[:, None] - r_lons[None, :])**2
        min_idx = np.argmin(dists_sq, axis=1)
        df['Route_Dist'] = r_cum[min_idx] # Distancia en KM
    else:
        df['Route_Dist'] = np.nan


    # ------------------------------------------------------------
    # FUNCIÓN PRINCIPAL DE INFERENCIA DE DESTINO
    # Se ejecuta para cada tarjeta individual.
    # ------------------------------------------------------------
    def find_destinations_for_card(card_df):

        # Convertir columnas a arrays numpy para acceso rápido
        fechas = card_df['Fecha'].to_numpy()
        sentidos = card_df['Sentido'].to_numpy()
        lats = card_df['Latitud'].to_numpy()
        lons = card_df['Longitud'].to_numpy()
        fechas_horas = card_df['Fecha Hora'].to_numpy()
        r_dists = card_df['Route_Dist'].to_numpy()

        num_rows = len(card_df)

        # Listas donde se guardará el destino inferido
        dest_lat = [np.nan] * num_rows
        dest_lon = [np.nan] * num_rows
        dest_sentido = [None] * num_rows
        dest_fecha = [pd.NaT] * num_rows

        # ------------------------------------------------------------
        # PARÁMETROS DEL ALGORITMO
        # ------------------------------------------------------------

        # Tiempo mínimo entre viajes.
        # Evita emparejar transbordos o errores (ej: bajar y subir al siguiente).
        min_time_diff = pd.Timedelta(minutes=20)

        # Distancia mínima y máxima entre origen y destino.
        # Evita dos tipos de error:
        # - rebotes muy cortos (ej: subir y bajar en la misma parada)
        # - destinos absurdamente lejanos.
        max_dist = 60000
        min_dist = 50


        # ------------------------------------------------------------
        # RECORRIDO DE TODOS LOS VIAJES DE LA TARJETA
        # ------------------------------------------------------------
        for i in range(num_rows):

            current_fecha = fechas[i]
            current_sentido = sentidos[i]
            current_fecha_hora = fechas_horas[i]

            # Detectar si este es el último viaje del día.
            # Esto es importante para aplicar la regla T+1.
            is_last_trip_of_day = (
                i == num_rows - 1 or fechas[i+1] != current_fecha
            )

            found_t0 = False


            # ------------------------------------------------------------
            # PRIORIDAD 1: DESTINO EN EL MISMO DÍA (T+0)
            # ------------------------------------------------------------
            # El destino de un viaje suele ser el origen del próximo viaje
            # en sentido contrario dentro del mismo día.
            for j in range(i+1, num_rows):

                # Si cambió el día se corta la búsqueda.
                if fechas[j] != current_fecha:
                    break

                # Evitar transbordos o rebotes cercanos en el tiempo.
                if fechas_horas[j] - current_fecha_hora < min_time_diff:
                    continue

                # Buscamos viaje en sentido contrario.
                if sentidos[j] != current_sentido:

                    # Filtro rápido por bounding box (~5 km). MODIFICO para captar viajes largos de hasta 50 Kmts
                    # Evita calcular distancia exacta para puntos muy lejanos.
                    if abs(lats[i] - lats[j]) > 0.50 or abs(lons[i] - lons[j]) > 0.50:
                        continue

                    # Cálculo de distancia real.
                    dist = distancia_metros(
                        lats[i], lons[i],
                        lats[j], lons[j]
                    )

                    # Validar rango de distancia razonable.
                    if min_dist < dist < max_dist:

                        # Guardar destino inferido.
                        dest_lat[i] = lats[j]
                        dest_lon[i] = lons[j]
                        dest_sentido[i] = sentidos[j]
                        dest_fecha[i] = fechas[j]

                        found_t0 = True

                        # Se usa el PRIMER retorno del día.
                        # Esto suele representar mejor el destino real
                        # (ej: casa → trabajo → casa).
                        break

            # Si se encontró destino en el mismo día no se busca más.
            if found_t0:
                continue


            # ------------------------------------------------------------
            # PRIORIDAD 2: DESTINO EN DÍA SIGUIENTE (T+1)
            # ------------------------------------------------------------
            # Si es el último viaje del día, se asume que el destino
            # puede ser el origen del primer viaje del día siguiente
            # (ej: trabajo → casa).
            if is_last_trip_of_day:

                # Límite máximo de tiempo (16 horas).
                # Evita emparejar viajes separados por demasiadas horas.
                time_limit = current_fecha_hora + pd.Timedelta(hours=16)

                for j in range(i+1, num_rows):

                    # Si se supera la ventana temporal se detiene la búsqueda.
                    if fechas_horas[j] > time_limit:
                        break

                    # Debe ser en un día posterior.
                    if fechas[j] > current_fecha:

                        # Filtro rápido espacial.
                        if abs(lats[i] - lats[j]) > 0.05 or abs(lons[i] - lons[j]) > 0.05:
                            continue

                        # Distancia real.
                        # Si tenemos datos de ruta, usamos la distancia real sobre el recorrido
                        if not np.isnan(r_dists[i]) and not np.isnan(r_dists[j]):
                            dist = abs(r_dists[j] - r_dists[i]) * 1000 # Convertir KM a metros
                        else:
                            # Fallback a distancia lineal (Haversine)
                            dist = distancia_metros(
                                lats[i], lons[i],
                                lats[j], lons[j]
                            )

                        # Validar distancia razonable.
                        if min_dist < dist < max_dist:

                            dest_lat[i] = lats[j]
                            dest_lon[i] = lons[j]
                            dest_sentido[i] = sentidos[j]
                            dest_fecha[i] = fechas[j]

                            break


        # Crear DataFrame con destinos inferidos
        return pd.DataFrame({
            'Lat_Destino': dest_lat,
            'Lon_Destino': dest_lon,
            'Sentido_Siguiente': dest_sentido,
            'Fecha_Siguiente': dest_fecha
        }, index=card_df.index)


    # ------------------------------------------------------------
    # APLICAR INFERENCIA A CADA TARJETA
    # ------------------------------------------------------------
    destinations = df.groupby('Tarjeta', sort=False, group_keys=False).apply(find_destinations_for_card)


    # ------------------------------------------------------------
    # UNIR DESTINOS AL DATAFRAME ORIGINAL
    # ------------------------------------------------------------
    df_with_dest = df.join(destinations)


    # ------------------------------------------------------------
    # FILTRAR VIAJES SIN DESTINO VÁLIDO
    # ------------------------------------------------------------
    # Se eliminan:
    # - destinos nulos
    # - coordenadas 0
    mask = (
        df_with_dest['Lat_Destino'].notna() &
        (df_with_dest['Lat_Destino'] != 0) &
        (df_with_dest['Lon_Destino'] != 0)
    )

    return df_with_dest[mask].copy()

# --- 3. AGRUPACIÓN ---
@st.cache_data
def snap_to_route(df, df_ruta):
    """
    Ajusta los puntos de origen y destino de un DataFrame de flujos a la ruta más cercana.
    Devuelve un DataFrame con columnas adicionales para las distancias acumuladas y los índices.
    """
    df_snapped = df.copy()
    if df_ruta is None or df_ruta.empty:
        df_snapped['idx_ori'] = -1
        df_snapped['idx_des'] = -1
        df_snapped['dist_acum_ori'] = np.nan
        df_snapped['dist_acum_des'] = np.nan
        return df_snapped

    # 1. Creamos el índice espacial de la ruta (KDTree)
    puntos_ruta = df_ruta[['Latitud', 'Longitud']].values
    tree = KDTree(puntos_ruta)

    # 2. Consultamos el punto más cercano de forma eficiente
    _, idx_ori = tree.query(df_snapped[['Latitud', 'Longitud']].values)
    _, idx_des = tree.query(df_snapped[['Lat_Destino', 'Lon_Destino']].values)

    # 3. Asignamos los valores recuperados
    ruta_cum = df_ruta['Dist_Acum'].values
    df_snapped['idx_ori'] = idx_ori
    df_snapped['idx_des'] = idx_des
    df_snapped['dist_acum_ori'] = ruta_cum[idx_ori]
    df_snapped['dist_acum_des'] = ruta_cum[idx_des]
    
    return df_snapped

@st.cache_data
def agrupar_por_zonas(df, df_ruta, metros_sel=100, criterio="Distancia", kde_mode="Unidos"):
    
    if criterio == "Distancia":
        # Si no hay ruta, no se puede agrupar por distancia en ruta.
        if df_ruta.empty:
            return pd.DataFrame()

        # Usar la nueva función para obtener los puntos ajustados
        df_snapped_data = snap_to_route(df, df_ruta)
        dist_acum_ori = df_snapped_data['dist_acum_ori'].values
        dist_acum_des = df_snapped_data['dist_acum_des'].values
        
        ruta_lats = df_ruta['Latitud'].values
        ruta_lons = df_ruta['Longitud'].values
        ruta_cum = df_ruta['Dist_Acum'].values # Distancia acumulada en KM

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
    
    elif criterio == "Clusters":
        if df_ruta.empty:
            st.warning("El clustering en ruta requiere un archivo de recorrido.")
            return pd.DataFrame()

        # 1. Snap all points to route and get their 1D distance
        df_snapped_data = snap_to_route(df, df_ruta)
        dist_acum_ori = df_snapped_data['dist_acum_ori'].values
        dist_acum_des = df_snapped_data['dist_acum_des'].values

        ruta_cum = df_ruta['Dist_Acum'].values
        ruta_lats = df_ruta['Latitud'].values
        ruta_lons = df_ruta['Longitud'].values
        
        # 2. Run DBSCAN on the 1D distances
        all_dists_km = np.concatenate([dist_acum_ori, dist_acum_des])
        eps_km = metros_sel / 1000.0
        min_samples = 5 # Mínimo de puntos para formar un cluster denso

        # DBSCAN necesita un array con forma (n_samples, n_features)
        db = DBSCAN(eps=eps_km, min_samples=min_samples, metric='euclidean', n_jobs=-1).fit(all_dists_km.reshape(-1, 1))
        labels = db.labels_

        # 3. Calculate 1D centroids (mean distance for each cluster)
        df_dists = pd.DataFrame({'dist_km': all_dists_km, 'label': labels})
        valid_dists = df_dists[df_dists['label'] != -1]
        
        if valid_dists.empty:
            return pd.DataFrame()
            
        # El centroide es el "kilometraje" promedio del cluster
        centroids_1d = valid_dists.groupby('label')['dist_km'].mean()

        # 4. Map 1D centroids back to 2D route coordinates
        # Para cada "kilometraje" del centroide, encontramos el punto más cercano en la ruta
        centroid_indices = np.argmin(np.abs(centroids_1d.values[:, None] - ruta_cum[None, :]), axis=1)
        
        # Creamos un mapa de label -> coordenadas 2D
        centroid_coords = pd.DataFrame({
            'lat': ruta_lats[centroid_indices],
            'lon': ruta_lons[centroid_indices]
        }, index=centroids_1d.index) # El índice es el 'label' del cluster

        # 5. Asignar cada transacción original a su centroide de cluster
        n = len(df)
        labels_ori = labels[:n]
        labels_des = labels[n:]
        
        # Solo consideramos viajes donde tanto el origen como el destino pertenecen a un cluster
        mask_valid = (labels_ori != -1) & (labels_des != -1)
        
        if not np.any(mask_valid):
            return pd.DataFrame()
            
        df_valid = df[mask_valid].copy()
        valid_labels_ori = labels_ori[mask_valid]
        valid_labels_des = labels_des[mask_valid]
        
        # Asignamos las coordenadas del centroide correspondiente a cada viaje
        df_valid['lat_ori'] = centroid_coords.loc[valid_labels_ori, 'lat'].values
        df_valid['lon_ori'] = centroid_coords.loc[valid_labels_ori, 'lon'].values
        df_valid['lat_des'] = centroid_coords.loc[valid_labels_des, 'lat'].values
        df_valid['lon_des'] = centroid_coords.loc[valid_labels_des, 'lon'].values
        
        # 6. Agrupar por las nuevas coordenadas del cluster para obtener los flujos
        return df_valid.groupby([
            'lat_ori', 'lon_ori', 'lat_des', 'lon_des', 'Sentido'
        ]).size().reset_index(name='Pasajeros')

    elif criterio == "KDE":
        if df_ruta.empty:
            st.warning("El método KDE requiere una ruta de referencia cargada.")
            return pd.DataFrame()

        # 1. Ajustar puntos a la ruta (Snapping)
        df_snapped_data = snap_to_route(df, df_ruta)
        dist_acum_ori = df_snapped_data['dist_acum_ori'].dropna().values
        dist_acum_des = df_snapped_data['dist_acum_des'].dropna().values

        # Datos de la ruta para mapeo inverso
        ruta_cum = df_ruta['Dist_Acum'].values
        ruta_lats = df_ruta['Latitud'].values
        ruta_lons = df_ruta['Longitud'].values

        # 2. Configuración de KDE
        bandwidth = metros_sel / 1000.0  # Convertir metros a KM
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        max_dist = ruta_cum.max()
        # Resolución de muestreo: cada 10 metros
        grid_points = np.linspace(0, max_dist, int(max_dist * 100)) if max_dist > 0 else np.array([])
        
        peak_locs_km = np.array([])

        # 3. Detección de Picos (Hubs) según el modo seleccionado
        if kde_mode == "Separados":
            # --- MODO SEPARADO: Detectar picos de subida y bajada por separado ---
            peaks_ori, peaks_des = np.array([]), np.array([])
            
            if len(dist_acum_ori) > 2:
                kde.fit(dist_acum_ori.reshape(-1, 1))
                log_dens_ori = kde.score_samples(grid_points.reshape(-1, 1))
                peaks_idx_ori, _ = find_peaks(log_dens_ori)
                peaks_ori = grid_points[peaks_idx_ori]

            if len(dist_acum_des) > 2:
                kde.fit(dist_acum_des.reshape(-1, 1))
                log_dens_des = kde.score_samples(grid_points.reshape(-1, 1))
                peaks_idx_des, _ = find_peaks(log_dens_des)
                peaks_des = grid_points[peaks_idx_des]
            
            all_peaks = np.concatenate([peaks_ori, peaks_des])
            if len(all_peaks) > 0:
                # Agrupar picos cercanos para evitar duplicados (redondeando a una fracción del bandwidth)
                rounding_factor = bandwidth / 2
                peak_locs_km = np.unique(np.round(all_peaks / rounding_factor) * rounding_factor)

        else:  # "Unidos" (comportamiento original)
            # --- MODO UNIDO: Detectar hubs de actividad general ---
            valid_points = np.concatenate([dist_acum_ori, dist_acum_des])
            if len(valid_points) > 2:
                kde.fit(valid_points.reshape(-1, 1))
                log_dens = kde.score_samples(grid_points.reshape(-1, 1))
                peaks_idx, _ = find_peaks(log_dens)
                peak_locs_km = grid_points[peaks_idx]

        # 4. Fallback y Mapeo de Picos a Coordenadas 2D
        if len(peak_locs_km) == 0:
            st.sidebar.warning("KDE no encontró picos. Usando extremos de ruta como hubs.")
            peak_locs_km = np.array([0, max_dist]) if max_dist > 0 else np.array([])

        if len(peak_locs_km) == 0:
             return pd.DataFrame()

        # 5. Mapear los picos 1D a coordenadas 2D (Lat/Lon)
        # Encontramos el índice en df_ruta más cercano a cada pico kilométrico
        peak_route_indices = np.argmin(np.abs(peak_locs_km[:, None] - ruta_cum[None, :]), axis=1)
        
        # Tabla de referencia de Hubs: Indice Pico -> Lat/Lon
        hub_coords = pd.DataFrame({
            'lat': ruta_lats[peak_route_indices],
            'lon': ruta_lons[peak_route_indices],
            'km_peak': peak_locs_km
        })

        # 6. Asignar cada transacción original al Pico (Hub) más cercano
        # Para lat_ori
        idx_closest_peak_ori = np.argmin(np.abs(df_snapped_data['dist_acum_ori'].values[:, None] - peak_locs_km[None, :]), axis=1)
        # Para lat_des
        idx_closest_peak_des = np.argmin(np.abs(df_snapped_data['dist_acum_des'].values[:, None] - peak_locs_km[None, :]), axis=1)

        df_mapped = df.copy()
        df_mapped['lat_ori'] = hub_coords.iloc[idx_closest_peak_ori]['lat'].values
        df_mapped['lon_ori'] = hub_coords.iloc[idx_closest_peak_ori]['lon'].values
        
        df_mapped['lat_des'] = hub_coords.iloc[idx_closest_peak_des]['lat'].values
        df_mapped['lon_des'] = hub_coords.iloc[idx_closest_peak_des]['lon'].values

        # Evitar flujos "dentro del mismo hub" (ruido visual)
        df_mapped = df_mapped[idx_closest_peak_ori != idx_closest_peak_des]

        if df_mapped.empty:
            return pd.DataFrame()

        # 7. Agrupar
        return df_mapped.groupby([
            'lat_ori', 'lon_ori', 'lat_des', 'lon_des', 'Sentido'
        ]).size().reset_index(name='Pasajeros')
@st.cache_data
def calcular_estadisticas_nodos(df_zonas):
    """
    Calcula las estadísticas de subidas y bajadas para cada nodo,
    derivándolas directamente de los flujos agrupados (df_zonas).
    Esto asegura que los nodos de estadísticas coincidan exactamente
    con los puntos de origen y destino de los flujos.
    """
    if df_zonas.empty:
        return pd.DataFrame()

    # Calcular 'Subieron' sumando los pasajeros de los orígenes de los flujos
    sub_counts = df_zonas.groupby(['lat_ori', 'lon_ori'])['Pasajeros'].sum().reset_index()
    sub_counts.rename(columns={'lat_ori': 'lat', 'lon_ori': 'lon', 'Pasajeros': 'Subieron'}, inplace=True)

    # Calcular 'Bajaron' sumando los pasajeros de los destinos de los flujos
    baj_counts = df_zonas.groupby(['lat_des', 'lon_des'])['Pasajeros'].sum().reset_index()
    baj_counts.rename(columns={'lat_des': 'lat', 'lon_des': 'lon', 'Pasajeros': 'Bajaron'}, inplace=True)

    # Unir las estadísticas de subidas y bajadas por coordenadas
    nodos = pd.merge(sub_counts, baj_counts, on=['lat', 'lon'], how='outer').fillna(0)
    nodos['Subieron'] = nodos['Subieron'].astype(int)
    nodos['Bajaron'] = nodos['Bajaron'].astype(int)
    
    # Calcular el total de actividad para el porcentaje
    nodos['Total_Actividad'] = nodos['Subieron'] + nodos['Bajaron']
    total_general_actividad = nodos['Total_Actividad'].sum()

    if total_general_actividad > 0:
        nodos['Porcentaje'] = ((nodos['Total_Actividad'] / total_general_actividad) * 100).map('{:.1f}%'.format)
    else:
        nodos['Porcentaje'] = "0.0%"

    return nodos

# --- 5. FUNCIONES DE DISTANCIA (RUTA) ---
def distancia_metros(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2-lat1)
    dlambda = np.radians(lon2-lon1)

    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

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
    
    # Devolvemos la diferencia CON SIGNO para detectar retrocesos
    return ruta_cum[idx_des] - ruta_cum[idx_ori]

# --- 4. INTERFAZ DE USUARIO ---
#archivo_subido = st.sidebar.file_uploader("Cargar archivo Parquet", type=["parquet"])

st.sidebar.header("Selección de Línea")
linea_seleccionada = st.sidebar.selectbox("Seleccionar Línea", ["Línea 85", "Línea 98"], index=0)

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

    # Nuevo control para seleccionar el tipo de agrupación - Por Distancia o por Clusters DBSCAN
    criterio_agrupacion = st.sidebar.radio("Agrupar Por:", ["Distancia", "Clusters", "KDE"], index=0, key="criterio_agrupacion")
    
    # Ajustamos las opciones del slider según el criterio
    label_slider = "Tamaño"
    val_default = 100
    kde_mode = "Unidos" # Valor por defecto para evitar NameError
    if criterio_agrupacion == 'Distancia':
        label_slider = "Agrupación (mts):"
        metros_sel = st.sidebar.select_slider(f"{label_slider}", options=[100, 150, 200, 300, 400, 500, 3100], value=val_default)
    elif criterio_agrupacion == 'KDE':
        label_slider = "Suavizado (Bandwidth mts):"
        val_default = 300
        metros_sel = st.sidebar.select_slider(f"{label_slider}", options=[100, 200, 300, 400, 500, 800, 1000, 1500], value=val_default)
        kde_mode = st.sidebar.radio(
            "Modo Detección KDE:", ["Unidos", "Separados"], index=0, 
            help="**Unidos**: Detecta 'hubs' de actividad general (subidas+bajadas). **Separados**: Detecta hubs de subida y bajada de forma independiente y luego los combina."
        )
    else:
        label_slider = "Radio del Cluster (mts):"
        val_default = 200
        metros_sel = st.sidebar.select_slider(f"{label_slider}", options=[100, 150, 170, 180, 190, 200, 210, 220, 230, 240, 260, 300], value=val_default)

    
    
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
        df_flujos = calcular_vectores_flujo(df_filtrado, df_ruta=df_ruta)

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

    if not df_flujos.empty:
        # Filtros de Mapa aplicados sobre los vectores
        # Optimización: Filtrado directo sin copia inicial
        mask_hora = (df_flujos['Hora_Int'] >= hora_rango[0]) & (df_flujos['Hora_Int'] <= hora_rango[1])
        df_mapa = df_flujos[mask_hora]
        
        if sentido_sel != "Ambos":
            df_mapa = df_mapa[df_mapa['Sentido'] == sentido_sel]

        if not df_mapa.empty:
            capas = []
            df_zonas = agrupar_por_zonas(df_mapa, df_ruta, metros_sel, criterio_agrupacion, kde_mode=kde_mode)
            
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
                    get_color=[250, 40, 40, 160], # color
                    get_width=15,
                    width_min_pixels=3,
                    id='ruta_referencia_layer'
                ))

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
            df_nodos = calcular_estadisticas_nodos(df_zonas) # Ahora toma df_zonas directamente

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
                
                # Calcular porcentaje del flujo respecto al total visible
                total_visible = df_zonas['Pasajeros'].sum()
                if total_visible > 0:
                    df_zonas['Porcentaje'] = ((df_zonas['Pasajeros'] / total_visible) * 100).map('{:.1f}%'.format)
                else:
                    df_zonas['Porcentaje'] = "0.0%"

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
                # Aseguramos que 'Pasajeros' esté vacío para el tooltip de nodos,
                # ya que los nodos tienen 'Subieron' y 'Bajaron'
                df_nodos['Pasajeros'] = "" 
                # 'Porcentaje' ya se calcula dentro de calcular_estadisticas_nodos
                # 'Subieron' y 'Bajaron' ya están en df_nodos

                # --- RADIO DINÁMICO PARA NODOS 3D ---
                max_act_3d = df_nodos['Total_Actividad'].max()
                if max_act_3d > 0:
                    # Radio dinámico: Mínimo 20m, Máximo 250m (según actividad)
                    df_nodos['radius_3d'] = 20 + (df_nodos['Total_Actividad'] / max_act_3d) * 230
                else:
                    df_nodos['radius_3d'] = 20

                capas.append(pdk.Layer(
                    "ScatterplotLayer",
                    df_nodos,
                    get_position=["lon", "lat"],
                    get_color=[20, 150, 0, 120], # verde 
                    get_radius='radius_3d',
                    pickable=True,
                ))

            # 3. Renderizado del mapa si hay capas que mostrar
            if capas:
                max_p_display = int(df_zonas['Pasajeros'].max()) if not df_zonas.empty else 0

                # --- CREACIÓN DE PESTAÑAS DE VISUALIZACIÓN ---
                tab1, tab2, tab3 = st.tabs(["🗺️ Mapa 3D (Arcos)", "🛰️ Vista 2D (Vectores)", "📈 Vista Lineal 1D"])

                with tab1:
                    
                    st.pydeck_chart(pdk.Deck(
                        map_provider="carto",
                        map_style="light",
                        initial_view_state=st.session_state.view_state, # Usamos SIEMPRE la vista guardada
                        layers=capas,
                        tooltip={
                            "html": "<b>Pasajeros:</b> {Pasajeros}<br/>"
                                    "<b>Subieron:</b> {Subieron}<br/>"
                                    "<b>Bajaron:</b> {Bajaron}<br/>"
                                    "<b>% Actividad:</b> {Porcentaje}"
                        },
                        # height=700 se maneja por CSS ahora, pero dejamos un valor base
                    ), key="deck_map_3d", use_container_width=True)
                    st.write("")
                    st.write("")
                    st.write(" Suben: Naranja/Rojo - Bajan: Azul")
                
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
                                "Porcentaje": np.concatenate([df['Porcentaje'].values, df['Porcentaje'].values])
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
                                df_nodos_2d['radius_2d'] = 20

                            capas_2d.append(pdk.Layer(
                                "ScatterplotLayer",
                                df_nodos_2d,
                                get_position=["lon", "lat"],
                                get_color=[30, 180, 0, 140], # Verde semi-transparente
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
                                map_provider="carto", map_style="light", 
                                initial_view_state=view_state_2d,
                                layers=capas_2d, 
                                tooltip={
                                    "html": "<b>Pasajeros:</b> {Pasajeros}<br/>"
                                            "<b>Subieron:</b> {Subieron}<br/>"
                                            "<b>Bajaron:</b> {Bajaron}<br/>"
                                            "<b>% Actividad:</b> {Porcentaje}"
                                },
                                # height=700 se maneja por CSS
                            ), key="deck_map_2d", use_container_width=True
                        )
                        
                        # Mostrar leyenda debajo del mapa
                        st.write("") # dejo un espacio
                        st.write("") # dejo un espacio
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
                            
                            # 2. Ordenar por Pasajeros (Los "más chicos" primero, para estar cerca de la línea)
                            df_1d = df_1d.sort_values('Pasajeros', ascending=True).reset_index(drop=True)
                            
                            # 3. Algoritmo de Apilamiento (Stacking) sin superposición
                            levels = [] # Lista de listas con intervalos ocupados [(inicio, fin), ...]
                            row_levels = []
                            
                            for idx, row in df_1d.iterrows():
                                s, e = min(row['km_ori'], row['km_des']), max(row['km_ori'], row['km_des'])
                                # Pequeño margen para evitar toques visuales
                                s, e = s - 0.02, e + 0.02
                                
                                assigned_lvl = -1
                                # Buscamos el nivel más bajo disponible donde no choque
                                for i, lvl_intervals in enumerate(levels):
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
                                    assigned_lvl = len(levels)
                                    levels.append([(s, e)])
                                
                                row_levels.append(assigned_lvl)
                            
                            # 4. Dibujar Gráfico con Plotly
                            fig = go.Figure()
                            
                            # Línea base (Ruta)
                            fig.add_trace(go.Scatter(x=[0, ruta_cum.max()], y=[0, 0], mode='lines', line=dict(color='black', width=3), name='Ruta', hoverinfo='skip'))
                            
                            for i, row in df_1d.iterrows():
                                lvl = row_levels[i]
                                height = (lvl + 1) * 1.5 # Altura basada en el nivel asignado
                                x0, x1 = row['km_ori'], row['km_des']
                                
                                # Generar curva (Senoide) para el arco
                                xs = np.linspace(x0, x1, 50)
                                # t va de 0 a 1 normalizado a lo largo del arco
                                t = (xs - x0) / (x1 - x0) 
                                ys = height * np.sin(np.pi * t) # Forma de arco
                                
                                # Convertir color de lista [r,g,b,a] a string 'rgba(...)'
                                c = row['color_ori']
                                color_str = f"rgba({c[0]}, {c[1]}, {c[2]}, {c[3]/255:.2f})"
                                
                                fig.add_trace(go.Scatter(
                                    x=xs, y=ys, mode='lines',
                                    line=dict(color=color_str, width=max(1, row['grosor_final']/2)), # Ajustar grosor para Plotly
                                    text=f"Pax: {row['Pasajeros']}<br>Desde: {x0:.1f}km<br>Hasta: {x1:.1f}km<br>Dist: {abs(x1-x0):.1f}km",
                                    hoverinfo='text',
                                    showlegend=False,
                                    opacity=0.9
                                ))
                            
                            fig.update_layout(
                                #title=f"Diagrama de Flujos Lineal - {ramal_sel}",
                                xaxis_title="Kilómetros",
                                yaxis_title="Nivel de Pasajeros",
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), # Ocultar eje Y
                                xaxis=dict(showgrid=True, zeroline=True),
                                height=600,
                                plot_bgcolor='white'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Se requiere cargar una ruta de referencia (ACTrec) para visualizar el gráfico 1D.")

                # --- EXPORTACIÓN DE DATOS ---
                if not df_zonas.empty:
                    # st.markdown("---")
                    col1_dl, col2_dl = st.columns(2)

                    df_export = df_zonas.copy()
                    df_export['Ramal'] = ramal_sel
                    
                    # Renombrar para coincidir con solicitud
                    if 'Km_Recorridos' in df_export.columns:
                        df_export = df_export.rename(columns={'Km_Recorridos': 'distancia'})
                    
                    cols_export = ['Ramal', 'Sentido', 'lat_ori', 'lon_ori', 'lat_des', 'lon_des', 'distancia', 'Pasajeros', 'Porcentaje']
                    cols_final = [c for c in cols_export if c in df_export.columns]
                    
                    csv = df_export[cols_final].to_csv(index=False, sep=';', decimal=',')
                    
                    with col1_dl:
                        st.download_button(
                            label="📥 Descargar CSV (Flujos)",
                            data=csv,
                            file_name=f"flujos_{ramal_sel}_{sentido_sel}.csv",
                            mime="text/csv"
                        )

                    # Exportar estadísticas de Nodos (Subidas y Bajadas por ubicación)
                    if not df_nodos.empty:
                        cols_nodos = ['lat', 'lon', 'Subieron', 'Bajaron', 'Porcentaje']
                        cols_nodos_final = [c for c in cols_nodos if c in df_nodos.columns]
                        csv_nodos = df_nodos[cols_nodos_final].to_csv(index=False, sep=';', decimal=',')
                        
                        with col2_dl:
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