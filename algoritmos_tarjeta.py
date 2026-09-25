import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from scipy.spatial import KDTree
from spatial_utils import distancia_metros

def _inferir_destinos_tarjeta(card_df):
    """
    Algoritmo central de inferencia de destinos (bajadas) para una sola tarjeta SUBE.
    
    Lógica de funcionamiento:
    1. Se asume que las transacciones están ordenadas cronológicamente.
    2. Para cada transacción 'A' (subida actual), se busca la siguiente transacción 'B' de la misma tarjeta.
    3. Si 'B' ocurre en el mismo día y sentido opuesto, se asume que la persona bajó en el lugar 'B'
       antes de volver a subir, o bajó en algún lugar y caminó. Por ende, la bajada de 'A' se infiere
       como la coordenada de la subida 'B'.
    4. Se validan tiempos y distancias lógicas (no muy cerca ni muy lejos) para descartar errores.
    
    Args:
        card_df (pd.DataFrame): Transacciones de una única tarjeta ordenadas por fecha y hora.
        
    Returns:
        pd.DataFrame: DataFrame con las coordenadas de destino inferidas, fecha/hora y distancias.
    """
    fechas = card_df['Fecha'].to_numpy()
    sentidos = card_df['Sentido'].to_numpy()
    lats = card_df['Latitud'].to_numpy()
    lons = card_df['Longitud'].to_numpy()
    fechas_horas = card_df['Fecha Hora'].to_numpy()
    r_dists = card_df['Route_Dist'].to_numpy() # Distancia proyectada en la ruta, pre-calculada.
    num_rows = len(card_df)
    
    # Listas pre-asignadas para almacenar los resultados (optimización de velocidad)
    dest_lat = [np.nan] * num_rows
    dest_lon = [np.nan] * num_rows
    dest_sentido = [None] * num_rows
    dest_fecha = [pd.NaT] * num_rows
    dest_fecha_hora = [pd.NaT] * num_rows
    dist_route_km = [np.nan] * num_rows  # Distancia calculada a lo largo de la traza/ruta
    dist_linear_km = [np.nan] * num_rows # Distancia en línea recta (Haversine)
    
    # --- Parámetros de validación lógica ---
    min_time_diff = pd.Timedelta(minutes=20) # Evita inferir bajadas en subidas accidentales casi inmediatas
    max_dist_m = 60000 # 60 km máximo teórico razonable para descartar saltos erróneos de GPS
    min_dist_m = 100   # 100 metros mínimo (evita inferir viajes nulos donde bajó y subió en el mismo lugar físico)

    for i in range(num_rows):
        current_fecha = fechas[i]
        current_sentido = sentidos[i]
        current_fecha_hour = fechas_horas[i]
        
        # Bandera para saber si es el último viaje del día para esta tarjeta
        is_last = (i == num_rows - 1 or fechas[i+1] != current_fecha)
        found_t0 = False
        
        # Bucle para encontrar el próximo evento válido
        for j in range(i+1, num_rows):
            if fechas[j] != current_fecha: 
                break # Rompe si cambia de día (no inferimos saltando días por ahora)
                
            if fechas_horas[j] - current_fecha_hour < min_time_diff: 
                continue # Salta si ocurrió en menos de X minutos (posible doble cobro)
            
            # --- CONDICIÓN IDEAL: Siguiente viaje detectado en el mismo día ---
            # (Se eliminó la restricción de que deba ser un sentido distinto)
            # Verificación de sanity check de GPS: No pueden estar a medio mundo de distancia (ej. Latitud fallida)
            if abs(lats[i] - lats[j]) < 0.5 and abs(lons[i] - lons[j]) < 0.5:
                
                # Si ambas transacciones se proyectaron sobre la traza, usamos esa distancia 1D (muy preciso y rápido)
                if not np.isnan(r_dists[i]) and not np.isnan(r_dists[j]):
                    dist = abs(r_dists[j] - r_dists[i]) * 1000.0
                else:
                    # Fallback a distancia Haversine si alguna coordenada no se pudo ajustar a la traza
                    dist = distancia_metros(lats[i], lons[i], lats[j], lons[j])
                
                if min_dist_m < dist < max_dist_m:
                    dest_lat[i] = lats[j]
                    dest_lon[i] = lons[j]
                    dest_sentido[i] = sentidos[j]
                    dest_fecha[i] = fechas[j]
                    dest_fecha_hora[i] = fechas_horas[j]
                    
                    dist_linear_km[i] = dist / 1000.0
                    
                    if not np.isnan(r_dists[i]) and not np.isnan(r_dists[j]):
                        dist_route_km[i] = abs(r_dists[j] - r_dists[i])
                    else:
                        dist_route_km[i] = dist_linear_km[i]
                        
                    found_t0 = True
                    break # Ya encontramos el destino, pasamos al siguiente viaje 'i'
        
        # --- CONDICIÓN ALTERNATIVA: Último viaje del día ---
        # Si no hubo viaje de vuelta, pero este fue el último del día, buscamos al día siguiente
        # bajo la suposición de que su primer viaje de mañana iniciará donde bajó hoy.
        if not found_t0 and is_last:
            time_limit = current_fecha_hour + pd.Timedelta(hours=16) # Ventana lógica de retorno
            for j in range(i+1, num_rows):
                if fechas[j] == current_fecha: 
                    continue # Debe ser un día distinto (generalmente el posterior inmediato)
                if fechas_horas[j] > time_limit: 
                    break # Si pasó más de X tiempo, ya no es confiable inferir
                    
                if fechas[j] > current_fecha:
                    if not np.isnan(r_dists[i]) and not np.isnan(r_dists[j]):
                        dist = abs(r_dists[j] - r_dists[i]) * 1000.0
                    else:
                        dist = distancia_metros(lats[i], lons[i], lats[j], lons[j])
                        
                    if min_dist_m < dist < max_dist_m: 
                        dest_lat[i] = lats[j]
                        dest_lon[i] = lons[j]
                        dest_sentido[i] = sentidos[j]
                        dest_fecha[i] = fechas[j]
                        dest_fecha_hora[i] = fechas_horas[j]
                        
                        dist_linear_km[i] = distancia_metros(lats[i], lons[i], lats[j], lons[j]) / 1000.0
                        
                        if not np.isnan(r_dists[i]) and not np.isnan(r_dists[j]):
                            dist_route_km[i] = abs(r_dists[j] - r_dists[i])
                        else:
                            dist_route_km[i] = dist_linear_km[i]
                        break

    return pd.DataFrame({
        'Lat_Destino': dest_lat, 
        'Lon_Destino': dest_lon, 
        'Sentido_Siguiente': dest_sentido, 
        'Fecha_Siguiente': dest_fecha,
        'Fecha Hora_Siguiente': dest_fecha_hora,
        'distancia': dist_route_km,
        'distancia lineal': dist_linear_km
    }, index=card_df.index)


def calcular_vectores_flujo_core(df):
    """
    Función orquestadora principal para crear la matriz de flujos Origen-Destino (O-D).
    
    1. Ajusta primero las transacciones a los puntos conocidos más precisos (paradas o traza).
    2. Llama al algoritmo de inferencia de tarjetas para deducir las bajadas.
    3. Retorna un DataFrame limpio solo con viajes deducidos correctamente.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.sort_values(['Tarjeta', 'Fecha Hora'])
    df[['Latitud','Longitud']] = df[['Longitud','Latitud']].to_numpy()
    
    lat, lon = df['Latitud'].to_numpy(), df['Longitud'].to_numpy()
    # Corrección rápida para evitar coordenadas positivas en hemisferio sur/oeste.
    lat[lat != 0] = -np.abs(lat[lat != 0])
    lon[lon != 0] = -np.abs(lon[lon != 0])
    df['Latitud'], df['Longitud'] = lat, lon

    # --- FASE 1: INFERENCIA DE DESTINOS GLOBAL ---
    # Como ahora la inferencia se hace sobre TODO el dataset (para no perder transbordos entre ramales),
    # no hacemos snapping aquí. El snapping se hará después de filtrar por ramal en `snap_to_route_core`.
    df['Route_Dist'] = np.nan

    # Se agrupa por tarjeta y se aplica la lógica pura de negocio sobre las coordenadas GPS crudas.
    destinations = df.groupby('Tarjeta', sort=False, group_keys=False).apply(_inferir_destinos_tarjeta)
    df_with_dest = df.join(destinations)
    
    # Filtramos para quedarnos solo con las transacciones donde logramos inferir una bajada
    return df_with_dest[(df_with_dest['Lat_Destino'].notna()) & (df_with_dest['Lat_Destino'] != 0)].copy()


def snap_to_route_core(df, df_ruta, df_paradas=None):
    """
    Realiza el snapping final (tanto del Origen como del Destino inferido) hacia la traza 
    o paradas, principalmente requerido antes de las agrupaciones de DBSCAN y KDE.
    """
    df_snapped = df.copy()
    ref_df = df_paradas if (df_paradas is not None and not df_paradas.empty) else df_ruta
    if ref_df is None or ref_df.empty: 
        return df_snapped
    
    tree = KDTree(ref_df[['Latitud', 'Longitud']].values)
    _, idx_ori = tree.query(df_snapped[['Latitud', 'Longitud']].values)
    _, idx_des = tree.query(df_snapped[['Lat_Destino', 'Lon_Destino']].values)
    
    dist_vals = ref_df['Km_Posicion'].values if 'Km_Posicion' in ref_df.columns else ref_df['Dist_Acum'].values
    
    df_snapped['idx_ori'] = idx_ori
    df_snapped['idx_des'] = idx_des
    df_snapped['dist_acum_ori'] = dist_vals[idx_ori]
    df_snapped['dist_acum_des'] = dist_vals[idx_des]
    
    # Actualizar coordenadas físicas al punto ajustado
    df_snapped['Latitud'] = ref_df['Latitud'].values[idx_ori]
    df_snapped['Longitud'] = ref_df['Longitud'].values[idx_ori]
    df_snapped['Lat_Destino'] = ref_df['Latitud'].values[idx_des]
    df_snapped['Lon_Destino'] = ref_df['Longitud'].values[idx_des]
    
    # Actualizamos la columna "distancia" (que originalmente era lineal en la inferencia global)
    # usando la progresiva exacta de la traza/ruta que hemos ajustado.
    df_snapped['distancia'] = np.abs(df_snapped['dist_acum_des'] - df_snapped['dist_acum_ori'])
    
    return df_snapped


def agrupar_por_zonas_core(df, df_ruta, metros_sel=100, criterio="Distancia", kde_mode="Unidos", df_paradas=None, n_paradas=1):
    """
    Reduce los puntos de subida y bajada a "zonas" o "nodos" para optimizar la visualización 
    del mapa y facilitar el análisis macro de la demanda.
    """
    if criterio == "Distancia":
        if df_ruta.empty: return pd.DataFrame()
        df_s = snap_to_route_core(df, df_ruta, df_paradas=df_paradas)
        ref_snap = df_paradas if (df_paradas is not None and not df_paradas.empty) else df_ruta
        km_bin = metros_sel / 1000.0
        dist_b_ori, dist_b_des = (df_s['dist_acum_ori'] / km_bin).round() * km_bin, (df_s['dist_acum_des'] / km_bin).round() * km_bin
        ref_cum = ref_snap['Km_Posicion'].values if 'Km_Posicion' in ref_snap.columns else ref_snap['Dist_Acum'].values
        idx_b_ori = np.argmin(np.abs(dist_b_ori.values[:, None] - ref_cum[None, :]), axis=1)
        idx_b_des = np.argmin(np.abs(dist_b_des.values[:, None] - ref_cum[None, :]), axis=1)
        df_mapped = pd.DataFrame({'lat_ori': ref_snap['Latitud'].values[idx_b_ori], 'lon_ori': ref_snap['Longitud'].values[idx_b_ori],
                                 'lat_des': ref_snap['Latitud'].values[idx_b_des], 'lon_des': ref_snap['Longitud'].values[idx_b_des], 'Sentido': df['Sentido'].values})
        return df_mapped.groupby(['lat_ori', 'lon_ori', 'lat_des', 'lon_des', 'Sentido']).size().reset_index(name='Pasajeros')

    elif criterio == "Clusters":
        if df_ruta.empty: return pd.DataFrame()
        df_s = snap_to_route_core(df, df_ruta, df_paradas=df_paradas)
        ref_snap = df_paradas if (df_paradas is not None and not df_paradas.empty) else df_ruta
        all_dists = np.concatenate([df_s['dist_acum_ori'].values, df_s['dist_acum_des'].values])
        db = DBSCAN(eps=metros_sel/1000.0, min_samples=5).fit(all_dists.reshape(-1, 1))
        df_d = pd.DataFrame({'dist_km': all_dists, 'label': db.labels_})
        valid = df_d[df_d['label'] != -1]
        if valid.empty: return pd.DataFrame()
        centroids_1d = valid.groupby('label')['dist_km'].mean()
        ref_cum = ref_snap['Km_Posicion'].values if 'Km_Posicion' in ref_snap.columns else ref_snap['Dist_Acum'].values
        c_idx = np.argmin(np.abs(centroids_1d.values[:, None] - ref_cum[None, :]), axis=1)
        c_coords = pd.DataFrame({'lat': ref_snap['Latitud'].values[c_idx], 'lon': ref_snap['Longitud'].values[c_idx]}, index=centroids_1d.index)
        n = len(df)
        labels_ori, labels_des = db.labels_[:n], db.labels_[n:]
        mask = (labels_ori != -1) & (labels_des != -1)
        df_v = df[mask].copy()
        df_v['lat_ori'], df_v['lon_ori'] = c_coords.loc[labels_ori[mask], 'lat'].values, c_coords.loc[labels_ori[mask], 'lon'].values
        df_v['lat_des'], df_v['lon_des'] = c_coords.loc[labels_des[mask], 'lat'].values, c_coords.loc[labels_des[mask], 'lon'].values
        return df_v.groupby(['lat_ori', 'lon_ori', 'lat_des', 'lon_des', 'Sentido']).size().reset_index(name='Pasajeros')

    elif criterio == "KDE":
        if df_ruta.empty: return pd.DataFrame()
        df_s = snap_to_route_core(df, df_ruta, df_paradas=df_paradas)
        ref_snap = df_paradas if (df_paradas is not None and not df_paradas.empty) else df_ruta
        d_ori, d_des = df_s['dist_acum_ori'].dropna().values, df_s['dist_acum_des'].dropna().values
        bandwidth = metros_sel / 1000.0
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        max_d = ref_snap['Km_Posicion'].max() if 'Km_Posicion' in ref_snap.columns else df_ruta['Dist_Acum'].max()
        grid = np.linspace(0, max_d, int(max_d * 100))
        if kde_mode == "Separados":
            p_ori, p_des = np.array([]), np.array([])
            if len(d_ori) > 2: 
                kde.fit(d_ori.reshape(-1,1))
                p_ori = grid[find_peaks(kde.score_samples(grid.reshape(-1,1)))[0]]
            if len(d_des) > 2: 
                kde.fit(d_des.reshape(-1,1))
                p_des = grid[find_peaks(kde.score_samples(grid.reshape(-1,1)))[0]]
            peak_locs = np.unique(np.round(np.concatenate([p_ori, p_des]) / (bandwidth/2)) * (bandwidth/2))
        else:
            valid = np.concatenate([d_ori, d_des])
            kde.fit(valid.reshape(-1,1))
            peak_locs = grid[find_peaks(kde.score_samples(grid.reshape(-1,1)))[0]]
        if len(peak_locs) == 0: 
            peak_locs = np.array([0, max_d])
        ref_cum = ref_snap['Km_Posicion'].values if 'Km_Posicion' in ref_snap.columns else ref_snap['Dist_Acum'].values
        p_idx = np.argmin(np.abs(peak_locs[:, None] - ref_cum[None, :]), axis=1)
        hubs = pd.DataFrame({'lat': ref_snap['Latitud'].values[p_idx], 'lon': ref_snap['Longitud'].values[p_idx]})
        idx_p_ori = np.argmin(np.abs(df_s['dist_acum_ori'].values[:, None] - peak_locs[None, :]), axis=1)
        idx_p_des = np.argmin(np.abs(df_s['dist_acum_des'].values[:, None] - peak_locs[None, :]), axis=1)
        df_m = df.copy()
        df_m['lat_ori'], df_m['lon_ori'] = hubs.iloc[idx_p_ori]['lat'].values, hubs.iloc[idx_p_ori]['lon'].values
        df_m['lat_des'], df_m['lon_des'] = hubs.iloc[idx_p_des]['lat'].values, hubs.iloc[idx_p_des]['lon'].values
        return df_m[idx_p_ori != idx_p_des].groupby(['lat_ori', 'lon_ori', 'lat_des', 'lon_des', 'Sentido']).size().reset_index(name='Pasajeros')

    elif criterio == "Por Parada":
        if df_paradas is None or df_paradas.empty: return pd.DataFrame()
        df_p = df_paradas.sort_values('Orden').reset_index(drop=True)
        tree = KDTree(df_p[['Latitud', 'Longitud']].values)
        _, idx_ori = tree.query(df[['Latitud', 'Longitud']].values)
        _, idx_des = tree.query(df[['Lat_Destino', 'Lon_Destino']].values)
        
        nombres_ori = df_p.loc[idx_ori, 'Nombre parada'].values
        nombres_des = df_p.loc[idx_des, 'Nombre parada'].values

        if n_paradas > 1:
            idx_ori = (idx_ori // n_paradas) * n_paradas
            idx_des = (idx_des // n_paradas) * n_paradas
            max_i = len(df_p) - 1
            idx_ori, idx_des = np.clip(idx_ori, 0, max_i), np.clip(idx_des, 0, max_i)
        
        df_mapped = pd.DataFrame({
            'lat_ori': df_p.loc[idx_ori, 'Latitud'].values, 'lon_ori': df_p.loc[idx_ori, 'Longitud'].values, 'nombre_ori': nombres_ori,
            'lat_des': df_p.loc[idx_des, 'Latitud'].values, 'lon_des': df_p.loc[idx_des, 'Longitud'].values, 'nombre_des': nombres_des,
            'Sentido': df['Sentido'].values
        })
        return df_mapped.groupby(['lat_ori', 'lon_ori', 'nombre_ori', 'lat_des', 'lon_des', 'nombre_des', 'Sentido']).size().reset_index(name='Pasajeros')

    elif criterio == "Por Sección":
        if df_paradas is None or df_paradas.empty:
            return pd.DataFrame()

        df_paradas_copy = df_paradas.copy()
        df_paradas_copy['Seccion'] = pd.to_numeric(df_paradas_copy['Seccion'], errors='coerce').fillna(-1).astype(int)
        df_paradas_copy = df_paradas_copy[df_paradas_copy['Seccion'] != -1]

        if df_paradas_copy.empty:
            return pd.DataFrame()

        stop_coords = df_paradas_copy[['Latitud', 'Longitud']].values
        tree = KDTree(stop_coords)

        _, idx_ori_stop = tree.query(df[['Latitud', 'Longitud']].values)
        _, idx_des_stop = tree.query(df[['Lat_Destino', 'Lon_Destino']].values)

        section_ori_ids = df_paradas_copy.iloc[idx_ori_stop]['Seccion'].values
        section_des_ids = df_paradas_copy.iloc[idx_des_stop]['Seccion'].values

        if n_paradas > 1:
            section_ori_ids = (section_ori_ids // n_paradas) * n_paradas
            section_des_ids = (section_des_ids // n_paradas) * n_paradas

        ref_stops = df_paradas_copy.copy()
        if n_paradas > 1:
            ref_stops['Grouped_Seccion'] = (ref_stops['Seccion'] // n_paradas) * n_paradas
        else:
            ref_stops['Grouped_Seccion'] = ref_stops['Seccion']

        section_locations = ref_stops.sort_values('Orden').groupby('Grouped_Seccion')[['Latitud', 'Longitud']].first()

        lat_ori_grouped = section_locations.loc[section_ori_ids, 'Latitud'].values
        lon_ori_grouped = section_locations.loc[section_ori_ids, 'Longitud'].values
        lat_des_grouped = section_locations.loc[section_des_ids, 'Latitud'].values
        lon_des_grouped = section_locations.loc[section_des_ids, 'Longitud'].values

        df_mapped = pd.DataFrame({
            'lat_ori': lat_ori_grouped,
            'lon_ori': lon_ori_grouped,
            'seccion_ori': section_ori_ids,
            'lat_des': lat_des_grouped,
            'lon_des': lon_des_grouped,
            'seccion_des': section_des_ids,
            'Sentido': df['Sentido'].values
        })

        df_zonas = df_mapped.groupby([
            'lat_ori', 'lon_ori', 'seccion_ori', 'lat_des', 'lon_des', 'seccion_des', 'Sentido'
        ]).size().reset_index(name='Pasajeros')

        return df_zonas

    return pd.DataFrame()


def calcular_estadisticas_nodos_core(df_zonas):
    """
    Suma todos los flujos que entran (bajaron) y salen (subieron) de un mismo nodo (coordenada).
    """
    if df_zonas.empty: return pd.DataFrame()
    
    ori_cols, des_cols = ['lat_ori', 'lon_ori'], ['lat_des', 'lon_des']
    if 'nombre_ori' in df_zonas.columns: ori_cols.append('nombre_ori')
    if 'nombre_des' in df_zonas.columns: des_cols.append('nombre_des')
    if 'seccion_ori' in df_zonas.columns: ori_cols.append('seccion_ori')
    if 'seccion_des' in df_zonas.columns: des_cols.append('seccion_des')

    sub = df_zonas.groupby(ori_cols)['Pasajeros'].sum().reset_index()
    rename_sub = {'lat_ori': 'lat', 'lon_ori': 'lon', 'Pasajeros': 'Subieron', 'nombre_ori': 'nombre', 'seccion_ori': 'seccion'}
    sub = sub.rename(columns={k: v for k, v in rename_sub.items() if k in sub.columns})

    baj = df_zonas.groupby(des_cols)['Pasajeros'].sum().reset_index()
    rename_des = {'lat_des': 'lat', 'lon_des': 'lon', 'Pasajeros': 'Bajaron', 'nombre_des': 'nombre', 'seccion_des': 'seccion'}
    baj = baj.rename(columns={k: v for k, v in rename_des.items() if k in baj.columns})

    merge_on = ['lat', 'lon']
    if 'nombre' in sub.columns and 'nombre' in baj.columns: merge_on.append('nombre')
    if 'seccion' in sub.columns and 'seccion' in baj.columns: merge_on.append('seccion')

    nodos = pd.merge(sub, baj, on=merge_on, how='outer').fillna(0)
    
    nodos['Subieron'] = nodos['Subieron'].astype(int)
    nodos['Bajaron'] = nodos['Bajaron'].astype(int)
    nodos['Total_Actividad'] = nodos['Subieron'] + nodos['Bajaron']
    
    t_act = nodos['Total_Actividad'].sum()
    t_sub = nodos['Subieron'].sum()
    t_baj = nodos['Bajaron'].sum()
    
    nodos['Porcentaje_Actividad'] = (nodos['Total_Actividad'] / t_act * 100) if t_act > 0 else 0.0
    nodos['Porcentaje_Subieron'] = (nodos['Subieron'] / t_sub * 100) if t_sub > 0 else 0.0
    nodos['Porcentaje_Bajaron'] = (nodos['Bajaron'] / t_baj * 100) if t_baj > 0 else 0.0
    
    return nodos
