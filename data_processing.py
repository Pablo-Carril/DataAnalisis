import pandas as pd
import numpy as np
import streamlit as st
import os
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from scipy.spatial import KDTree
from spatial_utils import distancia_metros, haversine_np

@st.cache_data
def cargar_datos(archivo):
    df = pd.read_parquet(archivo)
    # Eliminamos columnas que no sirven para el análisis antes de cualquier proceso
    cols_to_drop = [
        "Turno", "Servicio", "Sección destino", "Legajo", "archivo", 
        "Tipo Trx", "integracion?", "Fecha", "Hora", "Minutos", "Segundos"
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    df['Fecha Hora'] = pd.to_datetime(df['Fecha Hora'])
    df['Fecha'] = df['Fecha Hora'].dt.date
    df['Hora_Int'] = df['Fecha Hora'].dt.hour
    df = df[(df['Latitud'] != 0) & (df['Longitud'] != 0)].dropna(subset=['Latitud', 'Longitud'])
    return df

@st.cache_data
def cargar_informacion_paradas():
    if not os.path.exists("Ramales.csv") or not os.path.exists("Paradas_SQL_todas.csv"):
        return pd.DataFrame(), {}
    df_r = pd.read_csv("Ramales.csv", sep=';')
    df_r.columns = [c.lstrip('\ufeff').strip() for c in df_r.columns]
    mapping = dict(zip(df_r['Ramal'], df_r['Codigo']))
    df_p = pd.read_csv("Paradas_SQL_todas.csv", sep=';', quotechar='"')
    df_p['Latitud'] = pd.to_numeric(df_p['Latitud'], errors='coerce')
    df_p['Longitud'] = pd.to_numeric(df_p['Longitud'], errors='coerce')
    df_p['Ramal_Cod'] = pd.to_numeric(df_p['Ramal'], errors='coerce')
    df_p['Seccion'] = pd.to_numeric(df_p['Seccion'], errors='coerce') # Ensure 'Seccion' is numeric
    return df_p, mapping

@st.cache_data
def calcular_vectores_flujo(df, df_ruta=None):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.sort_values(['Tarjeta', 'Fecha Hora'])
    df[['Latitud','Longitud']] = df[['Longitud','Latitud']].to_numpy()
    lat, lon = df['Latitud'].to_numpy(), df['Longitud'].to_numpy()
    lat[lat != 0], lon[lon != 0] = -np.abs(lat[lat != 0]), -np.abs(lon[lon != 0])
    df['Latitud'], df['Longitud'] = lat, lon

    if df_ruta is not None and not df_ruta.empty:
        r_lats, r_lons, r_cum = df_ruta['Latitud'].values, df_ruta['Longitud'].values, df_ruta['Dist_Acum'].values
        tx_lats, tx_lons = df['Latitud'].values, df['Longitud'].values
        dists_sq = (tx_lats[:, None] - r_lats[None, :])**2 + (tx_lons[:, None] - r_lons[None, :])**2
        df['Route_Dist'] = r_cum[np.argmin(dists_sq, axis=1)]
    else:
        df['Route_Dist'] = np.nan

    def find_destinations_for_card(card_df):
        fechas, sentidos, lats, lons = card_df['Fecha'].to_numpy(), card_df['Sentido'].to_numpy(), card_df['Latitud'].to_numpy(), card_df['Longitud'].to_numpy()
        fechas_horas, r_dists = card_df['Fecha Hora'].to_numpy(), card_df['Route_Dist'].to_numpy()
        num_rows = len(card_df)
        
        # Listas para almacenar resultados
        dest_lat, dest_lon, dest_sentido, dest_fecha, dest_fecha_hora = [np.nan]*num_rows, [np.nan]*num_rows, [None]*num_rows, [pd.NaT]*num_rows, [pd.NaT]*num_rows
        dist_route_km = [np.nan]*num_rows # Distancia por ruta (si disponible) o lineal
        dist_linear_km = [np.nan]*num_rows # Distancia Haversine (siempre)
        
        min_time_diff = pd.Timedelta(minutes=20)
        max_dist_m = 60000
        min_dist_m = 100

        for i in range(num_rows):
            current_fecha, current_sentido, current_fecha_hour = fechas[i], sentidos[i], fechas_horas[i]
            is_last = (i == num_rows - 1 or fechas[i+1] != current_fecha)
            found_t0 = False
            
            for j in range(i+1, num_rows):
                if fechas[j] != current_fecha: break
                if fechas_horas[j] - current_fecha_hour < min_time_diff: continue
                
                if sentidos[j] != current_sentido:
                    if abs(lats[i] - lats[j]) < 0.5 and abs(lons[i] - lons[j]) < 0.5:
                        # Optimización: Usar distancia en ruta si está disponible (es mucho más rápida)
                        if not np.isnan(r_dists[i]) and not np.isnan(r_dists[j]):
                            dist = abs(r_dists[j] - r_dists[i]) * 1000.0
                        else:
                            dist = distancia_metros(lats[i], lons[i], lats[j], lons[j])
                        
                        if min_dist_m < dist < max_dist_m:
                            dest_lat[i], dest_lon[i], dest_sentido[i], dest_fecha[i], dest_fecha_hora[i] = lats[j], lons[j], sentidos[j], fechas[j], fechas_horas[j]
                            dist_linear_km[i] = dist / 1000.0 # Convertir a KM
                            dist_route_km[i] = abs(r_dists[j] - r_dists[i]) if not np.isnan(r_dists[i]) and not np.isnan(r_dists[j]) else dist_linear_km[i]
                            found_t0 = True
                            break
            
            if not found_t0 and is_last:
                time_limit = current_fecha_hour + pd.Timedelta(hours=16)
                for j in range(i+1, num_rows):
                    if fechas[j] == current_fecha: continue
                    if fechas_horas[j] > time_limit: break
                    if fechas[j] > current_fecha:
                        dist = abs(r_dists[j] - r_dists[i])*1000 if not np.isnan(r_dists[i]) and not np.isnan(r_dists[j]) else distancia_metros(lats[i], lons[i], lats[j], lons[j])
                        if min_dist_m < dist < max_dist_m: 
                            dest_lat[i], dest_lon[i], dest_sentido[i], dest_fecha[i], dest_fecha_hora[i] = lats[j], lons[j], sentidos[j], fechas[j], fechas_horas[j]
                            
                            dist_linear_km[i] = distancia_metros(lats[i], lons[i], lats[j], lons[j]) / 1000.0 # Siempre Haversine en KM
                            
                            if not np.isnan(r_dists[i]) and not np.isnan(r_dists[j]):
                                dist_route_km[i] = abs(r_dists[j] - r_dists[i]) # r_dists ya está en KM
                            else:
                                dist_route_km[i] = dist_linear_km[i] # Fallback a lineal si no hay ruta
                            break
        return pd.DataFrame({
            'Lat_Destino': dest_lat, 
            'Lon_Destino': dest_lon, 
            'Sentido_Siguiente': dest_sentido, 
            'Fecha_Siguiente': dest_fecha,
            'Fecha Hora_Siguiente': dest_fecha_hora, # Nueva columna
            'distancia': dist_route_km, # Nueva columna
            'distancia lineal': dist_linear_km # Nueva columna
        }, index=card_df.index)


    destinations = df.groupby('Tarjeta', sort=False, group_keys=False).apply(find_destinations_for_card)
    df_with_dest = df.join(destinations)
    return df_with_dest[(df_with_dest['Lat_Destino'].notna()) & (df_with_dest['Lat_Destino'] != 0)].copy()

@st.cache_data
def snap_to_route(df, df_ruta):
    df_snapped = df.copy()
    if df_ruta is None or df_ruta.empty: return df_snapped
    tree = KDTree(df_ruta[['Latitud', 'Longitud']].values)
    _, idx_ori = tree.query(df_snapped[['Latitud', 'Longitud']].values)
    _, idx_des = tree.query(df_snapped[['Lat_Destino', 'Lon_Destino']].values)
    ruta_cum = df_ruta['Dist_Acum'].values
    df_snapped['idx_ori'], df_snapped['idx_des'] = idx_ori, idx_des
    df_snapped['dist_acum_ori'], df_snapped['dist_acum_des'] = ruta_cum[idx_ori], ruta_cum[idx_des]
    return df_snapped

@st.cache_data
def agrupar_por_zonas(df, df_ruta, metros_sel=100, criterio="Distancia", kde_mode="Unidos", df_paradas=None, n_paradas=1):
    if criterio == "Distancia":
        if df_ruta.empty: return pd.DataFrame()
        df_s = snap_to_route(df, df_ruta)
        km_bin = metros_sel / 1000.0
        dist_b_ori, dist_b_des = (df_s['dist_acum_ori'] / km_bin).round() * km_bin, (df_s['dist_acum_des'] / km_bin).round() * km_bin
        idx_b_ori = np.argmin(np.abs(dist_b_ori.values[:, None] - df_ruta['Dist_Acum'].values[None, :]), axis=1)
        idx_b_des = np.argmin(np.abs(dist_b_des.values[:, None] - df_ruta['Dist_Acum'].values[None, :]), axis=1)
        df_mapped = pd.DataFrame({'lat_ori': df_ruta['Latitud'].values[idx_b_ori], 'lon_ori': df_ruta['Longitud'].values[idx_b_ori],
                                 'lat_des': df_ruta['Latitud'].values[idx_b_des], 'lon_des': df_ruta['Longitud'].values[idx_b_des], 'Sentido': df['Sentido'].values})
        return df_mapped.groupby(['lat_ori', 'lon_ori', 'lat_des', 'lon_des', 'Sentido']).size().reset_index(name='Pasajeros')

    elif criterio == "Clusters":
        if df_ruta.empty: return pd.DataFrame()
        df_s = snap_to_route(df, df_ruta)
        all_dists = np.concatenate([df_s['dist_acum_ori'].values, df_s['dist_acum_des'].values])
        db = DBSCAN(eps=metros_sel/1000.0, min_samples=5).fit(all_dists.reshape(-1, 1))
        df_d = pd.DataFrame({'dist_km': all_dists, 'label': db.labels_})
        valid = df_d[df_d['label'] != -1]
        if valid.empty: return pd.DataFrame()
        centroids_1d = valid.groupby('label')['dist_km'].mean()
        c_idx = np.argmin(np.abs(centroids_1d.values[:, None] - df_ruta['Dist_Acum'].values[None, :]), axis=1)
        c_coords = pd.DataFrame({'lat': df_ruta['Latitud'].values[c_idx], 'lon': df_ruta['Longitud'].values[c_idx]}, index=centroids_1d.index)
        n = len(df); labels_ori, labels_des = db.labels_[:n], db.labels_[n:]
        mask = (labels_ori != -1) & (labels_des != -1)
        df_v = df[mask].copy()
        df_v['lat_ori'], df_v['lon_ori'] = c_coords.loc[labels_ori[mask], 'lat'].values, c_coords.loc[labels_ori[mask], 'lon'].values
        df_v['lat_des'], df_v['lon_des'] = c_coords.loc[labels_des[mask], 'lat'].values, c_coords.loc[labels_des[mask], 'lon'].values
        return df_v.groupby(['lat_ori', 'lon_ori', 'lat_des', 'lon_des', 'Sentido']).size().reset_index(name='Pasajeros')

    elif criterio == "KDE":
        if df_ruta.empty: return pd.DataFrame()
        df_s = snap_to_route(df, df_ruta)
        d_ori, d_des = df_s['dist_acum_ori'].dropna().values, df_s['dist_acum_des'].dropna().values
        bandwidth = metros_sel / 1000.0
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        grid = np.linspace(0, df_ruta['Dist_Acum'].max(), int(df_ruta['Dist_Acum'].max() * 100))
        if kde_mode == "Separados":
            p_ori, p_des = np.array([]), np.array([])
            if len(d_ori) > 2: kde.fit(d_ori.reshape(-1,1)); p_ori = grid[find_peaks(kde.score_samples(grid.reshape(-1,1)))[0]]
            if len(d_des) > 2: kde.fit(d_des.reshape(-1,1)); p_des = grid[find_peaks(kde.score_samples(grid.reshape(-1,1)))[0]]
            peak_locs = np.unique(np.round(np.concatenate([p_ori, p_des]) / (bandwidth/2)) * (bandwidth/2))
        else:
            valid = np.concatenate([d_ori, d_des])
            kde.fit(valid.reshape(-1,1)); peak_locs = grid[find_peaks(kde.score_samples(grid.reshape(-1,1)))[0]]
        if len(peak_locs) == 0: peak_locs = np.array([0, df_ruta['Dist_Acum'].max()])
        p_idx = np.argmin(np.abs(peak_locs[:, None] - df_ruta['Dist_Acum'].values[None, :]), axis=1)
        hubs = pd.DataFrame({'lat': df_ruta['Latitud'].values[p_idx], 'lon': df_ruta['Longitud'].values[p_idx]})
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
        if n_paradas > 1:
            idx_ori, idx_des = (idx_ori // n_paradas) * n_paradas, (idx_des // n_paradas) * n_paradas
            max_i = len(df_p) - 1
            idx_ori, idx_des = np.clip(idx_ori, 0, max_i), np.clip(idx_des, 0, max_i)
        df_mapped = pd.DataFrame({'lat_ori': df_p.loc[idx_ori, 'Latitud'].values, 'lon_ori': df_p.loc[idx_ori, 'Longitud'].values,
                                 'lat_des': df_p.loc[idx_des, 'Latitud'].values, 'lon_des': df_p.loc[idx_des, 'Longitud'].values, 'Sentido': df['Sentido'].values})
        return df_mapped.groupby(['lat_ori', 'lon_ori', 'lat_des', 'lon_des', 'Sentido']).size().reset_index(name='Pasajeros')

    elif criterio == "Por Sección":
        if df_paradas is None or df_paradas.empty:
            st.warning("La agrupación por sección requiere información de paradas.")
            return pd.DataFrame()

        # Ensure 'Seccion' is numeric and handle potential NaNs
        df_paradas_copy = df_paradas.copy()
        df_paradas_copy['Seccion'] = pd.to_numeric(df_paradas_copy['Seccion'], errors='coerce').fillna(-1).astype(int)
        df_paradas_copy = df_paradas_copy[df_paradas_copy['Seccion'] != -1] # Remove invalid sections

        if df_paradas_copy.empty:
            return pd.DataFrame()

        # Create KDTree for efficient nearest stop lookup
        stop_coords = df_paradas_copy[['Latitud', 'Longitud']].values
        tree = KDTree(stop_coords)

        # Find nearest stop for each transaction origin and destination
        _, idx_ori_stop = tree.query(df[['Latitud', 'Longitud']].values)
        _, idx_des_stop = tree.query(df[['Lat_Destino', 'Lon_Destino']].values)

        # Get the section ID for the nearest stops
        # Use .iloc to ensure correct indexing after KDTree query
        section_ori_ids = df_paradas_copy.iloc[idx_ori_stop]['Seccion'].values
        section_des_ids = df_paradas_copy.iloc[idx_des_stop]['Seccion'].values

        # Group sections into larger bins if n_paradas (interpreted as n_sections_to_group) > 1
        if n_paradas > 1:
            section_ori_ids = (section_ori_ids // n_paradas) * n_paradas
            section_des_ids = (section_des_ids // n_paradas) * n_paradas

        # Calculamos los centroides basados en la tabla de paradas de referencia
        ref_stops = df_paradas_copy.copy()
        if n_paradas > 1:
            ref_stops['Grouped_Seccion'] = (ref_stops['Seccion'] // n_paradas) * n_paradas
        else:
            ref_stops['Grouped_Seccion'] = ref_stops['Seccion']

        section_centroids = ref_stops.groupby('Grouped_Seccion')[['Latitud', 'Longitud']].mean()

        # Map the grouped section IDs from transactions to their centroid coordinates
        lat_ori_grouped = section_centroids.loc[section_ori_ids, 'Latitud'].values
        lon_ori_grouped = section_centroids.loc[section_ori_ids, 'Longitud'].values
        lat_des_grouped = section_centroids.loc[section_des_ids, 'Latitud'].values
        lon_des_grouped = section_centroids.loc[section_des_ids, 'Longitud'].values

        df_mapped = pd.DataFrame({
            'lat_ori': lat_ori_grouped,
            'lon_ori': lon_ori_grouped,
            'lat_des': lat_des_grouped,
            'lon_des': lon_des_grouped,
            'Sentido': df['Sentido'].values
        })

        # Group by the section coordinates
        df_zonas = df_mapped.groupby([
            'lat_ori', 'lon_ori', 'lat_des', 'lon_des', 'Sentido'
        ]).size().reset_index(name='Pasajeros')

        return df_zonas

    return pd.DataFrame()

@st.cache_data
def calcular_estadisticas_nodos(df_zonas):
    if df_zonas.empty: return pd.DataFrame()
    sub = df_zonas.groupby(['lat_ori', 'lon_ori'])['Pasajeros'].sum().reset_index().rename(columns={'lat_ori': 'lat', 'lon_ori': 'lon', 'Pasajeros': 'Subieron'})
    baj = df_zonas.groupby(['lat_des', 'lon_des'])['Pasajeros'].sum().reset_index().rename(columns={'lat_des': 'lat', 'lon_des': 'lon', 'Pasajeros': 'Bajaron'})
    nodos = pd.merge(sub, baj, on=['lat', 'lon'], how='outer').fillna(0)
    nodos['Subieron'], nodos['Bajaron'] = nodos['Subieron'].astype(int), nodos['Bajaron'].astype(int)
    nodos['Total_Actividad'] = nodos['Subieron'] + nodos['Bajaron']
    t_act, t_sub, t_baj = nodos['Total_Actividad'].sum(), nodos['Subieron'].sum(), nodos['Bajaron'].sum()
    nodos['Porcentaje_Actividad'] = (nodos['Total_Actividad'] / t_act * 100) if t_act > 0 else 0.0
    nodos['Porcentaje_Subieron'] = (nodos['Subieron'] / t_sub * 100) if t_sub > 0 else 0.0
    nodos['Porcentaje_Bajaron'] = (nodos['Bajaron'] / t_baj * 100) if t_baj > 0 else 0.0
    return nodos

@st.cache_data
def cargar_ruta_referencia(archivo):
    try:
        df = pd.read_csv(archivo, sep=';', decimal=',', names=['Ramal', 'Sentido', 'Latitud', 'Longitud', 'Orden'])
        df = df.sort_values('Orden').reset_index(drop=True)
        dists = haversine_np(df['Longitud'].values[:-1], df['Latitud'].values[:-1], df['Longitud'].values[1:], df['Latitud'].values[1:])
        df['Dist_Acum'] = np.concatenate(([0], np.cumsum(dists)))
        return df
    except: return pd.DataFrame()