import numpy as np

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

def calcular_distancia_traza_vectorizado(lats_ori, lons_ori, lats_des, lons_des, df_ruta):
    ruta_lats, ruta_lons, ruta_cum = df_ruta['Latitud'].values, df_ruta['Longitud'].values, df_ruta['Dist_Acum'].values
    idx_ori = np.argmin((lats_ori[:, None] - ruta_lats[None, :])**2 + (lons_ori[:, None] - ruta_lons[None, :])**2, axis=1)
    idx_des = np.argmin((lats_des[:, None] - ruta_lats[None, :])**2 + (lons_des[:, None] - ruta_lons[None, :])**2, axis=1)
    return ruta_cum[idx_des] - ruta_cum[idx_ori]