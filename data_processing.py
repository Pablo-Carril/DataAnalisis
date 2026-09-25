import pandas as pd
import numpy as np
import streamlit as st
import os
from spatial_utils import distancia_metros, haversine_np
from algoritmos_tarjeta import (
    calcular_vectores_flujo_core, 
    snap_to_route_core, 
    agrupar_por_zonas_core, 
    calcular_estadisticas_nodos_core
)

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
        print("cargando ramales y paradas: no se encuentra")
        return pd.DataFrame(), {}
    print("cargando ramales y paradas: OK")
    df_r = pd.read_csv("Ramales.csv", sep=';')
    df_r.columns = [c.lstrip('\ufeff').strip() for c in df_r.columns]
    mapping = dict(zip(df_r['Ramal'], df_r['Codigo']))
    df_p = pd.read_csv("Paradas_SQL_todas.csv", sep=';', quotechar='"')
    df_p['Latitud'] = pd.to_numeric(df_p['Latitud'], errors='coerce')
    df_p['Longitud'] = pd.to_numeric(df_p['Longitud'], errors='coerce')
    df_p['Ramal_Cod'] = pd.to_numeric(df_p['Ramal'], errors='coerce')
    df_p['Seccion'] = pd.to_numeric(df_p['Seccion'], errors='coerce') # Ensure 'Seccion' is numeric
    return df_p, mapping

# --- WRAPPERS CON CACHÉ DE STREAMLIT ---
# La lógica matemática pura se encuentra ahora en algoritmos_tarjeta.py
# para que sea más fácil de probar y aislar de la interfaz.

@st.cache_data
def calcular_vectores_flujo(df):
    return calcular_vectores_flujo_core(df)

@st.cache_data
def snap_to_route(df, df_ruta, df_paradas=None):
    return snap_to_route_core(df, df_ruta, df_paradas)

@st.cache_data
def agrupar_por_zonas(df, df_ruta, metros_sel=100, criterio="Distancia", kde_mode="Unidos", df_paradas=None, n_paradas=1):
    return agrupar_por_zonas_core(df, df_ruta, metros_sel, criterio, kde_mode, df_paradas, n_paradas)

@st.cache_data
def calcular_estadisticas_nodos(df_zonas):
    return calcular_estadisticas_nodos_core(df_zonas)

# --- FIN WRAPPERS ---

@st.cache_data
def cargar_recorridos_todos():
    archivo = "Recorridos_Todos_con_punto.csv"
    if not os.path.exists(archivo):
        print(f"No se encuentra {archivo}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(archivo, sep=';', decimal='.')
        df['Latitud'] = pd.to_numeric(df['Latitud'], errors='coerce')
        df['Longitud'] = pd.to_numeric(df['Longitud'], errors='coerce')
        df['Ramal_Cod'] = pd.to_numeric(df['Ramal'], errors='coerce')
        return df
    except Exception as e:
        print(f"Error al cargar {archivo}: {e}")
        return pd.DataFrame()

def procesar_ruta_filtrada(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_values('Orden').reset_index(drop=True)
    if len(df) > 1:
        dists = haversine_np(df['Longitud'].values[:-1], df['Latitud'].values[:-1], df['Longitud'].values[1:], df['Latitud'].values[1:])
        df['Dist_Acum'] = np.concatenate(([0], np.cumsum(dists)))
    else:
        df['Dist_Acum'] = 0.0
    return df