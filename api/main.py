# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import pandas as pd
import os

# Definición del Esquema de Datos
# Usamos Pydantic para definir la estructura de los datos que la API envía.
class ItemResponse(BaseModel):
    id: int
    nombre: str
    descripcion: str
    
#Inicialización de FastAPI
app = FastAPI(
    title="API de Backend con FastAPI para Dashboard",
    version="1.0.0",
)

#Endpoint de Salud (Health Check)
@app.get("/")
def health_check() -> Dict[str, str]:
    """Ruta simple para verificar que la API está viva."""
    return {"status": "ok", "service": "FastAPI Backend activo"}

#Endpoint Principal de Datos
@app.get("/api/item/{item_id}", response_model=ItemResponse)
def read_item(item_id: int) -> ItemResponse:
    """
    Retorna datos de ejemplo basados en el ID solicitado.
    """
    
    # Lógica de negocio simulada (ej. buscar en una base de datos)
    if item_id == 1:
        data = {
            "id": 1,
            "nombre": "Reporte de Ventas Q4",
            "descripcion": "Datos agregados de las ventas trimestrales con métricas clave."
        }
    elif item_id == 2:
        data = {
            "id": 2,
            "nombre": "Análisis de Stock",
            "descripcion": "Detalle del inventario actual y proyecciones futuras."
        }
    else:
        data = {
            "id": item_id,
            "nombre": f"Ítem No Encontrado (#{item_id})",
            "descripcion": "El ID ingresado no corresponde a un ítem conocido."
        }
        
    return ItemResponse(**data)

#Cargar y servir el CSV
CSV_TRANSACCIONES_PATH = os.path.join(os.path.dirname(__file__), "Transacciones 506 julio.csv")
# Usar Pandas para leer el archivo CSV
df = pd.read_csv(CSV_TRANSACCIONES_PATH, delimiter=';')  
print("ARCHIVO CARGADO")  
# 1. Procesamiento Opcional: Agregar una columna nueva
#df['Margen'] = df['Ventas'] * 0.20

@app.get("/api/dashboard/data")  #, response_model=List[Dict[str, Any]])

def get_csv_data():
    try:
       
        df_primeros_20 = df.head(20)   #primeras 20 filas

       
        if 'Ramal' in df.columns and 'Tarifa Cobrada' in df.columns and not df['Ramal'].empty:
            recuento_por_ramal = df.groupby('Ramal')['Tarifa Cobrada'].count()
            if not recuento_por_ramal.empty:
                print("Recuento total de transacciones por cada Ramal:")
                print(recuento_por_ramal.to_string())
            else:
                print("No hay datos válidos en las columnas 'Ramal' o 'Tarifa Cobrada' para realizar el recuento.")
        else:
            print("Las columnas 'Ramal' o 'Tarifa Cobrada' no se encuentran o no tienen datos.")
       
        
        # 2. Conversión a formato JSON compatible (lista de diccionarios)
        # 'records' es un formato eficiente para enviar a un frontend
        #data_to_send = df_primeros_20.to_dict(orient='records')
        
        data_to_send = recuento_por_ramal.to_dict()

        return data_to_send
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Archivo CSV no encontrado en: {CSV_TRANSACCIONES_PATH}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")
    
