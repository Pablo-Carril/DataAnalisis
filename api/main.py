# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

# 1. Definición del Esquema de Datos
# Usamos Pydantic para definir la estructura de los datos que la API envía.
class ItemResponse(BaseModel):
    id: int
    nombre: str
    descripcion: str
    
# 2. Inicialización de FastAPI
app = FastAPI(
    title="API de Backend con FastAPI",
    version="1.0.0",
)

# 3. Endpoint de Salud (Health Check)
@app.get("/")
def health_check() -> Dict[str, str]:
    """Ruta simple para verificar que la API está viva."""
    return {"status": "ok", "service": "FastAPI Backend activo"}

# 4. Endpoint Principal de Datos
@app.get("/api/v1/item/{item_id}", response_model=ItemResponse)
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