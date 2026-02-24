import pandas as pd

# 1. Cargas tu CSV actual (con tus parámetros)
df = pd.read_csv("./api/Transacciones 506 julio.csv", delimiter=';', decimal=',', encoding='utf-8', parse_dates=['Fecha Hora'], dayfirst=True)

# 2. Lo guardas como Parquet (usando compresión snappy que es la estándar)
df.to_parquet("Transacciones.parquet", compression='snappy')

print("¡Listo!")