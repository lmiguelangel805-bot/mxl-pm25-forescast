"""
Módulo de Predicción de Calidad del Aire (PM2.5)
------------------------------------------------
Descripción:
    Este script carga datos históricos de sensores de calidad del aire,
    realiza ingeniería de características (feature engineering) para series de tiempo,
    y entrena un modelo de regresión (Random Forest) para predecir niveles de PM2.5.

Autor: [Tu Nombre]
Fecha: Diciembre 2025
Fuente de Datos: SINAICA / PurpleAir (Estación Municipal Mexicali)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# CONFIGURACIÓN Y CONSTANTES
# ==========================================
ARCHIVO_DATOS = 'datos.csv'
TARGET = 'pm2.5_atm'
FEATURES = ['hora', 'dia_semana', 'mes', 'pm2.5_lag1']
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ==========================================
# 1. INGESTIÓN Y PREPROCESAMIENTO
# ==========================================
def cargar_y_procesar_datos(ruta_archivo):
    """
    Carga el dataset, ajusta la zona horaria y establece el índice temporal.
    """
    try:
        df = pd.read_csv(ruta_archivo)
        
        # Conversión de timestamp UTC a zona horaria local (Tijuana/Mexicali)
        df['time_stamp'] = pd.to_datetime(df['time_stamp'], utc=True)
        df['time_stamp'] = df['time_stamp'].dt.tz_convert('America/Tijuana')
        
        # Ordenamiento cronológico para asegurar integridad de la serie de tiempo
        df = df.sort_values(by='time_stamp')
        df = df.set_index('time_stamp')
        
        print(f"[INFO] Datos cargados exitosamente. Total de registros: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] No se encontró el archivo: {ruta_archivo}")
        return None

# ==========================================
# 2. INGENIERÍA DE CARACTERÍSTICAS (FEATURE ENGINEERING)
# ==========================================
def generar_features(df):
    """
    Genera variables temporales y de rezago (lags) para el modelo supervisado.
    """
    df_processed = df.copy()
    
    # Extracción de componentes temporales para capturar estacionalidad diaria/semanal
    df_processed['hora'] = df_processed.index.hour
    df_processed['dia_semana'] = df_processed.index.dayofweek
    df_processed['mes'] = df_processed.index.month
    
    # Generación de Lags (Autocorrelación): Valor de t-1 para predecir t
    df_processed['pm2.5_lag1'] = df_processed[TARGET].shift(1)
    
    # Eliminación de valores nulos generados por el desplazamiento (shift)
    df_processed = df_processed.dropna()
    
    print(f"[INFO] Ingeniería de características completada. Registros útiles: {len(df_processed)}")
    return df_processed

# ==========================================
# 3. ENTRENAMIENTO DEL MODELO
# ==========================================
def entrenar_modelo(df):
    """
    Entrena un modelo Random Forest Regressor.
    Nota: shuffle=False es crítico para series de tiempo.
    """
    X = df[FEATURES]
    y = df[TARGET]
    
    # División Train/Test respetando el orden temporal
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False
    )
    
    print(f"[INFO] Iniciando entrenamiento con {len(X_train)} muestras...")
    
    # Inicialización y ajuste del modelo
    modelo = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    modelo.fit(X_train, y_train)
    
    return modelo, X_test, y_test

# ==========================================
# 4. EVALUACIÓN Y VISUALIZACIÓN
# ==========================================
def evaluar_rendimiento(modelo, X_test, y_test):
    """
    Genera métricas de desempeño y gráfica comparativa.
    """
    predicciones = modelo.predict(X_test)
    
    # Cálculo de métricas
    mae = mean_absolute_error(y_test, predicciones)
    r2 = r2_score(y_test, predicciones)
    
    print("-" * 40)
    print("REPORTE DE EVALUACIÓN DEL MODELO")
    print("-" * 40)
    print(f"Métrica MAE (Error Absoluto Medio): {mae:.2f}")
    print(f"Métrica R2 (Coeficiente de Determinación): {r2:.4f}")
    print("-" * 40)
    
    # Visualización de resultados (Zoom a las últimas 120 horas)
    resultados = pd.DataFrame({'Real': y_test, 'Prediccion': predicciones}, index=y_test.index)
    zoom_data = resultados.tail(120)
    
    plt.figure(figsize=(14, 6))
    plt.plot(zoom_data.index, zoom_data['Real'], label='Valor Real (Sensor)', color='#2c3e50', linewidth=2)
    plt.plot(zoom_data.index, zoom_data['Prediccion'], label='Predicción Modelo', color='#e74c3c', linestyle='--', linewidth=2)
    
    plt.title('Validación del Modelo: Realidad vs Predicción (Últimas 120 horas)', fontsize=12)
    plt.xlabel('Fecha / Hora')
    plt.ylabel('Concentración PM2.5')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==========================================
# EJECUCIÓN PRINCIPAL (PIPELINE)
# ==========================================
if __name__ == "__main__":
    # 1. Carga
    df_raw = cargar_y_procesar_datos(ARCHIVO_DATOS)
    
    if df_raw is not None:
        # 2. Procesamiento
        df_clean = generar_features(df_raw)
        
        # 3. Entrenamiento
        modelo_rf, X_test, y_test = entrenar_modelo(df_clean)
        
        # 4. Evaluación
        evaluar_rendimiento(modelo_rf, X_test, y_test)