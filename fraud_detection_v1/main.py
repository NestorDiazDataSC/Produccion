# main.py - VERSIÓN CORRECTA CON 31 COLUMNAS (incluye score_if)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import numpy as np
from typing import List, Optional
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API Detección de Fraude Bancario",
    description="Modelo híbrido: XGBoost + Isolation Forest (con score_if)",
    version="2.1.0"
)

# 1. Cargar modelo y nombres de columnas
try:
    model = joblib.load('modelo_fraude.pkl')
    feature_names = joblib.load('features_hybrid.pkl')  # 31 columnas
    logger.info(f"✅ Modelo cargado. Espera {len(feature_names)} columnas")
    logger.info(f"📋 Columnas: {feature_names}")
except Exception as e:
    logger.error(f"❌ Error al cargar modelo: {e}")
    raise e

# 2. Definir esquema de entrada (31 columnas: 30 originales + score_if)
class TransactionInput(BaseModel):
    Time: float = Field(..., description="Tiempo transcurrido (segundos)")
    V1: float = Field(..., description="Componente PCA #1")
    V2: float = Field(..., description="Componente PCA #2")
    V3: float = Field(..., description="Componente PCA #3")
    V4: float = Field(..., description="Componente PCA #4")
    V5: float = Field(..., description="Componente PCA #5")
    V6: float = Field(..., description="Componente PCA #6")
    V7: float = Field(..., description="Componente PCA #7")
    V8: float = Field(..., description="Componente PCA #8")
    V9: float = Field(..., description="Componente PCA #9")
    V10: float = Field(..., description="Componente PCA #10")
    V11: float = Field(..., description="Componente PCA #11")
    V12: float = Field(..., description="Componente PCA #12")
    V13: float = Field(..., description="Componente PCA #13")
    V14: float = Field(..., description="Componente PCA #14")
    V15: float = Field(..., description="Componente PCA #15")
    V16: float = Field(..., description="Componente PCA #16")
    V17: float = Field(..., description="Componente PCA #17")
    V18: float = Field(..., description="Componente PCA #18")
    V19: float = Field(..., description="Componente PCA #19")
    V20: float = Field(..., description="Componente PCA #20")
    V21: float = Field(..., description="Componente PCA #21")
    V22: float = Field(..., description="Componente PCA #22")
    V23: float = Field(..., description="Componente PCA #23")
    V24: float = Field(..., description="Componente PCA #24")
    V25: float = Field(..., description="Componente PCA #25")
    V26: float = Field(..., description="Componente PCA #26")
    V27: float = Field(..., description="Componente PCA #27")
    V28: float = Field(..., description="Componente PCA #28")
    Amount: float = Field(..., description="Monto de la transacción (USD)", ge=0)
    score_if: float = Field(..., description="Score de anomalía de Isolation Forest")

    class Config:
        schema_extra = {
            "example": {
                "Time": 0.0,
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.025791,
                "V19": 0.403993,
                "V20": 0.251412,
                "V21": -0.018307,
                "V22": 0.277838,
                "V23": -0.110474,
                "V24": 0.066928,
                "V25": 0.128539,
                "V26": -0.189115,
                "V27": 0.133558,
                "V28": -0.021053,
                "Amount": 149.62,
                "score_if": 0.35
            }
        }

class PredictionResponse(BaseModel):
    es_fraude: bool
    probabilidad_fraude: float
    mensaje: str
    score_anomalia: Optional[float] = None

# 3. Endpoint de salud
@app.get("/")
def health_check():
    return {
        "status": "online",
        "modelo": "XGBoost + Isolation Forest (Híbrido)",
        "features_count": len(feature_names),
        "features": feature_names,
        "version": "2.1.0"
    }

# 4. Función para convertir input a DataFrame
def transaction_to_dataframe(transaction: TransactionInput) -> pd.DataFrame:
    """Convierte transacción a DataFrame con el orden correcto de columnas"""
    data = [[
        transaction.Time,
        transaction.V1, transaction.V2, transaction.V3, transaction.V4,
        transaction.V5, transaction.V6, transaction.V7, transaction.V8,
        transaction.V9, transaction.V10, transaction.V11, transaction.V12,
        transaction.V13, transaction.V14, transaction.V15, transaction.V16,
        transaction.V17, transaction.V18, transaction.V19, transaction.V20,
        transaction.V21, transaction.V22, transaction.V23, transaction.V24,
        transaction.V25, transaction.V26, transaction.V27, transaction.V28,
        transaction.Amount,
        transaction.score_if
    ]]
    return pd.DataFrame(data, columns=feature_names)

# 5. Endpoint de predicción
@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    try:
        # Convertir a DataFrame
        input_data = transaction_to_dataframe(transaction)
        
        # Verificar columnas
        if list(input_data.columns) != feature_names:
            return {
                "error": f"Orden de columnas incorrecto. Esperado: {feature_names}"
            }
        
        # Predecir
        prediccion = int(model.predict(input_data)[0])
        probabilidad = float(model.predict_proba(input_data)[0][1])
        
        mensaje = "🚨 ALERTA: Transacción Fraudulenta" if prediccion else "✅ Transacción Segura"
        nivel_riesgo = "Alto" if probabilidad > 0.7 else "Medio" if probabilidad > 0.3 else "Bajo"
        
        return PredictionResponse(
            es_fraude=bool(prediccion),
            probabilidad_fraude=probabilidad,
            mensaje=f"{mensaje} (Riesgo: {nivel_riesgo})",
            score_anomalia=transaction.score_if
        )
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

# 6. Endpoint para predicción por lote
@app.post("/predict_batch")
def predict_batch(transactions: List[TransactionInput]):
    try:
        if not transactions:
            raise HTTPException(status_code=400, detail="Lista vacía")
        
        data = []
        for t in transactions:
            data.append([
                t.Time, t.V1, t.V2, t.V3, t.V4, t.V5, t.V6, t.V7, t.V8, t.V9,
                t.V10, t.V11, t.V12, t.V13, t.V14, t.V15, t.V16, t.V17, t.V18,
                t.V19, t.V20, t.V21, t.V22, t.V23, t.V24, t.V25, t.V26, t.V27,
                t.V28, t.Amount, t.score_if
            ])
        
        df_batch = pd.DataFrame(data, columns=feature_names)
        predicciones = model.predict(df_batch)
        probabilidades = model.predict_proba(df_batch)[:, 1]
        
        return [
            {
                "es_fraude": bool(pred),
                "probabilidad_fraude": float(prob)
            }
            for pred, prob in zip(predicciones, probabilidades)
        ]
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en lote: {str(e)}")

# 7. Punto de entrada
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )