from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier  # Importa el modelo específico

app = FastAPI()

# Cargar el modelo de tu archivo .pkl
with open("tree.pkl", "rb") as model_file:
    modelo = pickle.load(model_file)

# Definir una clase para recibir los datos de entrada
from pydantic import BaseModel

class DatosEntrada(BaseModel):
    MTRANS_Automobile: float
    MTRANS_Bike: float
    MTRANS_Motorbike: float
    MTRANS_Public_Transportation: float
    MTRANS_Walking: float
    female: int
    Age: float
    Height: float
    Weight: float
    family_history_overweight: int
    FAVC: int
    FCVC: float
    NCP: float
    CAEC: float
    SMOKE: int
    CH2O: float
    SCC: int
    FAF: float
    TUE: float
    CALC: float

# Definir una ruta para hacer predicciones
@app.post("/predecir_obesidad/")
def predecir_obesidad(datos: DatosEntrada):
    # Convertir los datos en un array NumPy
    datos_array = np.array([[
        datos.MTRANS_Automobile, datos.MTRANS_Bike, datos.MTRANS_Motorbike,
        datos.MTRANS_Public_Transportation, datos.MTRANS_Walking,
        datos.female, datos.Age, datos.Height, datos.Weight,
        datos.family_history_overweight, datos.FAVC, datos.FCVC,
        datos.NCP, datos.CAEC, datos.SMOKE, datos.CH2O,
        datos.SCC, datos.FAF, datos.TUE, datos.CALC
    ]])

    # Hacer la predicción usando el modelo
    prediccion = modelo.predict(datos_array)

    # Devolver la categoría de obesidad
    categorias = ['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III', 'Overweight_Level_I', 'Overweight_Level_II']
    
    if 0 <= prediccion < len(categorias):
        categoria_obesidad = categorias[prediccion]
    else:
        categoria_obesidad = 'Categoría no válida'

    return {"categoria_obesidad": categoria_obesidad}
