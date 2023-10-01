from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS, cross_origin
app = Flask(__name__)
CORS(app)
@cross_origin
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos JSON del cuerpo de la solicitud
        json_data = request.get_json()

        # Extraer los valores numéricos de las claves del JSON
        medidas = [
            json_data.get("MTRANS_Automobile", 0.0),
            json_data.get("MTRANS_Bike", 0.0),
            json_data.get("MTRANS_Motorbike", 0.0),
            json_data.get("MTRANS_Public_Transportation", 0.0),
            json_data.get("MTRANS_Walking", 0.0),
            int(json_data.get("female", 0)),
            float(json_data.get("Age", 0.0)),
            float(json_data.get("Height", 0.0)),
            float(json_data.get("Weight", 0.0)),
            int(json_data.get("family_history_overweight", 0)),
            int(json_data.get("FAVC", 0)),
            float(json_data.get("FCVC", 0.0)),
            float(json_data.get("NCP", 0.0)),
            float(json_data.get("CAEC", 0.0)),
            int(json_data.get("SMOKE", 0)),
            float(json_data.get("CH2O", 0.0)),
            int(json_data.get("SCC", 0)),
            float(json_data.get("FAF", 0.0)),
            float(json_data.get("TUE", 0.0)),
            float(json_data.get("CALC", 0.0))
        ]

        # Cargar el modelo
        clf = joblib.load('tree.pkl')

        # Realizar la predicción
        prediccion = clf.predict([medidas])

        # Devolver la predicción como respuesta en formato JSON
        return jsonify({"categoria_obesidad": prediccion[0]})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
