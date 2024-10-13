from flask import Flask, request, jsonify
import numpy as np
import joblib
app = Flask(__name__)

model = joblib.load ("f.joblib")
sc_x = joblib.load ("x.joblib")
sc_y = joblib.load ("y.joblib")

@app.route('/api/ml/predict/price', methods=['GET'])
def predict_price():
    args = request.args

    floor = args.get('floor', default = -1, type=int)
    rooms = args.get('rooms', default = -1, type=int)
    area = args.get ('area', default = -1, type=float)
    renovation = args.get('renovation', default = -1, type=int)

    if floor == -1 or rooms == -1 or area == -1 or renovation == -1:
        return jsonify({"success":0, "error":"floor, rooms ,area, renovation are required"}),500
    
    x = np.array([floor, rooms, area, renovation]).reshape (1,-1)
    x = sc_x.transform(x)
    result = model.predict(x)

    result = sc_y.inverse_transform(result.reshape (1, -1))
    return str(result[0][0])

if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug=True)
