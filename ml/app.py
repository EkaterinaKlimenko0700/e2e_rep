from flask import Flask, request, jsonify
import pandas as pd
import joblib
app = Flask(__name__)

list_fields_inout = ['offer_id', 'first_day_exposition', 'last_day_exposition',
       'floor', 'open_plan', 'rooms', 'studio', 'area', 'kitchen_area',
       'living_area', 'agent_fee', 'renovation', 'offer_type', 'category_type',
       'unified_address', 'building_id']

list_filds_for_analis = ['floor', 'rooms', 'area']

@app.route('/api/ml/predict/price', methods=['GET', 'POST'])
def predict_price():
    content = request.json
    
    list_miss_fields = []

    for item in list_fields_inout:
        if item not in content:
            list_miss_fields.append(item)

    if len(list_miss_fields)>0:
        return jsonify({"success":0, "content":content, "error":"miss", "fields":list_miss_fields}),500

    df = pd.json_normalize(content)
    df = df[list_filds_for_analis]
    
    model = joblib.load("rf.joblib")

    result = model.predict(df)

    return jsonify({"success":1, "content":content, "result":result.tolist()}), 200    

if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug=True)
