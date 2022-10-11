import pickle
import numpy as np

from flask import Flask, request, jsonify

def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]

#Load model
with open('model1.bin', 'rb') as model_file:  
    model = pickle.load(model_file)
model_file.close()

#Load dict
with open('dv.bin', 'rb') as dic_file:  
    dict_vectorizer = pickle.load(dic_file)
dic_file.close()

app = Flask('flask-probability')

@app.route('/predict', methods=['POST'])
def predict():

    customer = request.get_json()

    prediction = predict_single(customer, dict_vectorizer, model)
    
    result = {
        'credit_probability': float(prediction),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)