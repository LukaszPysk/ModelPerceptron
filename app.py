from flask import Flask, jsonify, request
import numpy as np
from sklearn.linear_model import Perceptron

app = Flask(__name__)


# [Staż pracownika w firmie, Wynik oceny pracownika]
X = np.array([
    [1, 5], [2, 6], [2, 7], [3, 6], [3, 7],
    [4, 7], [5, 6], [6, 7], [6, 8], [7, 7],
    [8, 7], [9, 8], [10, 9], [11, 7], [12, 8],
    [13, 9], [14, 9], [15, 6], [16, 9], [17, 10]
])

# [0 - brak awansu, 1 - awans]
y = np.array([
    0, 0, 0, 0, 1, 
    0, 0, 1, 1, 0, 
    0, 1, 1, 0, 1, 
    1, 1, 0, 1, 1
])

model = Perceptron()
model.fit(X, y)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    X_new = np.array(data["input"])
    prediction = model.predict(X_new)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
