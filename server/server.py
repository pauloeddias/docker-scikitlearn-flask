import flask
from flask import request
import joblib
import numpy as np

app = flask.Flask(__name__)
app.config["DEBUG"] = False

model = joblib.load('model.joblib')


@app.route('/', methods=['GET'])
def home():
    x = request.args.get('x')
    x = np.array(x, dtype=np.float64).reshape(1,-1)
    y = model.predict(x)
    y = y[0][0]
    return str(y)

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=80)
      