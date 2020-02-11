import flask
from flask import request
import joblib
import numpy as np

app = flask.Flask(__name__)
app.config["DEBUG"] = True

model = joblib.load('../train/model.joblib')


@app.route('/', methods=['GET'])
def home():
    x = request.args.get('x')
    x = np.array(x, dtype=np.float64).reshape(1,-1)
    y = model.predict(x)
    # y = y.tolist()
    y = y[0][0]
    return str(y)
app.run()