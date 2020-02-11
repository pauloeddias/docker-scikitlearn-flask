import numpy as np
from sklearn.linear_model import LinearRegression

n=1000
x = np.linspace(0,10,n)
y = 0.5*x+2
noise = np.random.normal(0,0.1,n)
y = y + noise

x = x.reshape(-1,1)
y = y.reshape(-1,1)
model = LinearRegression()
model.fit(x,y)
yhat = model.predict(x)

import joblib
joblib.dump(model,'../server/model.joblib')