import numpy as np
import sklearn.neural_network
import pandas
from sklearn.metrics import mean_squared_error


xs =pandas.read_csv("data.csv", usecols =["field_soil_temp_c","field_air_temp_c","field_rh"])
ys = pandas.read_csv("data.csv", usecols =["field_soil_wc"])


model=sklearn.neural_network.MLPRegressor(
    activation='logistic',
    learning_rate_init=0.001,
    solver='sgd',
    learning_rate='invscaling',
    hidden_layer_sizes=(200,),
    verbose=True,
    max_iter=2000,
    tol=1e-6
)

model.fit(xs,ys)
print('Accuracy training : {:.3f}'.format(model.score(xs, ys)))
predicted_scale=model.predict(xs)
print("RMSE", mean_squared_error(ys,predicted_scale))