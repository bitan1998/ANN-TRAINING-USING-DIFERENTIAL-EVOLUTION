import numpy as np
import sklearn.neural_network
import pandas
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt



xs =pandas.read_csv("data.csv", usecols =["field_soil_temp_c","field_air_temp_c","field_rh"],dtype=np.float64)
ys = pandas.read_csv("data.csv", usecols =["field_soil_wc"],dtype=np.float64).values.ravel()


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

x=model.fit(xs,ys)
print('Accuracy training : {:.3f}'.format(model.score(xs, ys)))
print('\nprediction:')
predicted_scale=model.predict(xs)
print(predicted_scale)
print("RMSE", mean_squared_error(ys,predicted_scale))
p1=np.polyfit(predicted_scale,ys,1)
plt.plot(np.polyval(p1,predicted_scale),predicted_scale,'r--',label='Expected_Output')
plt.xlabel('target')
plt.ylabel('output')
#plt.ylim((0,1))
plt.scatter(ys,predicted_scale)
plt.plot(ys,ys,label='Output')
plt.legend(loc='upper left')
plt.show()