from keras.models import Sequential
from keras.layers.core import Dense , Dropout , Activation
from keras.optimizers import SGD
import numpy as np
import pandas 

X =pandas.read_csv("data.csv", usecols =["field_soil_temp_c","field_air_temp_c","field_rh"])
Y = pandas.read_csv("data.csv", usecols =["field_soil_wc"])

model = Sequential()
model.add(Dense(8, input_dim=3))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.6, momentum=0.6)
model.compile(loss= 'mean_squared_error', optimizer=sgd)

model.fit(X,Y,verbose=1,batch_size=4,nb_epoch=1000)
print(model.predict_proba(X))
loss = np.subtract(Y,model.predict_proba(X))
print('Loss:\n',loss)
square= np.square(loss)
RMSE=np.std(square)
print('\nRMSE:',RMSE)
