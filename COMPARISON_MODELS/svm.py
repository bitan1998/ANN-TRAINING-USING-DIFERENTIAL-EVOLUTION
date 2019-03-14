import numpy as np
from sklearn.svm import SVR
import pandas
from sklearn.metrics import mean_squared_error

dataset1= pandas.read_csv("data.csv")

X=dataset1.iloc[:,:-1].values # REJECTING THE LAST COLUMN
y=dataset1.iloc[:,3].values

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

svr_poly = SVR(kernel='poly', C=1e3, degree=6)


from sklearn.preprocessing import StandardScaler ## BEST FOR  CLASSIFICATION TYPE MODEL
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 

y_poly = svr_poly.fit(X_train, y_train).predict(X_test)


print("\n||MODEL SCORE||\n")
print('MODEL_POLY:',svr_poly.score(X_train,y_train))

z=np.std(np.square(np.subtract(y_test,y_poly)))

print("\n||MODEL RMSE||\n")
print('RMSE_POLY::',z)
