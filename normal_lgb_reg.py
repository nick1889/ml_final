import numpy as np
from sklearn.preprocessing import normalize
import lightgbm as lgb
from sklearn.externals import joblib
from sklearn.multioutput import MultiOutputRegressor
x_test = np.load('data/X_test.npz')['arr_0']
x_train = np.load('data/X_train.npz')['arr_0']
y_train = np.load('data/Y_train.npz')['arr_0']

for i in range(4999,0,-1):
    x_train[:,i] = x_train[:,i]-x_train[:,i-1]
    x_test[:,i] = x_test[:,i]-x_test[:,i-1]
for i in range(50):
    for j in range(99,0,-1):
        x_train[:,5000+50*i+j] = x_train[:,5000+50*i+j] - x_train[:,5000+50*i+j-1]
        x_test[:,5000+50*i+j] = x_test[:,5000+50*i+j] - x_test[:,5000+50*i+j-1]
y_train[:,0] = y_train[:,0]*300;
y_train[:,2] = y_train[:,2]*200;

x_train = normalize(x_train)
x_test = normalize(x_test)
bst = MultiOutputRegressor(lgb.LGBMRegressor(objective='mae')).fit(x_train,y_train)
y_pred_lightgbm = bst.predict(x_test)
y_pred_lightgbm[:,0] = y_pred_lightgbm[:,0]/300;
y_pred_lightgbm[:,2] = y_pred_lightgbm[:,2]/200;
np.savetxt('normal_lightgbm_reg.csv',y_pred_lightgbm,delimiter=',')
joblib.dump(bst, 'normal_bst.pkl')