import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

x_test = np.load('data/X_test.npy')
X = np.load('data/X.npy')
Y = np.load('data/Y_train.npz')['arr_0']

Y[:,0] = Y[:,0]*300
Y[:,2] = Y[:,2]*200

y_pred_val = []
y_pred = []

for i in range(3):
	x_train, x_val, y_train, y_val = train_test_split(X,Y[:,i], test_size=0.05)
	lgb_train = lgb.Dataset(x_train,y_train)
	lgb_eval = lgb.Dataset(x_val,y_val,reference=lgb_train)
	params = {
	    'boosting_type': 'gbdt',
	    'objective': 'regression_l1',
	    'metric': {'l1'},
	    'num_leaves': 200,
	    'learning_rate':0.05,
	    'feature_fraction': 0.9
	}
	learning_rate_decay = lgb.reset_parameter(learning_rate=lambda iter: np.max([0.05 * (0.996 ** iter), 0.00001]))

	print('Starting training...','index:',i)
	# train
	gbm = lgb.train(params,
		            lgb_train,
	                num_boost_round=3000,
	                init_model='model/model'+str(i)+'1.txt',
	                valid_sets=lgb_eval,
	                early_stopping_rounds=100,
	                callbacks=[learning_rate_decay])

	print('Saving model...','index:',i)
	# save model to file
	gbm.save_model('model'+str(i)+'new.txt')

	print('Starting predicting...')
	# predict
	pred_val = gbm.predict(x_val, num_iteration=gbm.best_iteration)
	pred_test = gbm.predict(x_test, num_iteration=gbm.best_iteration)
	y_pred_val.append(pred_val)
	y_pred.append(pred_test)
	# eval
	print('The mae of prediction for ',i,' is:', mean_absolute_error(y_val, pred_val))

result = np.asarray(y_pred)
result = result.T
print(result.shape)
result[:,0] = result[:,0]/300
result[:,2] = result[:,2]/200
result[:,0] = np.clip(result[:,0], 0, 1)
result[:,1] = np.clip(result[:,1], 25, 250)
result[:,2] = np.clip(result[:,2], 0.5, 1)
np.savetxt('pred_lightGBM_16.csv',result,delimiter=',')









