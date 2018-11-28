import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import lightgbm as lgb

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

zhengqi_train = pd.read_table('./zhengqi_train.txt',encoding='utf-8')
zhengqi_test = pd.read_table('./zhengqi_test.txt',encoding='utf-8')

X = np.array(zhengqi_train.drop(['target'], axis = 1))
y = np.array(zhengqi_train.target)

# print('================================')
# print(X.shape)
# print(y.shape)
# print('================================')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(len(X_train))
print(len(X_test))

pca = PCA(n_components=0.97)
pca.fit(X)
X_pca = pca.transform(X)
X1_pca = pca.transform(zhengqi_test)

print(X_pca.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X_pca, y, test_size=0.3, random_state=0)

clfL = linear_model.LinearRegression()

clfL.fit(X_train,Y_train)

y_true, y_pred = Y_test, clfL.predict(X_test)

print(mean_squared_error(y_true, y_pred))

ans_Liner = clfL.predict(X1_pca)
print(ans_Liner.shape)
df = pd.DataFrame(ans_Liner)
df.to_csv('./LineR&PCA.txt',index=False,header=False)


clfGB  = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                                  learning_rate=0.03, loss='huber', max_depth=14,
                                  max_features='sqrt', max_leaf_nodes=None,
                                  min_impurity_decrease=0.0, min_impurity_split=None,
                                  min_samples_leaf=10, min_samples_split=40,
                                  min_weight_fraction_leaf=0.0, n_estimators=300,
                                  presort='auto', random_state=10, subsample=0.8, verbose=0,
                                  warm_start=False)
clfGB.fit(X_train,Y_train)
y_true, y_pred = Y_test, clfGB.predict(X_test)

print(mean_squared_error(y_true, y_pred))

ans_Liner = clfGB.predict(X1_pca)
print(ans_Liner.shape)
df = pd.DataFrame(ans_Liner)
df.to_csv('./GB&PCA.txt',index=False,header=False)