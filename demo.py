import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
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

pca = PCA(n_components=0.9)
pca.fit(X)
X_pca = pca.transform(X)
X1_pca = pca.transform(zhengqi_test)

X_train, X_test, Y_train, Y_test = train_test_split(X_pca, y, test_size=0.3, random_state=0)

clfL = linear_model.LinearRegression()

clfL.fit(X_train,Y_train)

y_true, y_pred = Y_test, clfL.predict(X_test)

print(mean_squared_error(y_true, y_pred))

ans_Liner = clfL.predict(X1_pca)
print(ans_Liner.shape)
df = pd.DataFrame(ans_Liner)
df.to_csv('./LineR&PCA.txt',index=False,header=False)