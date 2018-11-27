from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import lightgbm as lgb

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

zhengqi_train = pd.read_table('./zhengqi_train.txt', encoding='utf-8')
zhengqi_test = pd.read_table('./zhengqi_test.txt', encoding='utf-8')

zhengqi_test = np.array(zhengqi_test)

X = np.array(zhengqi_train.drop(['target'], axis=1))
y = np.array(zhengqi_train.target)

# print('================================')
# print(X.shape)
# print(y.shape)
# print('================================')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 选择K个最好的特征，返回选择特征后的数据
import numpy as np
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr

# 选择K个最好的特征，返回选择特征后的数据
# 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
# 参数k为选择的特征个数
# SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)


x_train = SelectKBest(lambda X, Y: np.array(list(map(lambda x: pearsonr(x, Y), X.T))).T[0], k=25).fit(X_train,
                                                                                                      y_train).get_support(
    indices=True)

print(X_train[:, x_train].shape)

clfL = linear_model.LinearRegression()

clfL.fit(X_train[:, x_train],y_train)

y_true, y_pred = y_test, clfL.predict(X_test[:, x_train])

print(mean_squared_error(y_true, y_pred))

ans_Liner = clfL.predict(zhengqi_test[:, x_train])
print(ans_Liner.shape)
df = pd.DataFrame(ans_Liner)
df.to_csv('./LineR&KBEST.txt',index=False,header=False)
