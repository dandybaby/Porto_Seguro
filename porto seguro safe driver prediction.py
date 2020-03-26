import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

n_estimator = 10
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
NUMERIC_COLS = ["ps_reg_01", "ps_reg_02", "ps_reg_03", "ps_car_12",
                "ps_car_13", "ps_car_14", "ps_car_15"]
# 将样本集分成测试集和训练集
X_train, X_test, y_train, y_test = train_test_split(
    train[NUMERIC_COLS], train['target'], test_size=0.3)
# 再将训练集拆成两个部分（GBDT/RF，LR）
X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)

# 基于GBDT监督变换
grd = GradientBoostingClassifier(n_estimators=n_estimator)
grd.fit(X_train, y_train)
# 得到OneHot编码
grd_enc = OneHotEncoder(categories='auto')

temp = grd.apply(X_train)
np.set_printoptions(threshold=np.inf)
grd_enc.fit(grd.apply(X_train)[:, :, 0])
# print(grd_enc.get_feature_names()) # 查看每一列对应的特征
# 使用OneHot编码作为特征，训练LR
grd_lm = LogisticRegression(solver='lbfgs', max_iter=1000)
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
# 使用LR进行预测
y_pred_grd_lm = grd_lm.predict_proba(grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]

NE = (-1) / len(y_pred_grd_lm) * sum(
    ((1 + y_test) / 2 * np.log(y_pred_grd_lm[:, 1]) + (1 - y_test) / 2 * np.log(1 - y_pred_grd_lm[:, 1])))
print("Normalized Cross Entropy " + str(NE))
