import  pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('D:\\ml\\wine\\winequalityN.csv')
df_clean = df.dropna()
y = df_clean['type']
X = df_clean.drop('type', axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import MinMaxScaler
scl = MinMaxScaler().fit(X_train)
X_train_norm = scl.transform(X_train)
X_test_norm = scl.transform(X_test)



le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
from xgboost import XGBClassifier 
xgb = XGBRegressor(n_estimators=200, random_state=0)

# rnd = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
# fit_rnd = rnd.fit(X_train, )




fit_rnd = xgb.fit(X_train,y_train_encoded)

rnd_score = xgb.score(X_test, y_test_encoded)
print('score of model is : ',rnd_score)
x_predict = list(xgb.predict(X_test))

import joblib

model_filename = 'winequality.joblib'
joblib.dump(xgb, model_filename)

