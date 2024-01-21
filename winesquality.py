
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import f1_score, accuracy_score,classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv("D:\\ml\\wine\\winequalityN.csv")

np.unique(df.quality.values.tolist())


df.drop_duplicates(inplace=True)
le = LabelEncoder()
df.type = le.fit_transform(df.type.values)

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep].astype(np.float64)

df = clean_dataset(df)


features = df.drop(['type'],axis=1)
labels = df['type']

scaler = StandardScaler()
features = scaler.fit_transform(features)
X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=0.2, random_state=42)

from catboost import CatBoostRegressor

import xgboost as xgb
rnd =xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

fit_rnd = rnd.fit(X_train,y_train)
rnd_score = rnd.score(X_test,y_test)
print('score of model is : ',rnd_score)
x_predict = list(rnd.predict(X_test))

import joblib

model_filename = 'winesquality.joblib'
joblib.dump(rnd, model_filename)
