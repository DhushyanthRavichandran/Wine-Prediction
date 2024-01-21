import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

data=pd.read_csv("D:\ml\wine\WineQT.csv")

data.drop(labels=['Id'],inplace=True,axis=1)

x=data.drop(labels=['quality'],axis=1)
y=data['quality']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=40)
scaler = StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)  # Changed from fit_transform to transform

from sklearn.ensemble import RandomForestClassifier
rnd = RandomForestClassifier(max_depth= 20,
                               min_samples_leaf= 3,
                               min_samples_split= 10,
                               n_estimators= 100)
fit_rnd = rnd.fit(x_train,y_train)
rnd_score = rnd.score(x_test,y_test)
print('score of model is : ',rnd_score)
x_predict = list(rnd.predict(x_test))

# Save the model to a file
with open('winequality.pkl', 'wb') as f:
    pickle.dump(rnd, f)