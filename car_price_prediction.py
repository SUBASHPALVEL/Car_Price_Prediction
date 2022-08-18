import pandas as pd
df=pd.read_csv('car_data.csv')
final_dataset=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven','Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_dataset['Current_Year']=2022
final_dataset['no_of_year']=final_dataset['Current_Year']-final_dataset['Year']
final_dataset.drop(['Year'],axis=1,inplace=True)
final_dataset.drop(['Current_Year'],axis=1,inplace=True)
final_dataset=pd.get_dummies(final_dataset,drop_first=True)
final_dataset.corr()

X=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]

# ordering of features importance
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestRegressor
rf_random = RandomForestRegressor()

# Hyperparameters
# Randomized Search CV
# Number Of trees in  random forest
import numpy as np

n_estimators=[int(x) for x in np.linspace(start = 100,stop = 1200,num = 12)]

#Number of features to consider at every split
max_features=['auto','sqrt']

# Maximum number of levels in a tree
max_depth =[int(x) for x in np.linspace(5, 30,num =6)]

# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split =[2,5,10,15,100]

# Minimum number of samples required to split each leaf node
min_samples_leaf = [1,2,5,10]



from sklearn.model_selection import RandomizedSearchCV

# create random grid
random_grid = {'n_estimators':n_estimators,
               'max_features':max_features,
               'max_depth':max_depth,
               'min_samples_split':min_samples_split,
               'min_samples_leaf':min_samples_leaf}

print(random_grid)

# use random grid  to search for best hyperparameters
# first create the base model to tune
rf=RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf,param_distributions = random_grid,scoring ='neg_mean_squared_error',n_iter=10,cv=5,verbose=2,random_state=42,n_jobs =1)

rf_random.fit(X_train,y_train)

predictions = rf_random.predict(X_test)

import pickle
# open a file where you want to store data
file=open('random_forest_regression_model.pkl' , 'wb')

#dump information to that file
pickle.dump(rf_random,file)
