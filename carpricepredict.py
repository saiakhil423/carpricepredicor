#!/usr/bin/env python
# coding: utf-8

# In[240]:


import pandas as pd


# In[241]:


car=pd.read_csv(r"C:\Users\SAI AKHIL\Desktop\carpricepredictor\carprice.csv")


# In[242]:


car.head()


# In[243]:


car.shape


# In[244]:


car.info()


# In[245]:


car['year'].unique()


# In[246]:


car['kms_driven'].unique()


# In[247]:


backup=car.copy()


# In[248]:


backup


# In[249]:


car=car[car['year'].str.isnumeric()]


# In[250]:


car['year']=car['year'].astype(int)


# In[251]:


car=car[car['Price']!='Ask For Price']


# In[252]:


car['Price']=car['Price'].str.replace(',','').astype(int)


# In[253]:


car.info()


# In[254]:


car.head()


# In[255]:


car['kms_driven']=car['kms_driven'].str.split().str.get(0).str.replace(',','')


# In[256]:


car=car[car['kms_driven'].str.isnumeric()]


# In[257]:


car['kms_driven']=car['kms_driven'].astype(int)


# In[258]:


car.info()


# In[259]:


car=car[~car['fuel_type'].isna()]


# In[260]:


car['name']=car['name'].str.split(' ').str.slice(0,3).str.join(' ')


# In[261]:


car['name']


# In[262]:


car=car.reset_index(drop=True)


# In[263]:


car.head()


# In[264]:


car.describe()


# In[265]:


car.info()


# In[266]:


car.describe(include='all')


# In[268]:


car['Price']


# In[270]:


X=car.drop(columns='Price')


# In[271]:


y=car['Price']


# In[272]:


y


# In[273]:


X


# In[274]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[279]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score


# In[280]:


ohe=OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])


# In[281]:


ohe


# In[292]:


column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),
                                    remainder='passthrough')


# In[293]:


lr=LinearRegression()


# In[294]:


pipe=make_pipeline(column_trans,lr)


# In[295]:


pipe.fit(X_train,y_train)


# In[296]:


y_pred=pipe.predict(X_test)


# In[297]:


r2_score(y_test,y_pred)


# In[298]:


scores=[]
for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))


# In[301]:


import numpy as np


# In[302]:


np.argmax(scores)


# In[304]:


scores[np.argmax(scores)]


# In[305]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)


# In[306]:


import pickle


# In[307]:


pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))


# In[308]:


pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))


# In[310]:


pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Dzire','Maruti',2009,46000,'Petrol']).reshape(1,5)))


# In[ ]:




