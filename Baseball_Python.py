#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression in Python
# 
# Learning Python

# In[282]:


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression


# In[283]:


os.chdir("C:\\Users\\Matt\\Documents\\Python_Projects")


# In[284]:


pwd


# In[285]:


baseball_train = pd.read_csv(r"baseball_train.csv",index_col=0,
                             dtype={'Opp': 'category', 'Result': 'category', 'Name': 'category'}, header=0)
baseball_test = pd.read_csv(r"baseball_test.csv",index_col=0,
                            dtype={'Opp': 'category', 'Result': 'category', 'Name': 'category'}, header=0)
print(baseball_test.head())
encoded_categories = dict(enumerate(baseball_test.Name.cat.categories))
print(encoded_categories)


# In[286]:


X = baseball_train.iloc[:,:-1]
X = X.drop(['Opp','Result'],axis=1)
y = baseball_train.iloc[:,-1]

# Create logistic regression
logit = LogisticRegression(fit_intercept=True)

# Create repeated kfold
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=21191)

# Do repeated k-fold cross-validation
cv_results = cross_val_score(logit,
                             X,
                             y,
                             cv=rkf,
                             scoring="roc_auc")


# # Repeated K-Fold Cross Validation

# In[287]:


print(cv_results.min())
print(np.percentile(cv_results, 25))
print(cv_results.mean())
print(np.percentile(cv_results, 50))
print(np.percentile(cv_results, 75))
print(cv_results.max())


# In[288]:


model = logit.fit(X,y)

intercept = model.intercept_[0]

print("intercept = {}".format(intercept))
for idx, col_name in enumerate(X.columns):
    print("{} = {}".format(col_name, model.coef_[0][idx]))


# In[289]:


Xnew = baseball_test.iloc[:,:-1]
Xnew = Xnew.drop(['Opp','Result'],axis=1)
yTrue = baseball_test.iloc[:,-1]

# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
#for i in range(len(Xnew)):
#    print("Predicted=%s" % (ynew[i]))
    
baseball = {'predicted': ynew, 'truth': yTrue}
pd.DataFrame(data=baseball)

