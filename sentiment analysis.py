#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 11:50:42 2021

@author: jialinshang
"""

#%%% relevent packages & modules
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt



from sklearn                         import tree
from sklearn                         import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model            import LinearRegression
from sklearn.neighbors               import KNeighborsClassifier
from sklearn.naive_bayes             import GaussianNB
from sklearn.model_selection         import RepeatedKFold
from sklearn.metrics                 import confusion_matrix
from sklearn.metrics                 import accuracy_score
from sklearn.pipeline                import Pipeline
from sklearn.model_selection         import GridSearchCV
from sklearn.metrics                 import classification_report

#%%% data preparetion
dta = pd.read_excel('/Users/jialinshang/Desktop/marketing/final project/Oculus_reviews.xlsx').reset_index()

#filter out noise information
dta1 = dta[['index','scrapping_date','one_review_text','review_date','one_review_stars','Rating']]
print(dta1)

#%%% review distribution
Rating_count=dta1['Rating'].value_counts()
plt.pie(Rating_count,labels=Rating_count.index,autopct='%1.1f%%', shadow=False,startangle=140)
plt.show()

#%%% data splitting
np.random.seed(1)
dta1['ML_group']   = np.random.randint(100, size = dta1.shape[0])
dta1               = dta1.sort_values(by = 'ML_group')
#80% for trainning; 10% for validating; 10% for testing
inx_train          = dta1.ML_group  <  80                                
inx_valid          = (dta1.ML_group >= 80) & (dta1.ML_group<90)          
inx_test           = (dta1.ML_group >= 90)

#%%% text vectorization
corpus      = dta1.one_review_text.to_list()
vectorizer  = CountVectorizer(lowercase  = True, ngram_range=(1,1),max_df=0.85,min_df=0.01);
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
vc_mtx=vectorizer.transform(corpus)
vc_mtx.toarray()

#%%% TVT Split
Y_train   = dta1.Rating[inx_train].to_list()
Y_valid   = dta1.Rating[inx_valid].to_list()
Y_test    = dta1.Rating[inx_test ].to_list()

X_train   = X[np.where(inx_train)[0],:]
X_valid   = X[np.where(inx_valid)[0],:]
X_test    = X[np.where(inx_test) [0],:]

# check wheher all reviews are assigned to groups
X_train.shape[0]+X_valid.shape[0]+X_test.shape[0] == 5054

#%%% KNN classification
#To find the best K
###elbow
test_error_rates=[]
k_max=30
for k in range(1,k_max+1):
    # fit the model
    knn_model=KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train,Y_train)
    # y-hat based on x test
    y_pred_test=knn_model.predict(X_test)
    # find the test error
    test_error=1-accuracy_score(Y_test, y_pred_test)
      
    test_error_rates.append(test_error)
# minimum the test error
min(test_error_rates)
best_k=test_error_rates.index(min(test_error_rates))+1

plt.plot(range(1,k_max+1),test_error_rates)
plt.ylabel('Error Rates')
plt.xlabel('K neighbors')

###cross validation
knn=KNeighborsClassifier()
operations=[('knn',knn)]
pipe=Pipeline(operations)
k_max=30
k_values=list(range(1,k_max+1))
param_grid={'knn__n_neighbors':k_values}
full_cv_classifier=GridSearchCV(pipe,param_grid,cv=5,scoring='accuracy')
full_cv_classifier.fit(X_train,Y_train)
full_cv_classifier.best_estimator_.get_params()
full_pred = full_cv_classifier.predict(X_test)
confusion_matrix(Y_test, full_pred)
print(classification_report(Y_test,full_pred))

#predict results
results_list_knn_actual = []
knn      = KNeighborsClassifier(n_neighbors=13).fit(X_train, Y_train)
results_list_knn_actual.append(
        np.concatenate([knn.predict(X_train),
                 knn.predict(X_valid),
                 knn.predict(X_test )]))

results_list_knn_actual              = pd.DataFrame(results_list_knn_actual).transpose()
dta2=dta1
dta2['Predicted_star'] = results_list_knn_actual[0]
dta2 = dta2.sort_values(by='index')
dta2.to_excel('Oculus_reviews.xlsx', index = False)
print(dta2)


np.mean()

        



        









#def knn_model(k_max):
    #results=[]
    #k_max=100
    
    #for k in range(1,k_max+1):
        #knn_model=KNeighborsClassifier(n_neighbors=k)
        #knn_model.fit(X_train,Y_train)
        
        #y_pred_test=knn_model.predict(X_valid) 
        
        #results.append(np.concatenate([y_pred_test]))
    #results_knn=pd.DataFrame(results).transpose()
    #return results_knn


    























