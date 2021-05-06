#!/usr/bin/env python
# coding: utf-8

# # Heart disease prediction

# In[20]:


# import the basic packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# ## Read Data

# In[21]:


data = pd.read_csv('Heart.csv')
data.head()


# ## Basic checks and EDA

# In[22]:


data.info()#To print a concise summary of a DataFrame
data.describe() # distribution of the features in the dataset


# In[23]:


data.shape


# In[24]:


data['heart_disease_present'].value_counts()


# In[25]:


data.isnull().sum()


# In[26]:


import seaborn as sns
from matplotlib import pyplot as plt


# In[27]:


# check the target = 'heart_disease_present' distribution

plt.rcParams['figure.figsize'] = (8, 6)

sns.countplot(x='heart_disease_present', data=data);


# ##### Let's see how much males and females are there in the dataset who have a heart problem

# In[28]:


data.groupby(by=['sex', 'heart_disease_present'])['heart_disease_present'].count()


# In[29]:


pd.crosstab(data['sex'], data['heart_disease_present'])


# In[30]:


sns.catplot(x='sex', col='heart_disease_present', kind='count', data=data);


# In[31]:


print("% of women suffering from heart disease: " , data.loc[data.sex == 0].heart_disease_present.sum()/data.loc[data.sex == 0].heart_disease_present.count())
print("% of men suffering from heart disease:   " , data.loc[data.sex == 1].heart_disease_present.sum()/data.loc[data.sex == 1].heart_disease_present.count())


# ##### Percentage of Males is more in this dataset who have a heart disease.

# In[32]:


data.groupby(by=['chest_pain_type', 'heart_disease_present'])['heart_disease_present'].count()


# In[33]:


pd.crosstab(data['chest_pain_type'], data['heart_disease_present']).style.background_gradient(cmap='autumn_r')


# In[34]:


sns.catplot(x='chest_pain_type', col='heart_disease_present', kind='count', data=data);


# #### Patients who had chest pain type 4 is more in the category of people with disease. Also, chest pain type 3 is not that serious as there are many people (~45) who had chest pain type 3 without heart disease.

# ##### Let's see the fbs feature now, fasting blood sugar  (1 = true; 0 = false)

# In[35]:


data.groupby(by=['fasting_blood_sugar_gt_120_mg_per_dl', 'heart_disease_present'])['heart_disease_present'].count()


# In[36]:


sns.catplot(x='fasting_blood_sugar_gt_120_mg_per_dl', col='heart_disease_present', kind='count', data=data);


# In[37]:


data.groupby(by=['thal', 'heart_disease_present'])['heart_disease_present'].count()


# In[38]:


sns.catplot(x='thal', col='heart_disease_present', kind='count', data=data);


# #### Most of the people with heart disease have thal as reversible defect.. Interesting.

# #### Let's check the exang feature
# 
# #### exercise induced angina (1 = yes; 0 = no)

# In[39]:


data.groupby(by=['exercise_induced_angina', 'heart_disease_present'])['heart_disease_present'].count()


# In[40]:


sns.catplot(x='exercise_induced_angina', col='heart_disease_present', kind='count', data=data);


# #### People with exercise induced angina is more in the category with disease.

# #### Let's go for scatter matrix for the continous variables, rather than plotting each pair 

# In[41]:


sns.pairplot(data[['serum_cholesterol_mg_per_dl', 'age', 'resting_blood_pressure', 'max_heart_rate_achieved', 'heart_disease_present']], hue='heart_disease_present');


# #### There is no notable relationship among the features that we can find from the scatter. Let's check the corrleation among these features.

# In[42]:


import seaborn as sns
#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# #### As expected, there is no notable correlation among these features.
# 
# ##### This the end of basic EDA on Heart Disease Dataset.

# In[43]:


X = data.iloc[:,1:-1]
y = data.heart_disease_present


# In[44]:


enc = LabelEncoder()
X.thal = enc.fit_transform(X.thal)


# ## Training data

# In[45]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[46]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Model 1: Logistic Regression 

# In[47]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
alpha = [10**x for x in range(-4,4,1)]
acc = []
for c in alpha:
    classifier = LogisticRegression(penalty='l2',tol=0.0001,C=c,max_iter=1000)
    classifier.fit(X_train,y_train)
    y_predict = classifier.predict(X_test)
    accuracy = accuracy_score(y_predict,y_test)
    acc.append(accuracy)
    print(accuracy)
    print(c)

    


# In[48]:


classifier = LogisticRegression(penalty='l2',tol=0.0001,C=0.01,max_iter=1000)
classifier.fit(X_train,y_train)
y_pred= classifier.predict(X_test)
print(classification_report(y_test, y_predict))


# In[49]:


y_pred_train = classifier.predict(X_train)
cf_test = confusion_matrix(y_pred, y_test)
cf_train = confusion_matrix(y_pred_train, y_train)
cf_train


# In[50]:


print('Accuracy for training set for Logistic Regression = {}'.format((cf_train[0][0] + cf_train[1][1])/len(y_train)))
print('Accuracy for test set for Logistic Regression = {}'.format((cf_test[0][0] + cf_test[1][1])/len(y_test)))


# # Model 2: XGBoost

# In[52]:


print(f"Best paramters: {best_params}")


# In[53]:


rf = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
rf.fit(X_train, y_train)


# In[54]:


y_pred=rf.predict(X_test)
y_pred_train=rf.predict(X_train)


# In[ ]:


cf_test = confusion_matrix(y_pred, y_test)
cf_train = confusion_matrix(y_pred_train, y_train)
cf_train


# In[ ]:


print('Accuracy for training set for Xgboost = {}'.format((cf_train[0][0] + cf_train[1][1])/len(y_train)))
print('Accuracy for test set for Xgboost = {}'.format((cf_test[0][0] + cf_test[1][1])/len(y_test)))


# # Model 3: SVM

# In[55]:


from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC 
# defining parameter range 
param_grid = {'C': [0.1,1.0, 1, 10,42],  
              'gamma': [1,0.1,'scale','auto'], 
              'kernel': ['rbf','linear','sigmoid','poly']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(X_train, y_train) 
#grid_predictions = grid.predict(X_test) 
# print best parameter after tuning 


# In[56]:


print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 


# In[57]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',C=0.1,gamma='scale')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_predict))


# In[58]:


y_pred_train = classifier.predict(X_train)
cf_test = confusion_matrix(y_pred, y_test)
cf_train = confusion_matrix(y_pred_train, y_train)
cf_train


# In[59]:


print('Accuracy for training set for SVM = {}'.format((cf_train[0][0] + cf_train[1][1])/len(y_train)))
print('Accuracy for test set for SVM = {}'.format((cf_test[0][0] + cf_test[1][1])/len(y_test)))


# # Model 4: DecisionTreeClassifier

# In[64]:


from sklearn.model_selection import GridSearchCV 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
param_dist = {
    "max_features": ['auto', 'sqrt', 'log2'],
    "criterion":["gini","entropy"],
    "max_depth":[1,2,3,4,5,6,7,8,9,10,None]
}  

grid = GridSearchCV(DecisionTreeClassifier(random_state = 10), param_grid = param_dist, cv=60, n_jobs=-2,)
grid.fit(X_train, y_train)
print (grid.best_estimator_)
print (grid.best_score_)
print (grid.best_params_)


# In[65]:



from sklearn.metrics import accuracy_score,confusion_matrix
model = DecisionTreeClassifier(criterion='gini', max_depth=3, max_features='auto',random_state = 10)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


# In[66]:


y_pred_train = classifier.predict(X_train)
cf_test = confusion_matrix(y_pred, y_test)
cf_train = confusion_matrix(y_pred_train, y_train)
cf_train


# In[67]:


print('Accuracy for training set for decisionTree = {}'.format((cf_train[0][0] + cf_train[1][1])/len(y_train)))
print('Accuracy for test set for decisionTree = {}'.format((cf_test[0][0] + cf_test[1][1])/len(y_test)))


# # Model 5: KNN

# In[68]:


from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# In[69]:


knn_scores = []
for k in range(1,7):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score = cross_val_score(knn_classifier,X,y,cv=10)
    knn_scores.append(score.mean())


# In[70]:


plt.plot([k for k in range(1,7)],knn_scores,color = 'red')
for i in range(1,7):
    plt.text(i,knn_scores[i-1],(i,knn_scores[i-1])) #plt.text(x,y,s),x,y denote position,s denote the text
plt.xticks([i for i in range(1,7)])
plt.xlabel('Numbers of neighbours(k)')
plt.ylabel('scores')
plt.title('K Neighbours Classifier scores for differrent K values')


# In[71]:


model2 = KNeighborsClassifier(n_neighbors=2,metric='euclidean')
model2.fit(X_train,y_train)
y_predict2 = model2.predict(X_test)
print( accuracy_score(y_test,y_predict2))
print(classification_report(y_test, y_predict2))


# In[72]:


y_pred_train = classifier.predict(X_train)
cf_test = confusion_matrix(y_predict2, y_test)
cf_train = confusion_matrix(y_pred_train, y_train)
cf_train


# In[73]:


print('Accuracy for training set for KNN = {}'.format((cf_train[0][0] + cf_train[1][1])/len(y_train)))
print('Accuracy for test set for KNN = {}'.format((cf_test[0][0] + cf_test[1][1])/len(y_test)))


# # Model 6: Random Forest

# In[74]:


from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier

model6 = RandomForestClassifier()# get instance of model
model6.fit(X_train, y_train) # Train/Fit model 

y_pred6 = model6.predict(X_test) # get y predictions

print(classification_report(y_test, y_pred6)) # output accuracy


# In[75]:


y_pred6_train = classifier.predict(X_train)
cf_test = confusion_matrix(y_pred6, y_test)
cf_train = confusion_matrix(y_pred6_train, y_train)
cf_train


# In[76]:


print('Accuracy for training set for Random Forest = {}'.format((cf_train[0][0] + cf_train[1][1])/len(y_train)))
print('Accuracy for test set for Random Forest = {}'.format((cf_test[0][0] + cf_test[1][1])/len(y_test)))


# ## Conclusion

# The project involved analysis of the heart disease patient dataset with proper data processing.<br>
# Then, 6 models were train & tested with maximum scores as follows:
# 
# 1.Logistic regression:86%<br>   
# 2.Support Vector Classifier: 86.1%<br>
# 3.Decision Tree Classifier: 77.8%<br>
# 4:K Neighbors Classifier: 83%<br>   
# 5.Random Forest Classifier: 83%<br>
#     
# 

# In[ ]:




