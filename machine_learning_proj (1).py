#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import accuracy_score,  precision_score,  recall_score, f1_score, mean_squared_error, mean_absolute_error

from scipy.stats import uniform, poisson

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import datetime
import warnings
warnings.filterwarnings('ignore')


# readinig train file

# In[2]:


df = pd.read_csv(r'C:\Users\Kamel\Downloads\3rd year subjects\ML\sample_loan_data\train.csv')

df.head()


# In[3]:


df['INPUT_VALUE_ID_FOR_industry_type'].value_counts()


# In[4]:


df.isnull().sum()


# dropping columns

# In[5]:


#dropping
df = df.drop(['id','RATE_owner_1','owner_3_score','owner_1_score','RATE_owner_3','CAP_AMOUNT_owner_3','RATE_ID_FOR_industry_type','RATE_ID_FOR_avg_net_deposits','RATE_ID_FOR_funded_last_30','RATE_ID_FOR_location','funded_last_30','RATE_ID_FOR_judgement_lien_amount','INPUT_VALUE_ID_FOR_judgement_lien_amount','RATE_ID_FOR_judgement_lien_percent','judgement_lien_percent','owner_2_score','RATE_owner_2','CAP_AMOUNT_owner_2','PERCENT_OWN_owner_2'],axis=1)
df = df.drop(['INPUT_VALUE_ID_FOR_judgement_lien_time','PERCENT_OWN_owner_3','RATE_ID_FOR_fsr','RATE_ID_FOR_judgement_lien_time','INPUT_VALUE_ID_FOR_tax_lien_amount','RATE_ID_FOR_tax_lien_amount','RATE_ID_FOR_tax_lien_percent','INPUT_VALUE_ID_FOR_tax_lien_count','RATE_ID_FOR_tax_lien_count',],axis=1)
df.drop(df.columns[[0]],axis=1, inplace=True)


# GETTING INFORMATION ABOUT DATA

# In[6]:


df.info()


# FILLING NULLS USING MEAN FOR NUMERICAL VALUES

# In[7]:


num_val = ['CAP_AMOUNT_owner_1', 'PERCENT_OWN_owner_1', 'years_in_business','fsr', 'INPUT_VALUE_ID_FOR_num_negative_days','INPUT_VALUE_ID_FOR_num_deposits', 'INPUT_VALUE_ID_FOR_monthly_gross', 'INPUT_VALUE_ID_FOR_average_ledger','INPUT_VALUE_ID_FOR_fc_margin', 'INPUT_VALUE_ID_FOR_tax_lien_percent' ,'INPUT_VALUE_ID_FOR_current_position','INPUT_VALUE_ID_FOR_avg_net_deposits', 'INPUT_VALUE_owner_4','CAP_AMOUNT_owner_4','PERCENT_OWN_owner_4','deal_application_thread_id']
col_vals = []

for i in num_val:
    mean_value = df[i].mean()
    df[i].fillna(mean_value, inplace=True)
    col_vals.append(mean_value)


# FILLING NULLS USING MODE FOR STRING VALUES

# In[8]:


str_val = ['RATE_ID_FOR_years_in_business','location','RATE_ID_FOR_num_negative_days','RATE_ID_FOR_num_deposits','RATE_ID_FOR_monthly_gross','RATE_ID_FOR_average_ledger','RATE_ID_FOR_fc_margin','RATE_ID_FOR_current_position','INPUT_VALUE_ID_FOR_industry_type','RATE_owner_4','completion_status']
column_values = []

for col in str_val:
    mode = df[col].mode()[0]
    df[col].fillna(mode, inplace=True)
    column_values.append(mode)
    


# In[9]:


df.isnull().sum()


# ENCODING USING LABELENCODER FOR COLUMNS WITH MORE THAN 2 VALUES

# In[10]:


catg = ['RATE_ID_FOR_current_position','INPUT_VALUE_ID_FOR_industry_type','RATE_owner_4','completion_status','RATE_ID_FOR_years_in_business','location','RATE_ID_FOR_num_deposits','RATE_ID_FOR_num_negative_days']
label_encoding_lst=[]
for i in catg:
    #print(df[i].unique())
    le = preprocessing.LabelEncoder()
    label_encoding_lst.append(le)
    df[i]=le.fit_transform(df[i])


# In[11]:


le.classes_


# ENCODING USING LABELENCODER FOR COLUMNS WITH 2 VALUES

# In[12]:


#using onehot
one_hot_cols = ['RATE_ID_FOR_monthly_gross','RATE_ID_FOR_fc_margin','RATE_ID_FOR_average_ledger']
df = pd.get_dummies(df, columns=one_hot_cols)


# VISUALIZATION 

# In[13]:


# for col in df:
#     sns.histplot(df[col], kde = True)
#     plt.show()


# In[14]:


# correlation
df.corr().style.background_gradient(cmap="Blues")


# In[15]:


# # histo. btw each 2 columns
# df.hist(figsize= [20,15])
# plt.show()
# plt.tight_layout()


# BOXPLOT FOR SHOWING OUTLIERS

# In[16]:


# for col in df.columns:
#     plt.figure(figsize=(70, 45))
#     sns.boxplot(data=df[col], palette='rainbow', orient='h')
#     plt.title(col)
#     plt.show()


# HANDLING OUTLIERS IQR METHOD 

# In[17]:


ndf=['CAP_AMOUNT_owner_1', 'PERCENT_OWN_owner_1', 'years_in_business','fsr', 'INPUT_VALUE_ID_FOR_num_negative_days','INPUT_VALUE_ID_FOR_num_deposits', 'INPUT_VALUE_ID_FOR_monthly_gross', 'INPUT_VALUE_ID_FOR_average_ledger','INPUT_VALUE_ID_FOR_fc_margin', 'INPUT_VALUE_ID_FOR_tax_lien_percent' ,'INPUT_VALUE_ID_FOR_current_position','INPUT_VALUE_ID_FOR_avg_net_deposits', 'INPUT_VALUE_owner_4','CAP_AMOUNT_owner_4','PERCENT_OWN_owner_4','deal_application_thread_id']

for col in ndf:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])


# DATA AFTER HANDLING OUTLIERS 

# In[18]:


# for col in df.columns:
#     plt.figure(figsize=(70, 45))
#     sns.boxplot(data=df[col], palette='rainbow', orient='h')
#     plt.title(col)
#     plt.show()


# SPLIT DATA INTO TRAIN AND VALIDATION SET

# In[19]:


x = df.drop(['completion_status'], axis=1)
y = df['completion_status']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=46)


# SCALING DATA

# In[20]:


scaler = StandardScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns = x.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns = x.columns)


# RIDGE MODEL

# In[21]:


ridge_model = Ridge(alpha=0.99)
ridge_model.fit(x_train, y_train)
y_pred = ridge_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rms = np.sqrt(mse)
print('The root mean square error is:', rms)


# LASSO MODEL

# In[22]:


lasso_model = Lasso(alpha=0.01)
lasso_model.fit(x_train, y_train)
y_pred = lasso_model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)


# ELASTICNET

# In[23]:


elasticnet = ElasticNet(alpha=0.01)
elasticnet.fit(x_train, y_train)
y_pred = elasticnet.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error: ", mse)


# MODEL INITIALIZATION 

# In[24]:


LR = LogisticRegression(solver = "liblinear")
DTC = DecisionTreeClassifier()
knn = KNeighborsClassifier(n_neighbors=1)
nvb = GaussianNB()
ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),learning_rate= 1, n_estimators= 200)


# CLASSIFICATION MODELS

# MODEL 1

# In[25]:


RFC = RandomForestClassifier()
RFC.fit(x_train,y_train)
output3 = RFC.predict(x_test)

print("RandomForestClassifier:")
print('accuracy score: ',  accuracy_score(y_test,output3))
print("Precision Score : ",precision_score(y_test, output3, pos_label='positive',average='macro'))
print("Recall Score : ",recall_score(y_test, output3,pos_label='positive',average='macro'))
print('f1 score: ' ,f1_score(y_test,output3,pos_label='positive',average='macro'))
# cf3 = confusion_matrix(y_test, output3)
cm1 = confusion_matrix(y_test,output3, labels=RFC.classes_)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1,display_labels=RFC.classes_)
disp1.plot()


#              

# FEATURE SELECTION FOR RFC

# In[26]:


importances = RFC.feature_importances_
print(importances)


#             

# MODEL 2

# FEATURE SELECTION FOR DTC

# In[27]:


# from sklearn.feature_selection import RFECV
# rfecv = RFECV(estimator=DTC, step=1, cv=10)

# x_dfc = df.drop(['completion_status'], axis=1)
# y_dfc = df['completion_status']
# rfecv.fit(x, y)

# # Print the selected features
# print("Optimal number of features: %d" % rfecv.n_features_)
# print("Selected features:", [i for i in range(len(rfecv.support_)) if rfecv.support_[i]])


# In[28]:


# x_trdfc, x_tedfc, y_trdfc, y_tedfc = train_test_split(x_dfc, y_dfc, test_size=0.2, random_state=46)


# In[60]:


DTC.fit(x_train,y_train)
output2 = DTC.predict(x_test)
print("DecisionTreeClassifier:")
print('accuracy score: ',  accuracy_score(y_test,output2))
print("Precision Score : ",precision_score(y_test, output2, pos_label='positive',average='macro'))
print("Recall Score : ",recall_score(y_test, output2,pos_label='positive',average='micro'))
print('f1 score: ' ,f1_score(y_test,output2,pos_label='positive',average='weighted'))
cm2 = confusion_matrix(y_test,output2, labels=DTC.classes_)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=DTC.classes_)
disp1.plot()


# MODEL 3

# In[63]:


knn.fit(x_train,y_train)
output1 = knn.predict(x_test)
print("knn:")
print('accuracy score: ',  accuracy_score(y_test,output1))
print("Precision Score : ",precision_score(y_test, output1, pos_label='positive',average='macro'))
print("Recall Score : ",recall_score(y_test, output1,pos_label='positive',average='micro'))
print('f1 score: ' , f1_score(y_test,output1,pos_label='positive',average='weighted'))
cm3 = confusion_matrix(y_test,output1, labels=knn.classes_)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm3,display_labels=knn.classes_)
disp1.plot()


# MODEL 4

# In[65]:


ada.fit(x_train,y_train)
output5 = ada.predict(x_test)
print("AdaBoostClassifier:")
print('Accuracy score:', (accuracy_score(y_test, output5)))
print("Precision Score : ",precision_score(y_test, output5, pos_label='positive',average='macro'))
print("Recall Score : ",recall_score(y_test, output5,pos_label='positive',average='micro'))
print('f1 score:' , f1_score(y_test,output5,pos_label='positive',average='weighted'))
cm6 = confusion_matrix(y_test,output5, labels=ada.classes_)
disp5 = ConfusionMatrixDisplay(confusion_matrix=cm6,display_labels=ada.classes_)
disp5.plot()


# MODEL 5

# In[69]:


LR.fit(x_train,y_train)
output = LR.predict(x_test)
print("Logistic Regression:")
print('accuracy score: ' , accuracy_score(y_test,output))
print("Precision Score : ",precision_score(y_test, output, pos_label='positive',average='macro'))
print("Recall Score : ",recall_score(y_test, output,pos_label='positive',average='micro'))
print('f1 score: ' ,f1_score(y_test,output,pos_label='positive',average='weighted'))
cm1 = confusion_matrix(y_test,output, labels=LR.classes_)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1,display_labels=LR.classes_)
disp1.plot()


# In[33]:


# #model 6
# from sklearn.svm import SVC
# svm_model = SVC(kernel='linear')
# svm_model.fit(x_train, y_train)
# y_pred = svm_model.predict(x_test)
# print('accuracy score: ' , accuracy_score(y_test,y_pred))
# cm4 = confusion_matrix(y_test,y_pred, labels=svm_model.classes_)
# disp1 = ConfusionMatrixDisplay(confusion_matrix=cm4,display_labels=svm_model.classes_)
# disp1.plot()


# BEST HYPERPARAMETER FOR ADABOOST USING DTC ESTIM.

# In[34]:


# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'learning_rate': [0.1, 0.5, 1],
#     'base_estimator__max_depth': [1, 2, 3]
# }
# # Create an instance of the AdaBoostClassifier
# # clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

# # Create a GridSearchCV object with the specified parameters
# grid_search = GridSearchCV(ada, param_grid=param_grid, cv=5)

# # Fit the grid search object to the data
# grid_search.fit(x_train, y_train)

# # Print the best parameters and score
# print("Best parameters:", grid_search.best_params_)
# print("Best score:", grid_search.best_score_)


# TUNING IN RANDOM FOREST

# In[35]:


# n_estimators = [50, 100, 150]
# max_depth = [3, 5, 7]
# max_features = ["sqrt", "log2"]

# param_grid = {
#     "n_estimators": n_estimators,
#     "max_depth": max_depth,
#     "max_features": max_features
# }
# # Create a GridSearchCV object with the specified parameters
# grid_search = GridSearchCV(RFC, param_grid=param_grid, cv=5)

# # Fit the grid search object to the data
# grid_search.fit(x_train, y_train)

# # Print the best parameters and score
# print("Best parameters:", grid_search.best_params_)
# print("Best score:", grid_search.best_score_)


# HYPERPARAMETER TUNING FOR LOGISTICREG

# In[36]:



# from sklearn.metrics import make_scorer

# # Define the hyperparameters to search over
# param_grid = {
#     'penalty': ['l1', 'l2'],
#     'C': np.logspace(-4, 4, 20),
#     'solver': ['liblinear', 'saga']
# }

# # Create a logistic regression model object
# model = LogisticRegression(max_iter=100)

# # Perform the grid search
# grid_search = GridSearchCV(model, param_grid, cv=5, scoring=make_scorer(accuracy_score))
# grid_search.fit(x_train, y_train)

# # Print the best hyperparameters and corresponding score
# print("Best parameters:", grid_search.best_params_)
# print("Best score:", grid_search.best_score_)

# # Evaluate the model on the test set
# y_pred = grid_search.predict(x_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Test set accuracy:", accuracy)


#              

#         

#            

#                 

#           

#            

# ------------------------------------------------------

# TEST FILE

# In[37]:


dftest = pd.read_csv(r'C:\Users\Kamel\Downloads\test.csv')


# In[38]:



#dropping
#to_be_dropped=['id','owner_3_score','RATE_owner_1','RATE_owner_3','CAP_AMOUNT_owner_3','RATE_ID_FOR_industry_type','RATE_ID_FOR_avg_net_deposits','RATE_ID_FOR_funded_last_30','RATE_ID_FOR_location','funded_last_30','RATE_ID_FOR_judgement_lien_amount','INPUT_VALUE_ID_FOR_judgement_lien_amount','RATE_ID_FOR_judgement_lien_percent','judgement_lien_percent','owner_2_score','RATE_owner_2','CAP_AMOUNT_owner_2','PERCENT_OWN_owner_2','INPUT_VALUE_ID_FOR_judgement_lien_time','PERCENT_OWN_owner_3','RATE_ID_FOR_fsr','RATE_ID_FOR_judgement_lien_time','INPUT_VALUE_ID_FOR_tax_lien_amount','RATE_ID_FOR_tax_lien_amount','RATE_ID_FOR_tax_lien_percent','INPUT_VALUE_ID_FOR_tax_lien_count','RATE_ID_FOR_tax_lien_count']
dftest = dftest.drop(['Unnamed: 0','id','owner_3_score','RATE_owner_1','RATE_owner_3','owner_1_score','CAP_AMOUNT_owner_3','RATE_ID_FOR_industry_type','RATE_ID_FOR_avg_net_deposits','RATE_ID_FOR_funded_last_30','RATE_ID_FOR_location','funded_last_30','RATE_ID_FOR_judgement_lien_amount','INPUT_VALUE_ID_FOR_judgement_lien_amount','RATE_ID_FOR_judgement_lien_percent','judgement_lien_percent','owner_2_score','RATE_owner_2','CAP_AMOUNT_owner_2','PERCENT_OWN_owner_2'],axis=1)
dftest = dftest.drop(['INPUT_VALUE_ID_FOR_judgement_lien_time','PERCENT_OWN_owner_3','RATE_ID_FOR_fsr','RATE_ID_FOR_judgement_lien_time','INPUT_VALUE_ID_FOR_tax_lien_amount','RATE_ID_FOR_tax_lien_amount','RATE_ID_FOR_tax_lien_percent','INPUT_VALUE_ID_FOR_tax_lien_count','RATE_ID_FOR_tax_lien_count',],axis=1)
# dftest.drop(df.columns[[0]],axis=1, inplace=True)


# In[39]:


# for col in dftest.columns:
#     if dftest[col].isnull().sum() / len(dftest) >= 0.8:
#         dftest = dftest.drop(col, axis=1)
# print("first", dftest.shape[1])


# In[40]:


print(num_val)


# In[41]:


#mean
i=0
for col in num_val:
    if col  in dftest.columns:
        dftest[col].fillna(col_vals[i], inplace=True)
        i+=1
#mode
j=0
for col in str_val :
    if col  in dftest.columns:
        dftest[col].fillna(column_values[j], inplace=True)
        j+=1

print(dftest.info())


# In[42]:


dftest['location'].unique()


# In[43]:


df['INPUT_VALUE_ID_FOR_industry_type'].value_counts()


# In[44]:


dftest['INPUT_VALUE_ID_FOR_industry_type'].value_counts()


# In[45]:


#encoding
#using labelencoding
catg = ['RATE_ID_FOR_current_position','INPUT_VALUE_ID_FOR_industry_type','RATE_owner_4','completion_status','RATE_ID_FOR_years_in_business','location','RATE_ID_FOR_num_deposits','RATE_ID_FOR_num_negative_days']
for i,col in enumerate(catg):
    #print(df[i].unique())
    print(label_encoding_lst[i].classes_)
    dftest[col]=label_encoding_lst[i].transform(dftest[col])
   


# for col in dftest.columns:
#     if dftest[col].dtype == 'object':  # check if column is string
#         if len(dftest[col].unique()) > 2:  # check if there are more than 2 unique values
#             encoder = LabelEncoder()
#             dftest[col] = encoder.fit_transform(dftest[col])
#         else:
#
#             encoder = OneHotEncoder()
#             temp = pd.DataFrame(encoder.fit_transform(dftest[[col]]).toarray())
#             temp.columns = [col+'_'+str(i) for i in range(len(temp.columns))]
#             dftest = pd.concat([dftest, temp], axis=1)
#             dftest.drop(col, axis=1, inplace=True)

#using onehot
one_hot_cols = ['RATE_ID_FOR_monthly_gross','RATE_ID_FOR_fc_margin','RATE_ID_FOR_average_ledger']
dftest = pd.get_dummies(dftest, columns=one_hot_cols)
# boxplot for showing outliers
print("testtttt")
dftest.info()


# In[46]:


# dftest = dftest.drop(['RATE_ID_FOR_tax_lien_percent','RATE_ID_FOR_tax_lien_amount'],axis = 1)


# In[47]:


# dftest['RATE_ID_FOR_tax_lien_count'].unique()


# In[48]:


# # boxplot for showing outliers
# for col in dftest.columns:
#     plt.figure(figsize=(70, 45))
#     sns.boxplot(data=dftest[col], palette='rainbow', orient='h')
#     plt.title(col)
#     plt.show()


# In[49]:


#features and target
x = dftest.drop(['completion_status'], axis=1)
y = dftest['completion_status']


# In[50]:



for col in ndf:
    if col  in dftest.columns:

        Q1 = dftest[col].quantile(0.25)
        Q3 = dftest[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        dftest[col] = np.where(dftest[col] < lower_bound, lower_bound, dftest[col])
        dftest[col] = np.where(dftest[col] > upper_bound, upper_bound, dftest[col])
#scaling
# transform DataFrame
x.info()
x= pd.DataFrame(scaler.transform(x), columns=x.columns)


# In[51]:


print(x.columns)


# In[52]:


# print(len(x.columns))
print(len(df.columns))


# In[53]:


dftest.info()


# In[54]:


# dftest = dftest.drop(['RATE_ID_FOR_tax_lien_percent'],axis =1)


# In[55]:


# #features and target
# x = dftest.drop(['completion_status'], axis=1)
# y = dftest['completion_status']


# In[56]:


#scaling
# transform DataFrame
# x = pd.DataFrame(scaler.fit_transform(x_train), columns = x.columns)

dftest= pd.DataFrame(scaler.fit_transform(dftest), columns=dftest.columns)


# In[57]:


#Model intialization
LR = LogisticRegression(solver = "liblinear")
DTC = DecisionTreeClassifier()
knn = KNeighborsClassifier(n_neighbors=1)
nvb = GaussianNB()
ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),learning_rate= 1, n_estimators= 200)


# In[73]:


#model 1

output3 = RFC.predict(x)
print("RandomForestClassifier:")
print('accuracy score: ',  accuracy_score(y,output3))
print("Precision Score : ",precision_score(y, output3, pos_label='positive',average='macro'))
print("Recall Score : ",recall_score(y, output3,pos_label='positive',average='macro'))
print('f1 score: ' ,f1_score(y,output3,pos_label='positive',average='weighted'))
# cf3 = confusion_matrix(y_test, output3)
cm1 = confusion_matrix(y,output3, labels=RFC.classes_)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1,display_labels=RFC.classes_)
disp1.plot()


# In[72]:


#model 2
output2 = DTC.predict(x)
print("DecisionTreeClassifier:")
print('accuracy score: ',  accuracy_score(y,output2))
print("Precision Score : ",precision_score(y, output2, pos_label='positive',average='macro'))
print("Recall Score : ",recall_score(y, output2,pos_label='positive',average='micro'))
print('f1 score: ' ,f1_score(y,output2,pos_label='positive',average='weighted'))
cm2 = confusion_matrix(y,output2, labels=DTC.classes_)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm2,display_labels=DTC.classes_)
disp1.plot()


# In[64]:


#model 3
output1 = knn.predict(x)
print("knn:")
print('accuracy score: ',  accuracy_score(y,output1))
print("Precision Score : ",precision_score(y, output1, pos_label='positive',average='macro'))
print("Recall Score : ",recall_score(y, output1,pos_label='positive',average='micro'))
print('f1 score: ' , f1_score(y,output1,pos_label='positive',average='weighted'))
cm3 = confusion_matrix(y,output1, labels=knn.classes_)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm3,display_labels=knn.classes_)
disp1.plot()


# In[67]:


#model 4
output5 = ada.predict(x)
print("AdaBoostClassifier:")
print('Accuracy score:', (accuracy_score(y, output5)))
print("Precision Score : ",precision_score(y, output5, pos_label='positive',average='macro'))
print("Recall Score : ",recall_score(y, output5,pos_label='positive',average='macro'))
print('f1 score:' , f1_score(y,output5,pos_label='positive',average='macro'))
cm6 = confusion_matrix(y,output5, labels=ada.classes_)
disp5 = ConfusionMatrixDisplay(confusion_matrix=cm6,display_labels=ada.classes_)
disp5.plot()


# In[70]:



#model 5
output = LR.predict(x)
print("Logistic Regression:")
print('accuracy score: ' ,accuracy_score(y,output))
print("Precision Score : ",precision_score(y, output, pos_label='positive',average='macro'))
print("Recall Score : ",recall_score(y, output,pos_label='positive',average='micro'))
print('f1 score: ' ,f1_score(y,output,pos_label='positive',average='weighted'))
cm1 = confusion_matrix(y,output, labels=LR.classes_)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1,display_labels=LR.classes_)
disp1.plot()


# In[ ]:


# #model 6
# from sklearn.svm import SVC
# svm_model = SVC(kernel='linear')
# svm_model.fit(x_train, y_train)
# y_pred = svm_model.predict(x_test)
# print('accuracy score: ' , accuracy_score(y_test,y_pred))
# cm4 = confusion_matrix(y_test,y_pred, labels=svm_model.classes_)
# disp1 = ConfusionMatrixDisplay(confusion_matrix=cm4,display_labels=svm_model.classes_)
# disp1.plot()


# In[ ]:


# #best hyp.para. for adaboost using dt estim.
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'learning_rate': [0.1, 0.5, 1],
#     'base_estimator__max_depth': [1, 2, 3]
# }
# # Create an instance of the AdaBoostClassifier
# # clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

# # Create a GridSearchCV object with the specified parameters
# grid_search = GridSearchCV(ada, param_grid=param_grid, cv=5)

# # Fit the grid search object to the data
# grid_search.fit(x, y)

# # Print the best parameters and score
# print("Best parameters:", grid_search.best_params_)
# print("Best score:", grid_search.best_score_)


# In[ ]:


# #tuning on random forest
# n_estimators = [50, 100, 150]
# max_depth = [3, 5, 7]
# max_features = ["sqrt", "log2"]

# param_grid = {
#     "n_estimators": n_estimators,
#     "max_depth": max_depth,
#     "max_features": max_features
# }

# # Create an instance of the RandomForestClassifier
# # clf = RandomForestClassifier()

# # Create a GridSearchCV object with the specified parameters
# grid_search = GridSearchCV(RFC, param_grid=param_grid, cv=5)

# # Fit the grid search object to the data
# grid_search.fit(x,y)

# # Print the best parameters and score
# print("Best parameters:", grid_search.best_params_)
# print("Best score:", grid_search.best_score_)

