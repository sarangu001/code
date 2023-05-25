#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('features_49.csv')


# In[4]:


data27 = pd.read_csv('features_27.csv')


# In[5]:


data16 = pd.read_csv('features_16.csv')


# In[6]:


data11 = pd.read_csv('features_11.csv')


# In[7]:


#Taking data of 49 features
#RF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


data.shape


# In[9]:


data.describe()


# In[10]:


pd.set_option('display.max_rows', 500)
data.isna().sum()


# In[11]:


pd.reset_option('display.max_rows')


# In[12]:


original_dataset = data.copy()


# In[13]:


class_map = {'legitimate':0, 'phishing':1}
original_dataset['status'] = original_dataset['status'].map(class_map)


# In[14]:


corr_matrix = original_dataset.corr()


# In[15]:


plt.figure(figsize=(60,60))
color = plt.get_cmap('viridis').copy()   # default color
color.set_bad('lightblue') 
sns.heatmap(corr_matrix, annot=True, linewidth=0.4, cmap=color)
plt.savefig('heatmap')
plt.show()


# In[16]:


corr_matrix.shape


# In[17]:


corr_matrix['status']


# In[18]:


status_corr = corr_matrix['status']
status_corr.shape


# In[19]:


def feature_selector_correlation(cmatrix, threshold):
    
    selected_features = []
    feature_score = []
    i=0
    for score in cmatrix:
        if abs(score)>threshold:
            selected_features.append(cmatrix.index[i])
            feature_score.append( ['{:3f}'.format(score)])
        i+=1
    result = list(zip(selected_features,feature_score)) 
    return result


# In[20]:


features_selected = feature_selector_correlation(status_corr, 0.2)
features_selected


# In[21]:


selected_features = [i for (i,j) in features_selected if i != 'status']
selected_features


# In[22]:


X_selected = original_dataset[selected_features]
X_selected


# In[23]:


X_selected.shape


# In[24]:


X_selected.shape


# In[25]:


y = original_dataset['status']
y


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X_selected, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle = True)


# In[28]:


from sklearn.ensemble import RandomForestClassifier


# In[29]:


model_random_forest = RandomForestClassifier(n_estimators=350,
                                             random_state=42,
                                             )


# In[30]:


model_random_forest.fit(X_train,y_train)


# In[31]:


from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


# In[34]:


def custom_accuracy_set (model, X_train, X_test, y_train, y_test, train=True):
    
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    
    
    if train:
        x = X_train
        y = y_train
    elif not train:
        x = X_test
        y = y_test
        
    y_predicted = model.predict(x)
    
    accuracy = accuracy_score(y, y_predicted)
    print('model accuracy: {0:4f}'.format(accuracy))
    oconfusion_matrix = confusion_matrix(y, y_predicted)
    print('Confusion matrix: \n {}'.format(oconfusion_matrix))
    oroc_auc_score = lb.transform(y), lb.transform(y_predicted)


# In[35]:


custom_accuracy_set(model_random_forest, X_train, X_test, y_train, y_test, train=True)


# In[36]:


custom_accuracy_set(model_random_forest, X_train, X_test, y_train, y_test, train=False)


# In[37]:


#MLP

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix , accuracy_score , precision_score , recall_score , f1_score
from sklearn.feature_selection import SelectPercentile , chi2 , f_classif
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


# In[38]:


X = data.copy()


# In[39]:


Y = data["status"]
# Y = Y == "legitimate"
X.drop(["status" , "url"] , axis=1 , inplace=True)
# print(X.columns)
Cols = X.columns;
# print(X.shape , Y.shape)
Y = Y == "legitimate"


# In[40]:


Scaler = StandardScaler(copy=True , with_mean=True , with_std=True)
X = Scaler.fit_transform(X)


# In[41]:


SP = SelectPercentile(score_func=f_classif , percentile=60)


# In[42]:


X = SP.fit_transform(X , Y)


# In[43]:


print("Number Of Features : " , Cols[SP.get_support()])


# In[44]:


x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size=0.20 , random_state=10 , shuffle=True)


# In[45]:


Model = MLPClassifier(solver='adam' , alpha=0.01 , hidden_layer_sizes=(100 , 100 , 100 , 100) , max_iter=100 , random_state=44)
Ans1 = Model.fit(x_train , y_train)
print("Score Model For Training Data : " , Model.score(x_train , y_train))


# In[46]:


YPred = Ans1.predict(x_test)
conf_matrix = confusion_matrix(y_test ,YPred)
print("Confusion Matrix of the Test Set")
print("-----------")
print(conf_matrix)
print("Precision of the MLP :\t"+str(precision_score(y_test,YPred)))
print("Recall of the MLP    :\t"+str(recall_score(y_test,YPred)))
print("F1 Score of the MLP :\t"+str(f1_score(y_test,YPred)))
print("Accuracy Score of the MLP :\t"+str(accuracy_score(y_test,YPred)))


# In[47]:


#XGBOOST
import numpy as np 
import pandas as pd 
import os


# In[48]:


pip install xgboost


# In[49]:


import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBRegressor
# import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# In[50]:


data.columns


# In[51]:


from sklearn.model_selection import train_test_split


# In[52]:


y = data.status
X = data.drop(['status'], axis=1)


# In[53]:


X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)


# In[54]:


categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]


numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]


# In[55]:


my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()


# In[56]:


numerical_transformer = SimpleImputer(strategy='constant')


# In[57]:


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# In[58]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# In[59]:


print(data.describe)


# In[60]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)


# In[61]:


y_valid = le.fit_transform(y_valid)


# In[62]:


print(y_train)


# In[63]:


import xgboost as xgb
from sklearn.metrics import accuracy_score


# In[64]:


model = xgb.XGBClassifier()


# In[65]:


my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])


# In[66]:


my_pipeline.fit(X_train, y_train)


# In[67]:


preds = my_pipeline.predict(X_valid)
print(accuracy_score(preds,y_valid))


# In[68]:


#Gausian Naive Bayes method

import numpy as np
import pandas as pd 

import os
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.naive_bayes import GaussianNB #model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import time


# In[69]:


start = time.time()


# In[70]:


removeCols = ['url', 'status']
featuresCol = [i for i in data.columns if i not in removeCols]
StatusCol = 'status'
data[StatusCol] = data[StatusCol]
features, targets = data[featuresCol], data[StatusCol]


# In[71]:


class Scaler():
    
    def __init__(self, scaler, feature_range=None):
        self.columns = None
        self.index = None
        self.feature_range = feature_range
        self.scaler = scaler()
        if (isinstance(self.scaler, sklearn.preprocessing._data.MinMaxScaler)
            and isinstance(self.feature_range, tuple)):
            self.scaler = scaler(feature_range = self.feature_range)
        
    def fit(self, X):
        self.scaler = self.scaler.fit(X)
        return self
                
    def transform(self, X):
        scaled_X = self.scaler.transform(X)
        
        if isinstance(X, pd.core.frame.DataFrame):
            self.columns = X.columns
            self.index = X.index
            scaled_X = pd.DataFrame(scaled_X, index=self.index, columns=self.columns)
        return scaled_X
    
    def inverse_transform(self, X):
        inversed_X = self.scaler.inverse_transform(X)
        
        if isinstance(X, pd.core.frame.DataFrame):
            self.columns = X.columns
            self.index = X.index
            inversed_X = pd.DataFrame(inversed_X, index=self.index, columns=self.columns)
        return inversed_X



train_size = 0.85 
X_train, X_test, y_train, y_test = train_test_split(
    features, targets, shuffle=True, train_size=train_size,
    random_state=42)


# In[72]:


scaler = Scaler(StandardScaler) 
scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)


# In[73]:


naive_bayes = GaussianNB() #call model
naive_bayes = naive_bayes.fit(X_train, y_train)
pred = naive_bayes.predict(X_train)
print(f'\nTraining Accuracy: {round(accuracy_score(y_train, pred)*100, 4)}%')


# In[74]:


X_test = scaler.transform(X_test)
pred = naive_bayes.predict(X_test)


# In[75]:


cm = confusion_matrix(y_test, pred)
display_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=naive_bayes.classes_)
display_cm.plot()
plt.show()


# In[76]:


print(f'Accuracy: {round(accuracy_score(y_test, pred)*100, 4)}%')
end = time.time()
finalTime = end - start
print("Total time (in seconds): " + str(finalTime))


# In[77]:


#Taking Data27 0f 27features
#RF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[78]:


data27.describe()


# In[79]:


pd.set_option('display.max_rows', 500)
data27.isna().sum()


# In[80]:


pd.reset_option('display.max_rows')


# In[81]:


original_dataset = data27.copy()


# In[82]:


class_map = {'legitimate':0, 'phishing':1}
original_dataset['status'] = original_dataset['status'].map(class_map)


# In[83]:


corr_matrix = original_dataset.corr()


# In[84]:


plt.figure(figsize=(60,60))
color = plt.get_cmap('viridis').copy()   # default color
color.set_bad('lightblue') 
sns.heatmap(corr_matrix, annot=True, linewidth=0.4, cmap=color)
plt.savefig('heatmap')
plt.show()


# In[85]:


corr_matrix.shape


# In[86]:


corr_matrix['status']


# In[87]:


status_corr = corr_matrix['status']
status_corr.shape


# In[88]:


def feature_selector_correlation(cmatrix, threshold):
    
    selected_features = []
    feature_score = []
    i=0
    for score in cmatrix:
        if abs(score)>threshold:
            selected_features.append(cmatrix.index[i])
            feature_score.append( ['{:3f}'.format(score)])
        i+=1
    result = list(zip(selected_features,feature_score)) 
    return result


# In[89]:


features_selected = feature_selector_correlation(status_corr, 0.2)
features_selected


# In[90]:


selected_features = [i for (i,j) in features_selected if i != 'status']
selected_features


# In[91]:


X_selected = original_dataset[selected_features]
X_selected


# In[92]:


X_selected.shape


# In[93]:


y = original_dataset['status']
y


# In[94]:


from sklearn.model_selection import train_test_split


# In[95]:


X_train, X_test, y_train, y_test = train_test_split(X_selected, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle = True)


# In[96]:


from sklearn.ensemble import RandomForestClassifier


# In[97]:


model_random_forest = RandomForestClassifier(n_estimators=350,
                                             random_state=42,
                                             )


# In[98]:


model_random_forest.fit(X_train,y_train)


# In[99]:


from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


# In[100]:


def custom_accuracy_set (model, X_train, X_test, y_train, y_test, train=True):
    
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    
    
    if train:
        x = X_train
        y = y_train
    elif not train:
        x = X_test
        y = y_test
        
    y_predicted = model.predict(x)
    
    accuracy = accuracy_score(y, y_predicted)
    print('model accuracy: {0:4f}'.format(accuracy))
    oconfusion_matrix = confusion_matrix(y, y_predicted)
    print('Confusion matrix: \n {}'.format(oconfusion_matrix))
    oroc_auc_score = lb.transform(y), lb.transform(y_predicted)		


# In[101]:


custom_accuracy_set(model_random_forest, X_train, X_test, y_train, y_test, train=True)


# In[102]:


custom_accuracy_set(model_random_forest, X_train, X_test, y_train, y_test, train=False)


# In[103]:


#MLP

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix , accuracy_score , precision_score , recall_score , f1_score
from sklearn.feature_selection import SelectPercentile , chi2 , f_classif
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


# In[104]:


X = data27.copy()
# data.head()
Y = data["status"]
# Y = Y == "legitimate"
X.drop(["status" , "url"] , axis=1 , inplace=True)
# print(X.columns)
Cols = X.columns;
# print(X.shape , Y.shape)
Y = Y == "legitimate"


# In[105]:


Scaler = StandardScaler(copy=True , with_mean=True , with_std=True)
X = Scaler.fit_transform(X)


# In[106]:


SP = SelectPercentile(score_func=f_classif , percentile=60)


# In[107]:


X = SP.fit_transform(X , Y)


# In[108]:


print("Number Of Features : " , Cols[SP.get_support()])


# In[109]:


x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size=0.20 , random_state=10 , shuffle=True)


# In[110]:


Model = MLPClassifier(solver='adam' , alpha=0.01 , hidden_layer_sizes=(100 , 100 , 100 , 100) , max_iter=100 , random_state=44)
Ans1 = Model.fit(x_train , y_train)
print("Score Model For Training Data : " , Model.score(x_train , y_train))


# In[111]:


YPred = Ans1.predict(x_test)
conf_matrix = confusion_matrix(y_test ,YPred)
print("Confusion Matrix of the Test Set")
print("-----------")
print(conf_matrix)
print("Precision of the MLP :\t"+str(precision_score(y_test,YPred)))
print("Recall of the MLP    :\t"+str(recall_score(y_test,YPred)))
print("F1 Score of the MLP :\t"+str(f1_score(y_test,YPred)))
print("Accuracy Score of the MLP :\t"+str(accuracy_score(y_test,YPred)))


# In[112]:


#XGBoost
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# In[113]:


pip install xgboost


# In[114]:


import pandas as pd


# In[115]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBRegressor
# import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# In[116]:


data27.columns


# In[117]:


from sklearn.model_selection import train_test_split


# In[118]:


y = data27.status
X = data27.drop(['status'], axis=1)


# In[119]:


X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)


# In[120]:


categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]


# In[121]:


numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]


# In[122]:


my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()


# In[123]:


numerical_transformer = SimpleImputer(strategy='constant')


# In[124]:


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# In[125]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# In[126]:


print(data27.describe)


# In[127]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)


# In[128]:


y_valid = le.fit_transform(y_valid)


# In[129]:


print(y_train)


# In[130]:


import xgboost as xgb
from sklearn.metrics import accuracy_score


# In[131]:


model = xgb.XGBClassifier()


# In[132]:


my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])


# In[133]:


my_pipeline.fit(X_train, y_train)


# In[134]:


preds = my_pipeline.predict(X_valid)
print(accuracy_score(preds,y_valid))


# In[135]:


#Gaussian
import numpy as np
import pandas as pd 

import os
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.naive_bayes import GaussianNB #model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import time




# In[136]:


start = time.time()


# In[138]:


removeCols = ['url', 'status']
featuresCol = [i for i in data27.columns if i not in removeCols]
StatusCol = 'status'
data27[StatusCol] = data27[StatusCol]
features, targets = data27[featuresCol], data27[StatusCol]


# In[139]:


class Scaler():
    
    def __init__(self, scaler, feature_range=None):
        self.columns = None
        self.index = None
        self.feature_range = feature_range
        self.scaler = scaler()
        if (isinstance(self.scaler, sklearn.preprocessing._data.MinMaxScaler)
            and isinstance(self.feature_range, tuple)):
            self.scaler = scaler(feature_range = self.feature_range)
        
    def fit(self, X):
        self.scaler = self.scaler.fit(X)
        return self
                
    def transform(self, X):
        scaled_X = self.scaler.transform(X)
        
        if isinstance(X, pd.core.frame.DataFrame):
            self.columns = X.columns
            self.index = X.index
            scaled_X = pd.DataFrame(scaled_X, index=self.index, columns=self.columns)
        return scaled_X
    
    def inverse_transform(self, X):
        inversed_X = self.scaler.inverse_transform(X)
        
        if isinstance(X, pd.core.frame.DataFrame):
            self.columns = X.columns
            self.index = X.index
            inversed_X = pd.DataFrame(inversed_X, index=self.index, columns=self.columns)
        return inversed_X



train_size = 0.85 
X_train, X_test, y_train, y_test = train_test_split(
    features, targets, shuffle=True, train_size=train_size,
    random_state=42)

scaler = Scaler(StandardScaler) 
scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)


# In[140]:


naive_bayes = GaussianNB() #call model
naive_bayes = naive_bayes.fit(X_train, y_train)
pred = naive_bayes.predict(X_train)
print(f'\nTraining Accuracy: {round(accuracy_score(y_train, pred)*100, 4)}%')


# In[141]:


X_test = scaler.transform(X_test)
pred = naive_bayes.predict(X_test)


# In[142]:


cm = confusion_matrix(y_test, pred)
display_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=naive_bayes.classes_)
display_cm.plot()
plt.show()


# In[143]:


print(f'Accuracy: {round(accuracy_score(y_test, pred)*100, 4)}%')
end = time.time()
finalTime = end - start
print("Total time (in seconds): " + str(finalTime))


# In[144]:


#Taking data11 of 11 features
#RF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[145]:


data11.shape


# In[146]:


data11.describe()


# In[148]:


pd.set_option('display.max_rows', 500)
data11.isna().sum()


# In[149]:


pd.reset_option('display.max_rows')


# In[150]:


original_dataset = data11.copy()


# In[151]:


class_map = {'legitimate':0, 'phishing':1}
original_dataset['status'] = original_dataset['status'].map(class_map)


# In[152]:


corr_matrix = original_dataset.corr()


# In[153]:


plt.figure(figsize=(60,60))
color = plt.get_cmap('viridis').copy()   # default color
color.set_bad('lightblue') 
sns.heatmap(corr_matrix, annot=True, linewidth=0.4, cmap=color)
plt.savefig('heatmap')
plt.show()


# In[154]:


corr_matrix.shape


# In[155]:


corr_matrix['status']


# In[156]:


status_corr = corr_matrix['status']
status_corr.shape


# In[157]:


def feature_selector_correlation(cmatrix, threshold):
    
    selected_features = []
    feature_score = []
    i=0
    for score in cmatrix:
        if abs(score)>threshold:
            selected_features.append(cmatrix.index[i])
            feature_score.append( ['{:3f}'.format(score)])
        i+=1
    result = list(zip(selected_features,feature_score)) 
    return result


# In[158]:


features_selected = feature_selector_correlation(status_corr, 0.2)
features_selected


# In[159]:


selected_features = [i for (i,j) in features_selected if i != 'status']
selected_features


# In[160]:


X_selected = original_dataset[selected_features]
X_selected


# In[161]:


X_selected.shape


# In[162]:


y = original_dataset['status']
y


# In[163]:


from sklearn.model_selection import train_test_split


# In[164]:


X_train, X_test, y_train, y_test = train_test_split(X_selected, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle = True)


# In[165]:


from sklearn.ensemble import RandomForestClassifier


# In[166]:


model_random_forest = RandomForestClassifier(n_estimators=350,
                                             random_state=42,
                                             )


# In[167]:


model_random_forest.fit(X_train,y_train)


# In[168]:


from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


# In[169]:


def custom_accuracy_set (model, X_train, X_test, y_train, y_test, train=True):
    
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    
    
    if train:
        x = X_train
        y = y_train
    elif not train:
        x = X_test
        y = y_test
        
    y_predicted = model.predict(x)
    
    accuracy = accuracy_score(y, y_predicted)
    print('model accuracy: {0:4f}'.format(accuracy))
    oconfusion_matrix = confusion_matrix(y, y_predicted)
    print('Confusion matrix: \n {}'.format(oconfusion_matrix))
    oroc_auc_score = lb.transform(y), lb.transform(y_predicted)		


# In[170]:


custom_accuracy_set(model_random_forest, X_train, X_test, y_train, y_test, train=True)


# In[171]:


custom_accuracy_set(model_random_forest, X_train, X_test, y_train, y_test, train=False)


# In[172]:


#MLP
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix , accuracy_score , precision_score , recall_score , f1_score
from sklearn.feature_selection import SelectPercentile , chi2 , f_classif
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


# In[173]:


X = data11.copy()
# data.head()
Y = data["status"]
# Y = Y == "legitimate"
X.drop(["status" , "url"] , axis=1 , inplace=True)
# print(X.columns)
Cols = X.columns;
# print(X.shape , Y.shape)
Y = Y == "legitimate"


# In[174]:


Scaler = StandardScaler(copy=True , with_mean=True , with_std=True)
X = Scaler.fit_transform(X)


# In[175]:


SP = SelectPercentile(score_func=f_classif , percentile=60)


# In[176]:


X = SP.fit_transform(X , Y)


# In[177]:


print("Number Of Features : " , Cols[SP.get_support()])


# In[178]:


x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size=0.20 , random_state=10 , shuffle=True)


# In[179]:


Model = MLPClassifier(solver='adam' , alpha=0.01 , hidden_layer_sizes=(100 , 100 , 100 , 100) , max_iter=100 , random_state=44)
Ans1 = Model.fit(x_train , y_train)
print("Score Model For Training Data : " , Model.score(x_train , y_train))


# In[180]:


YPred = Ans1.predict(x_test)
conf_matrix = confusion_matrix(y_test ,YPred)
print("Confusion Matrix of the Test Set")
print("-----------")
print(conf_matrix)
print("Precision of the MLP :\t"+str(precision_score(y_test,YPred)))
print("Recall of the MLP    :\t"+str(recall_score(y_test,YPred)))
print("F1 Score of the MLP :\t"+str(f1_score(y_test,YPred)))
print("Accuracy Score of the MLP :\t"+str(accuracy_score(y_test,YPred)))



# In[181]:


#XGBoost
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# In[182]:


pip install xgboost


# In[183]:


import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBRegressor

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# In[184]:


data11.columns


# In[185]:


from sklearn.model_selection import train_test_split


# In[187]:


y = data11.status
X = data11.drop(['status'], axis=1)


# In[188]:


X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)


# In[189]:


categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]


# In[190]:


numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]


# In[191]:


my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()


# In[192]:


numerical_transformer = SimpleImputer(strategy='constant')


# In[193]:


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# In[194]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# In[195]:


print(data11.describe)


# In[196]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)


# In[197]:



y_valid = le.fit_transform(y_valid)


# In[198]:


print(y_train)


# In[199]:


import xgboost as xgb
from sklearn.metrics import accuracy_score


# In[200]:


model = xgb.XGBClassifier()
  


# In[201]:


my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])


# In[202]:


my_pipeline.fit(X_train, y_train)


# In[203]:


preds = my_pipeline.predict(X_valid)
print(accuracy_score(preds,y_valid))


# In[204]:


#Gaussian Naives Bayes method
import numpy as np
import pandas as pd 

import os
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.naive_bayes import GaussianNB #model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import time


# In[205]:


start = time.time()


# In[206]:


removeCols = ['url', 'status']
featuresCol = [i for i in data11.columns if i not in removeCols]
StatusCol = 'status'
data11[StatusCol] = data11[StatusCol]
features, targets = data11[featuresCol], data11[StatusCol]


# In[207]:


class Scaler():
    
    def __init__(self, scaler, feature_range=None):
        self.columns = None
        self.index = None
        self.feature_range = feature_range
        self.scaler = scaler()
        if (isinstance(self.scaler, sklearn.preprocessing._data.MinMaxScaler)
            and isinstance(self.feature_range, tuple)):
            self.scaler = scaler(feature_range = self.feature_range)
        
    def fit(self, X):
        self.scaler = self.scaler.fit(X)
        return self
                
    def transform(self, X):
        scaled_X = self.scaler.transform(X)
        
        if isinstance(X, pd.core.frame.DataFrame):
            self.columns = X.columns
            self.index = X.index
            scaled_X = pd.DataFrame(scaled_X, index=self.index, columns=self.columns)
        return scaled_X
    
    def inverse_transform(self, X):
        inversed_X = self.scaler.inverse_transform(X)
        
        if isinstance(X, pd.core.frame.DataFrame):
            self.columns = X.columns
            self.index = X.index
            inversed_X = pd.DataFrame(inversed_X, index=self.index, columns=self.columns)
        return inversed_X


# In[208]:


train_size = 0.85 
X_train, X_test, y_train, y_test = train_test_split(
    features, targets, shuffle=True, train_size=train_size,
    random_state=42)


# In[209]:


scaler = Scaler(StandardScaler) 
scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)


# In[210]:


naive_bayes = GaussianNB() #call model
naive_bayes = naive_bayes.fit(X_train, y_train)
pred = naive_bayes.predict(X_train)
print(f'\nTraining Accuracy: {round(accuracy_score(y_train, pred)*100, 4)}%')


# In[211]:


X_test = scaler.transform(X_test)
pred = naive_bayes.predict(X_test)


# In[212]:


cm = confusion_matrix(y_test, pred)
display_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=naive_bayes.classes_)
display_cm.plot()
plt.show()


# In[247]:


import os


home_dir = os.path.expanduser('~')
models_dir = os.path.join(home_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

filename = os.path.join(models_dir, 'saved_model_features.sav')


# In[ ]:




