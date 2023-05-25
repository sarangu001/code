#!/usr/bin/env python
# coding: utf-8

# In[1]:


#RF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('features_87.csv')


# In[3]:


data.shape


# In[4]:


data.describe()


# In[5]:


pd.set_option('display.max_rows', 500)
data.isna().sum()


# In[6]:


pd.reset_option('display.max_rows')

original_dataset = data.copy()

class_map = {'legitimate':0, 'phishing':1}
original_dataset['status'] = original_dataset['status'].map(class_map)


# In[7]:


corr_matrix = original_dataset.corr()

plt.figure(figsize=(60,60))
color = plt.get_cmap('viridis').copy()   # for showing colors
color.set_bad('lightblue') 
sns.heatmap(corr_matrix, annot=True, linewidth=0.4, cmap=color)
plt.savefig('heatmap')
plt.show()


# In[9]:


corr_matrix.shape


# In[10]:


corr_matrix['status']


# In[11]:


status_corr = corr_matrix['status']


# In[12]:


status_corr.shape


# In[13]:


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


# In[14]:


features_selected = feature_selector_correlation(status_corr, 0.2)
features_selected


# In[15]:


selected_features = [i for (i,j) in features_selected if i != 'status']
selected_features


# In[16]:


X_selected = original_dataset[selected_features]
X_selected


# In[17]:


y = original_dataset['status']
y


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X_selected, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle = True)
													
													
												


# In[20]:


from sklearn.ensemble import RandomForestClassifier


# In[21]:


model_random_forest = RandomForestClassifier(n_estimators=350,
                                             random_state=42,
                                             )
											 
											 


# In[22]:


model_random_forest.fit(X_train,y_train)


# In[23]:


from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


# In[27]:


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

custom_accuracy_set(model_random_forest, X_train, X_test, y_train, y_test, train=True)

custom_accuracy_set(model_random_forest, X_train, X_test, y_train, y_test, train=False)


# In[28]:


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


# In[30]:


X = data.copy()
# data.head()
Y = data["status"]
# Y = Y == "legitimate"
X.drop(["status" , "url"] , axis=1 , inplace=True)
# print(X.columns)
Cols = X.columns;
# print(X.shape , Y.shape)
Y = Y == "legitimate"


# In[31]:


Scaler = StandardScaler(copy=True , with_mean=True , with_std=True)
X = Scaler.fit_transform(X)


# In[32]:


SP = SelectPercentile(score_func=f_classif , percentile=60)


# In[33]:


X = SP.fit_transform(X , Y)


# In[34]:


print("Number Of Features : " , Cols[SP.get_support()])


# In[35]:


x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size=0.20 , random_state=10 , shuffle=True)


# In[36]:


Model = MLPClassifier(solver='adam' , alpha=0.01 , hidden_layer_sizes=(100 , 100 , 100 , 100) , max_iter=100 , random_state=44)
Ans1 = Model.fit(x_train , y_train)
print("Score Model For Training Data : " , Model.score(x_train , y_train))


# In[37]:


YPred = Ans1.predict(x_test)
conf_matrix = confusion_matrix(y_test ,YPred)
print("Confusion Matrix of the Test Set")
print("-----------")
print(conf_matrix)
print("Precision of the MLP :\t"+str(precision_score(y_test,YPred)))
print("Recall of the MLP    :\t"+str(recall_score(y_test,YPred)))
print("F1 Score of the MLP :\t"+str(f1_score(y_test,YPred)))
print("Accuracy Score of the MLP :\t"+str(accuracy_score(y_test,YPred)))


# In[38]:


#XGBoost
import numpy as np 
import pandas as pd 
import os


# In[39]:


pip install xgboost


# In[40]:


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

data.columns


# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


y = data.status
X = data.drop(['status'], axis=1)


# In[43]:


X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)


# In[44]:


categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]


# In[45]:


numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]


# In[46]:


my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()


# In[47]:


numerical_transformer = SimpleImputer(strategy='constant')


# In[48]:


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# In[49]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# In[50]:


print(data.describe)


# In[51]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)


# In[52]:


y_valid = le.fit_transform(y_valid)


# In[53]:


print(y_train)


# In[54]:


import xgboost as xgb
from sklearn.metrics import accuracy_score


# In[55]:


model = xgb.XGBClassifier()


# In[56]:


my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])


# In[57]:


my_pipeline.fit(X_train, y_train)


# In[58]:


preds = my_pipeline.predict(X_valid)
print(accuracy_score(preds,y_valid))


# In[59]:


#Gaussian Naive Bayes method
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


# In[60]:


start = time.time()


# In[61]:


removeCols = ['url', 'status']
featuresCol = [i for i in data.columns if i not in removeCols]
StatusCol = 'status'
data[StatusCol] = data[StatusCol]
features, targets = data[featuresCol], data[StatusCol]


# In[62]:


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


# In[63]:


scaler = Scaler(StandardScaler) 
scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)


# In[64]:


naive_bayes = GaussianNB() #call model
naive_bayes = naive_bayes.fit(X_train, y_train)
pred = naive_bayes.predict(X_train)
print(f'\nTraining Accuracy: {round(accuracy_score(y_train, pred)*100, 4)}%')


# In[65]:


X_test = scaler.transform(X_test)
pred = naive_bayes.predict(X_test)


# In[66]:


cm = confusion_matrix(y_test, pred)
display_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=naive_bayes.classes_)
display_cm.plot()
plt.show()


# In[67]:


print(f'Accuracy: {round(accuracy_score(y_test, pred)*100, 4)}%')
end = time.time()
finalTime = end - start
print("Total time (in seconds): " + str(finalTime))


# In[70]:


import os
import joblib


home_dir = os.path.expanduser('~')
models_dir = os.path.join(home_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

filename = os.path.join(models_dir, 'saved_model_87.sav') # Saving it into sav model


# In[ ]:




