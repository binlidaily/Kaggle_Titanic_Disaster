#!/usr/bin/env python
# coding: utf-8

# In[156]:


import pandas as pd
import numpy as np
import featuretools as ft
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns
import matplotlib as plt
# get_ipython().magic(u'matplotlib inline')

import warnings
warnings.filterwarnings('ignore')


# ## Load data

# In[157]:


train_df = pd.read_csv('/Users/Bin/Downloads/datasets/Titanic, Machine Learning from Disaster/train.csv', delimiter=',')
test_df = pd.read_csv('/Users/Bin/Downloads/datasets/Titanic, Machine Learning from Disaster/test.csv', delimiter=',')


# In[158]:


# the result is Index
# train_df.columns
train_df.columns.values


# ## Clean data

# In[159]:


# find missing data
print train_df.isnull().sum()
print ''
print test_df.isnull().sum()


# In[160]:


# fill Fare in test data
test_df.fillna(test_df['Fare'].mean(), inplace=True)


# In[161]:


combi = train_df.append(test_df, ignore_index=True)
passenger_id = test_df['PassengerId']

combi.drop(['Ticket', 'Cabin'], axis=1, inplace=True)

combi['Sex'] = combi['Sex'].apply(lambda x: 0 if x == 'female' else 1)


# In[162]:


# Name
# print combi['Name'].value_counts()
combi['Name'].head()

# for name_str in combi['Name']:
#     combi['Title'] = combi['Name'].str.extract('([A-Za-z]+)\.', expand=True)
combi['Title'] = combi['Name'].str.extract('([A-Za-z]+)\.', expand=True)
print combi['Title'].head()

#replacing the rare title with more common one.
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
combi.replace({'Title': mapping}, inplace=True)
print combi['Title'].head()
combi.drop(['Name'], axis=1, inplace=True)


# In[163]:


# impute missing Age
titles = ['Mr','Miss','Mrs','Master','Rev','Dr']
for title in titles:
    age_to_impute = combi.groupby('Title')['Age'].median()[titles.index(title)]
    combi.loc[ (combi['Age'].isnull()) & (combi['Title'] == title), 'Age'] = age_to_impute
    
combi.isnull().sum()


# In[164]:


# print len(train_df.Embarked.dropna().sum())
# print len(train_df.Embarked)

freq_port = train_df['Embarked'].dropna().mode()[0]
combi['Embarked'].fillna(freq_port, inplace=True)

combi['Embarked'] = combi['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
combi['Title'] = combi['Title'].map({'Mr': 0, 'Mrs':1, 'Miss':2, 'Master':3, 'Rev':4, 'Dr':5}).astype(int)


# In[165]:


print combi.isnull().sum()
combi.fillna(0, inplace=True)
combi.info()


# ## Perform automated feature engineering
# The entity is just a table with a uniquely identifying column known as an index. 

# In[166]:


es = ft.EntitySet(id='titanic_data')

es = es.entity_from_dataframe(entity_id='combi', dataframe=combi.drop(['Survived'], axis=1),
                             variable_types=
                             {
                                 'Embarked': ft.variable_types.Categorical,
                                 'Sex': ft.variable_types.Boolean,
                                 'Title': ft.variable_types.Categorical
                             },
                             index='PassengerId')
print es


# In[167]:


es = es.normalize_entity(base_entity_id='combi', new_entity_id='Embarked', index='Embarked')
es = es.normalize_entity(base_entity_id='combi', new_entity_id='Sex', index='Sex')
es = es.normalize_entity(base_entity_id='combi', new_entity_id='Title', index='Title')
es = es.normalize_entity(base_entity_id='combi', new_entity_id='Pclass', index='Pclass')
es = es.normalize_entity(base_entity_id='combi', new_entity_id='Parch', index='Parch')
es = es.normalize_entity(base_entity_id='combi', new_entity_id='SibSp', index='SibSp')
print es


# In[168]:


primitives = ft.list_primitives()
pd.options.display.max_colwidth = 100
primitives[primitives['type'] == 'aggregation'].head(primitives[primitives['type'] =='aggregation'].shape[0])


# As we can see, the most of "transformation" functions are applied to datetime or time-dependent variables. In our dataset we do not have such variables. Therefore these functions will not be used.

# In[169]:


primitives[primitives['type'] == 'transform'].head(primitives[primitives['type'] == 'transform'].shape[0])


# 1.Now we will apply a deep feature synthesis (dfs) function that will generate new features by automatically applying suitable aggregations, I selected a depth of 2. Higher depth values will stack more primitives.

# In[170]:


features, feature_names = ft.dfs(entityset = es, 
                                 target_entity = 'combi',
                                max_depth = 2)


# This is a list of new features. For example, "Title.SUM(combine.Age)" means the sum of Age values for each unique value of Title.

# In[171]:


feature_names


# In[172]:


print len(feature_names)


# In[173]:


features[features['Age'] == 22][['Title.SUM(combi.Age)', 'Age', 'Title']].head()


# By using "featuretools", we were able to generate 146 features just in a moment.
# 
# The "featuretools" is a powerful package that allows saving time to create new features from multiple tables of data. However, it does not completely subsitute the human domain knowledge. Additionally, now we are facing another problem known as the "curse of dimensionality".

# ### Determine collinear features

# In[174]:


# Threshold for removing correlated variables
threshold = 0.96

# Absolute value correlation matrix
corr_matrix = features.corr().abs()
# np.trup: Upper triangle of an array.
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.head(50)


# In[175]:


# Select columns with correlations above threshold
collinear_features = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d features to remove.' % (len(collinear_features)))


# In[176]:


features_filtered = features.drop(columns=collinear_features)

print('The number of features that passed the collinearity threshold: ', features_filtered.shape[1])


# ### Detect the most relevant features using linear models penalized with the L1 norm

# In[177]:


features_positive = features_filtered.loc[:, features_filtered.ge(0).all()]


# In[178]:


train_X = features_positive[:train_df.shape[0]]
train_y = train_df['Survived']

test_X = features_positive[train_df.shape[0]:]


# Since the number of features is smaller than the number of observations in "train_X", the parameter "dual" is equal to False.

# In[179]:


lsvc = LinearSVC(C=0.01, penalty='l1', dual=False).fit(train_X, train_y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(train_X)
X_selected_df = pd.DataFrame(X_new, columns=[train_X.columns[i] for i in range(len(train_X.columns)) if model.get_support()[i]])
X_selected_df.shape


# In[180]:


print X_selected_df.columns


# ### Training and testing the simple model 
# Finally, we will create a basic random forest classifier with 2000 estimators. Please notice that I skip essential steps such as crossvalidation, the analysis of learning curves, etc.

# In[181]:


random_forest = RandomForestClassifier(n_estimators=2000, oob_score=True)
random_forest.fit(X_selected_df, train_y)


# In[182]:


y_pred = random_forest.predict(test_X[X_selected_df.columns])


# In[183]:


print y_pred


# In[184]:


my_submission = pd.DataFrame({'PassengerId': passenger_id, 'Survived': y_pred})
my_submission.to_csv('auto_ft_submission.csv', index=False)


# In[ ]:




