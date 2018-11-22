# Simple tutorial for Beginners
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

train_path = '/Users/Bin/Downloads/datasets/Titanic, Machine Learning from Disaster/train.csv'
test_path = '/Users/Bin/Downloads/datasets/Titanic, Machine Learning from Disaster/test.csv'


def combine_train_test(train_raw, test_raw):
    full_ds = pd.concat([train_raw, test_raw])
    return full_ds


def get_train_ds(full_ds, train_raw):
    return full_ds[full_ds.index.isin(train_raw.index)]


def get_test_ds(full_ds, test_raw):
    return full_ds[full_ds.index.isin(test_raw.index)]


# drop unused columns
def drop_columns(dataset, cols):
    dataset = dataset.drop(cols, axis=1)
    return dataset


# Name, Age, Sex, Pclass, Embarked
# 1. Sex
def sex_transform(dataset):
    print 'Number of null values in Sex:', sum(dataset.Sex.isnull())
    labelEncoder_ds = LabelEncoder()
    dataset.Sex = labelEncoder_ds.fit_transform(dataset.Sex)
    return dataset


# 2. Embarked
def embarked_transform(dataset):
    print 'Number of null values in Embarked:', sum(dataset.Embarked.isnull())

    # set null rows to the major value 'S'
    null_rows = dataset.Embarked.isnull()
    dataset.loc[null_rows, 'Embarked'] = 'S'

    new_embarked = pd.get_dummies(dataset.Embarked, prefix='Embarked')
    dataset = pd.concat([dataset, new_embarked], axis=1)

    dataset = dataset.drop(['Embarked'], axis=1)
    # we can also drop the extra one column, cause we can use others to represent this column
    dataset = dataset.drop(['Embarked_S'], axis=1)
    return dataset


# 3. Name
def name_transform(dataset):
    print 'Number of null values in Name:', sum(dataset.Name.isnull())

    dataset.Name = dataset.Name.str.split(',').str[1]
    dataset.Name = dataset.Name.str.split('\s+').str[1]
    #     print dataset.Name
    return dataset


# 4. Age
def age_transform(dataset):
    print 'Number of null values in Age:', sum(dataset.Age.isnull())

    dataset = name_transform(dataset)
    name_group_age_mean = dataset.groupby('Name').mean()['Age']

    n_samples = dataset.shape[0]
    n_unique_title = len(name_group_age_mean)

    ref_unique_title = []
    ref_unique_title.append(list(set(dataset.Name)))
    ref_unique_title.append(name_group_age_mean)

    for i in range(0, n_samples):
        if np.isnan(dataset.Age.loc[i]) == True:
            for j in range(0, n_unique_title):
                if dataset.Name.loc[i] == ref_unique_title[0][j]:
                    dataset.Age.loc[i] = ref_unique_title[1][j]

        # transform the 'Age' feature in order to simplify it
        if dataset.Age.loc[i] > 18:
            dataset.Age.loc[i] = 0
        else:
            dataset.Age.loc[i] = 1

    # drop Name
    dataset = dataset.drop(['Name'], axis=1)
    return dataset


def data_preprocess(dataset, cols):
    dataset = drop_columns(dataset, cols)
    dataset = sex_transform(dataset)
    dataset = embarked_transform(dataset)
    dataset = age_transform(dataset)
    return dataset


if __name__ == '__main__':
    train_raw = pd.read_csv(train_path, sep=',')
    test_raw = pd.read_csv(test_path, sep=',')

    y_train = train_raw['Survived']
    X_train = data_preprocess(train_raw, ['PassengerId', 'Cabin', 'Ticket', 'Fare', 'Parch', 'SibSp', 'Survived'])
    X_test = data_preprocess(test_raw, ['PassengerId', 'Cabin', 'Ticket', 'Fare', 'Parch', 'SibSp'])

    clf = SVC(kernel='rbf', random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    svc_predictions = [int(value) for value in y_pred]

    test_PassengerId = test_raw['PassengerId']
    test_Survived = pd.Series(svc_predictions, name='Survived')
    submission = pd.concat([test_PassengerId, test_Survived], axis=1)
    save_path = '/Users/Bin/Downloads/datasets/Titanic, Machine Learning from Disaster/submission.csv'
    submission.to_csv(save_path, sep=',', index=False)
    print 'Fished!'