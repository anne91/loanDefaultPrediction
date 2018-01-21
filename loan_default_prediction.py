# -*- coding: utf-8 -*-
"""
Anne Bernhart


Data Science Challenge



Loan Default Prediction - Imperial College London

Instructions:
Develop an algorithm using Python to predict loan defaults and the amount of the default

o Code a data loading, pre-processing and preliminary analysis script. Explain how
    you efficiently deal with the large number of training instances, how you handle the
    high dimensionality of the problem as well as any observed data problems (e.o.
                                                                              missing data).
o Explain if and how you will use the unlabeled data set.

o Explain which machine learning algorithm is in your opinion theoretically best suited
    for the given task.
    
o Implement the machine learning algorithm, that you think is most appropriate (using
public available Python packages and a tailor made script).
   Train and validate the model and explain the chosen procedure and your results.

o Discuss pro and cons of the developed approach and suggest improvement
directions 

"""

import numpy as np
import pandas as pd
import copy
import time
from sklearn import preprocessing
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt

#date = time.strftime("%Y%m%d")

#filename = r"C:\Users\bma\Desktop\KaggleChallenge\data\train_v2.csv"
#data = pd.read_csv(filename) # DtypeWarning: Columns (417) have mixed types. Specify dtype option on import or set low_memory=False. ???


#data = pd.read_csv(r"C:\Users\bma\Desktop\KaggleChallenge\data\train_v2_to_100000.csv", dtype = 'float64') # OK, no more error in the import such as 
#DtypeWarning: Columns (135) have mixed types. Specify dtype option on import or set low_memory=False.
#  interactivity=interactivity, compiler=compiler, result=result)
data = pd.read_csv(r"C:\Users\bma\Desktop\KaggleChallenge\data\train_v2.csv", dtype = 'float64') #@ OK, no more error in the import such as 

data[['f776','f777']].describe() # categorical: indeed, 0 or 1


data.head()
data.info() # dtypes: float64(653), int64(99), object(19)
data.describe()
#data.describe(include=['O'])
#data.describe(include=['category'])

##data.loc[:, data.dtypes == object]
#data.select_dtypes(include=['O'])
## for instance: f391
#data['f391']
#test = pd.to_numeric(data['f391'], errors='coerce')
#test[test.isnull()]
#data['f391'].iloc[test[test.isnull()].index].unique() # so the import was wrong: we imported some NA (checked in the csv) into strings 'nan'



# We have 771 columns but f1 to f778: so it means that some f... are missing between 1 and 778
cols = list(data)
for i in range(1,779):
    if 'f' + str(i) not in cols:
        print('Column f' + str(i) , ' does not exist.')
    
        
# What do we do with the categorical values? do we have other categorical values?
# TODO: We could implement a function that detect if it contains categorical values or not (not implemented here)


################ Pre Analysis ################ 
# Pre-processing

plt.plot(data['id'][:], data['loss'][:].sort_values(ascending=False))
plt.show()

len(data[data['loss'] > 0])

# only around 10000 observations are actually default loans.

# So we could devide the problem in 2:
    # First: detect if yes or no this is a case of default loan -> and for that use a dataset with around 50% of yes and 50% of no
    # Then: if yes, compute the amount of the default --> and train on data with indeed a default loan


# do we have some missing values in some columns?
data['f1'].value_counts(dropna=False).head()
missing_values = data.isnull().sum()
missing_values.sort_values(ascending=False).plot()

# So to solve the issue of missing values we could:
    #1. Remove the columns where the missing values are above > 2500 rows
    #2. Replace the missing values by the mean or median
# We could also remove all rows where too many missing values (not implemented here) 
 
# Let's keep only the columns for which there are not more than 2500 rows with missing values
skip_NAN = 2500
cols_to_ignore = list(missing_values[missing_values > skip_NAN].index)
len(cols_to_ignore)

df = data[[i for i in cols if i not in cols_to_ignore]]

df_missing_values = df.isnull().sum()
df_missing_values = df_missing_values[df_missing_values > 0]
df_cols_missing_values = list(df_missing_values.index)

# Replace missing values by median
for i in df_cols_missing_values: 
    df.loc[df[i].isnull(), i] = df[i].median()  


df_X = copy.deepcopy(df)
del df_X['loss']
df_X.set_index('id',inplace=True)



################ High Dimensionality ################ 
# we have so many features that we could do a PCA (Principal Component Analysis) to select some features out of all of them available

# But before implementing that, maybe we can already try a Neural Network on all the features.


# In any case (especially for PCA), good to standardize (scale) our data
# But what about columns containing categorical values...? we should not normalize them.

# TODO: implement function to detect categorical columns
cols_category = ['f776', 'f777']
cols_not_category = [i for i in list(df_X) if i not in cols_category]

# Standardization: mean removal and variance scaling
scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(df_X[cols_not_category])
#scaler.mean_
#scaler.scale_
X_scaled = scaler.transform(df_X[cols_not_category])

# Add the categorical columns
X_scaled = np.append(X_scaled, df_X[cols_category].as_matrix(), axis=1)

# Try PCA
pca = PCA()
pca.fit(X_scaled)
#print(pca.explained_variance_)
plt.plot(pca.explained_variance_, linewidth=2)

# so let's reduce our dimensions space to 50
pca_reduc = PCA(n_components=50)
X_reduced = pca_reduc.fit(X_scaled).transform(X_scaled)


################ Machine Learning / Training models  ################ 

# Due to the high proportion of cases of non default loans, let's select randomly 10000 rows where the loss = 0
# to balance the proportions of cases defaults and not defaults

# select randomly 10000 rows where the loss = 0
random_index = np.array(df.loc[df['loss'] == 0].sample(n=10000, random_state=2).index) # we can use random_state for reproducibility

default_loans_index = np.array(df.loc[df['loss'] > 0].index)

selection_index = np.append(random_index, default_loans_index)
selection_index.sort()


################ STEP 1: Predicting if YES or NO there this is a cae of default loan ################ 
"""
This is a classification and supervised problem - because we have labelled data
Can be addressed with an algorithm of classification:
    - Logistic regression
    - SVM
    - Neural Network

Let's take a Neural Network
"""

# Create new columns containing our label 0: no default loan and 1: case of default loan
df['default_loan_Y'] = 0
df.loc[df['loss'] > 0, 'default_loan_Y'] = 1
df['default_loan_Y'].value_counts() # OK
df['default_loan_N'] = 1 - df['default_loan_Y']
df[['default_loan_Y','default_loan_N']].drop_duplicates() # check: OK

X_train_step1 = X_reduced[selection_index] # use X_reduced if we want to use the output of the PCA, X_scaled if we want to train with all columns
Y_train_step1 = df[['default_loan_Y','default_loan_N']].as_matrix()[selection_index]


# Validation in deep learning: Commonly use validation split rather than crossvalidation
# (single validation score is based on large amount of data, and is reliable)

n_cols = X_train_step1.shape[1]

# Model specification: neural network architecture
model1 = Sequential()
model1.add(Dense(10, activation='relu', input_shape=(n_cols,)))
model1.add(Dense(10, activation='relu'))
model1.add(Dense(2, activation='softmax'))

# Compiling the model
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting a model: with validation split and early stopping
early_stopping_monitor = EarlyStopping(patience=2)

#model.fit(X_scaled, Y_train)
print('\n ----> Starting fitting model, at %s \n' %time.strftime('%H:%M.%S'))

model1.fit(X_train_step1, Y_train_step1, validation_split=0.3,
          epochs=4, callbacks=[early_stopping_monitor])

print('\n ----> Finishing fitting model, at %s \n' %time.strftime('%H:%M.%S'))

test1 = model1.predict(X_train_step1[:])
compare1 = Y_train_step1[:]
print(sum(test1), sum(compare1))


################ STEP 2: Predicting the amount of the defaults in case of default loans ################ 
"""
Predicting the loss is a regression problem: we want to predict an amount (here a value bewteen 0 and 100)
based on features
Since we have labelled data, we can train a model - this problem is a supervised one,

We can address this problem with several Machine Learning algorithms that are 1) supervised and 2) for regression analysis,
such as:
    - Linear regression
    - Linear regression with features x^2, ...
    - Linear regression with splines
    - Neural Network

Due to the high amount of features, let's try a neural network. We'll use Keras' framework.
"""

X_train = X_reduced[default_loans_index]
#X_train = X_scaled[:] # if we want to train with all columns
Y_train = df['loss'].as_matrix()[default_loans_index]


# Validation in deep learning: Commonly use validation split rather than crossvalidation
# (single validation score is based on large amount of data, and is reliable)

n_cols = X_train.shape[1]

# Model specification: neural network architecture
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Fitting a model: with validation split and early stopping
early_stopping_monitor = EarlyStopping(patience=2)

#model.fit(X_scaled, Y_train)
print('\n ----> Starting fitting model, at %s \n' %time.strftime('%H:%M.%S'))
model.fit(X_train, Y_train, validation_split=0.3,
          epochs=10, callbacks=[early_stopping_monitor])
print('\n ----> Finishing fitting model, at %s \n' %time.strftime('%H:%M.%S'))

test = model.predict(X_train[:])
compare = Y_train[:]
print(sum(test), sum(compare))
#predictions = model.predict(data_to_predict_with)




################ Predict on test dataset ################ 
data_test = pd.read_csv(r"C:\Users\bma\Desktop\KaggleChallenge\data\test_v2.csv", dtype = 'float64') #@ OK, no more error in the import such as 

# Check that the data_test has the same format
data_test.head()
data_test.info() # float64(654), int64(96), object(19), uint64(1)
data_test.describe()
data_test[['f776','f777']].describe() # categorical: indeed, 0 or 1

len(data_test['id'].unique())

cols_test = list(data_test)
for i in range(1,779):
    if 'f' + str(i) not in cols_test:
        print('Column f' + str(i) , ' does not exist.')


# Keep only columns with enough values
df_test = data_test[[i for i in cols_test if i not in cols_to_ignore]]

df_missing_values_test = df_test.isnull().sum()
df_missing_values_test = df_missing_values_test[df_missing_values_test > 0]
df_cols_missing_values_test = list(df_missing_values_test.index)


# Replace missing values by median
for i in df_cols_missing_values_test: 
    df_test.loc[df_test[i].isnull(), i] = df_test[i].median()  #  TODO: It would have been better to do the median of all data train + test


df_X_test = copy.deepcopy(df_test)
df_X_test.set_index('id',inplace=True)

X_scaled_test = scaler.transform(df_X_test[cols_not_category])

# Add the categorical columns
X_scaled_test = np.append(X_scaled_test, df_X_test[cols_category].as_matrix(), axis=1)

# Apply PCA
X_reduced_test = pca_reduc.fit(X_scaled_test).transform(X_scaled_test)

# Apply model1 of detection of default or non default
Y_test_default = model1.predict(X_reduced_test[:])

df_Y_test_default = pd.DataFrame(Y_test_default, columns=['default_loan_Y','default_loan_N'])
df_Y_test_default['loss_Y_N'] = 0
df_Y_test_default.loc[df_Y_test_default['default_loan_Y'] > df_Y_test_default['default_loan_N'] ,'loss_Y_N'] = 1

Y_test_default_result = df_Y_test_default[['loss_Y_N']].as_matrix()

X_reduced_test_default = np.append(X_reduced_test, Y_test_default_result, axis=1)
X_reduced_test_default = pd.DataFrame(X_reduced_test_default)

# Store index of cases of default loans
index_default_loans = X_reduced_test_default[X_reduced_test_default[50] > 0]
index_default_loans['row_nb'] = [i for i in range(len(index_default_loans))]
index_default_loans = index_default_loans[['row_nb']]
index_default_loans = index_default_loans.reset_index()

Y_test_default_result = X_reduced_test_default[X_reduced_test_default[50] > 0]
del Y_test_default_result[50]

X_test_amount = Y_test_default_result.as_matrix()

# Apply model of prediction of amount of defaults
Y_test_amount = model.predict(X_test_amount)

max(Y_test_amount) # the max is > 100: it is obviously not correct



################ Prepare the output: id, loss ################ 
df_Y_test_amount = pd.DataFrame(Y_test_amount, columns=['loss'])
df_Y_test_amount['loss'] = df_Y_test_amount['loss'].round(0)
df_Y_test_amount['loss'].value_counts()

df_Y_test_amount.loc[df_Y_test_amount['loss'] > 100, 'loss'] = 100.0

# Add the original ID
df_Y_test_amount_with_ID = pd.merge(df_Y_test_amount, index_default_loans, how='inner', left_index=True, right_index=True)

df_result = pd.merge(df_test[['id']], df_Y_test_amount_with_ID, how='left', left_on='id', right_on='index')

df_result = df_result[['id','loss']].fillna(0.0)
df_result['id'] = df_result['id'].astype('int')


plt.plot(df_result['id'][:], df_result['loss'][:].sort_values(ascending=False))
plt.show()


df_result.to_csv(r"C:\Users\bma\Desktop\KaggleChallenge\20180121_loss_prediction.csv",columns=['id','loss'],index=False)




"""
Propositions to improve the pipeline:
    - implement recognition of catagrocial and non categorical columns
    - improve the model so that amount of default can not exceed 100
    - try different models of neural network: more hidden layers, more nodes
    - to fill missing values: use the median over all data (train + test)
    - for memory gain: do not use always float64
    
"""


