from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
pip install --upgrade tensorflow
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf



# load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
ytrain = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
# EDA graphs
dftrain.age.hist(bins=20)
dftrain['class'].value_counts().plot(kind='barh')

# deal with different data types (preprocessing)
CATEGORICAL_FEATURES = ['sex','n_siblings_spouses','parch','class','deck','embark_town','alone']
NUMERIC_FEATURES = ['age','fare']

feature_columns = []
for feature_name in CATEGORICAL_FEATURES:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocabulary))
    print(feature_columns)
for feature_name in NUMERIC_FEATURES:
    feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))
    
print(feature_columns)


# create proper input_data
def make_input_fn(data_df,label_df,num_epochs=10,shuffle=True,batch_size=32):
    def input_function():
        # 1. dictiona"rize" -- create tf.data.Dataset object
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
        # 2. randomize
        if shuffle:
            ds = ds.shuffle(1000)
        # 3. split data to batch_size & take care of epochs
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain,ytrain)
test_input_fn = make_input_fn(dfeval,y_eval,num_epochs=1,shuffle=False)

# create the model
## 1. define the model
linear_mdl = tf.estimator.LinearClassifier(feature_columns=feature_columns)
## 2. fit in the training dataset
linear_mdl.train(train_input_fn)
## 3. apply on testing dataset
result = linear_mdl.evaluate(test_input_fn)

print(result['accuracy'])
print(result)

# how to see the results model predicted
results = list(linear_mdl.predict(test_input_fn))
print(results[0]['probabilities'])