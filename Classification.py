from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd

CSV_COL_NAMES = ['SepalLength','SepalWidth','PetalLegth','PetalWidth','Species']
SPECIES = ['Setosa','Versicolor','Virginica']

# load the data
train_path = tf.keras.utils.get_file("iris_training.csv","https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv","https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COL_NAMES, header=0)
train_y = train.pop('Species')
test = pd.read_csv(test_path,names=CSV_COL_NAMES, header=0)
test_y = test.pop('Species')

# deal with different data types (preprocessing)
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(feature_columns)

# create proper input_data
## no epochs
def input_fn(features,labels,training=True,batch_size=256):
    # 1. create tf.data.Dataset object
    ds = tf.data.Dataset.from_tensor_slices((dict(features),labels))
    # 2. randomize
    if training:
        ds = ds.shuffle(1000).repeat()
    # 3. split data to batch_size 
    return ds.batch(batch_size)

# create the model
classifier = tf.estimator.DNNClassifier(
    feature_columns = my_feature_columns,
    hidden_units=[30,10],
    n_classes=3)

classifier.train(
    input_fn=lambda:input_fn(train,train_y,training=True),steps=5000)

test_result = classifier.evaluate(
    input_fn=lambda: input_fn(test,test_y,training=False))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**test_result))
