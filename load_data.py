import numpy as np
import os
import tensorflow as tf

files = os.listdir("sample")
x = []
x_load = []
y = []
y_load = []

def load_data():
    count = 0
    for file in files:
        file = "sample/" + file
        x = np.load(file)
        x = x.astype('float32') / 255.
        x = x[10000:12000, :]
        x_load.append(x)
        y = [count for _ in range(2000)]
        count += 1
        y = np.array(y).astype('float32')
        y = y.reshape(y.shape[0], 1)
        y_load.append(y)

    return x_load, y_load


features, labels = load_data()
features = np.array(features).astype('float32')
labels = np.array(labels).astype('float32')
features=features.reshape(features.shape[0]*features.shape[1],features.shape[2])
labels=labels.reshape(labels.shape[0]*labels.shape[1],labels.shape[2])
def save_tfrecords(data, label, desfile):
    with tf.python_io.TFRecordWriter(desfile) as writer:
        for i in range(len(data)):
            features = tf.train.Features(
                feature = {
                    "data":tf.train.Feature(bytes_list = tf.train.BytesList(value = [data[i].astype(np.float64).tostring()])),
                    "label":tf.train.Feature(int64_list = tf.train.Int64List(value = [label[i]]))
                }
            )
            example = tf.train.Example(features = features)
            serialized = example.SerializeToString()
            writer.write(serialized)
        

save_tfrecords(features, labels, "./eval.tfrecords")