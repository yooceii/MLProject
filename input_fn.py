from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import argparse
import ast
import functools
import sys
import tensorflow as tf


def get_num_classes(classes_file):
    classes = []
    with tf.gfile.GFile(classes_file, "r") as f:
        classes = [x for x in f]
    num_classes = len(classes)
    return num_classes


def input_fn(mode,tfrecord_path,batch_size):
    #create input_set and storage
    def _parse_tffn(example_proto, mode):
        #Create a dictionary having two of features
        feature={}
        feature["ink"]=tf.VarLenFeature(dtype=tf.float32)
        feature["shape"]=tf.FixedLenFeature([2],dtype=tf.int64)

        if mode!=tf.estimator.ModeKeys.PREDICT:
            feature["class"]=tf.FixedLenFeature([1],dtype=tf.int64)
        
        parsedFeatures = tf.parse_single_example(example_proto,feature)
        labels = None
        if mode!=tf.estimator.ModeKeys.PREDICT:
            labels = parsedFeatures["class"]
        parsedFeatures["ink"] = tf.sparse_tensor_to_dense(parsedFeatures["ink"])
        return parsedFeatures,labels
    
    def _input_fn():
        dataset = tf.data.TFRecordDataset.list_files(tfrecord_path)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=10)
        dataset = dataset.repeat()
        dataset = dataset.map(
            functools.partial(_parse_tffn,mode=mode),
            num_parallel_calls=10
        )
        dataset = dataset.prefetch(10000)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=1000000)
        
        dataset = dataset.padded_batch(
            batch_size,padded_shapes = dataset.output_shapes)
        features,labels = dataset.make_one_shot_iterator().get_next()
        return features, labels
    
    return _input_fn
    