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


#generate input_fn object
def get_input_fn(mode, tfrecord_pattern, batch_size):

    def _parse_tfexample_fn(example_proto, mode):
        feature_to_type = {
            "ink": tf.VarLenFeature(dtype=tf.float32),
            "shape": tf.FixedLenFeature([2], dtype=tf.int64)
        }
        if mode != tf.estimator.ModeKeys.PREDICT:
            feature_to_type["class_index"] = tf.FixedLenFeature(
                [1], dtype=tf.int64)

        parsed_features = tf.parse_single_example(
            example_proto, feature_to_type)
        labels = None
        if mode != tf.estimator.ModeKeys.PREDICT:
            print(parsed_features)
            labels = parsed_features["class_index"]
        parsed_features["ink"] = tf.sparse_tensor_to_dense(
            parsed_features["ink"])
        return parsed_features, labels

    def _input_fn():
        dataset = tf.data.TFRecordDataset.list_files(tfrecord_pattern)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=10)
        dataset = dataset.repeat()
        dataset = dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=10,
            block_length=1)
        dataset = dataset.map(
            functools.partial(_parse_tfexample_fn, mode=mode),
            num_parallel_calls=10)
        dataset = dataset.prefetch(10000)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=1000000)
        dataset = dataset.padded_batch(
            batch_size, padded_shapes=dataset.output_shapes)
        features, labels = dataset.make_one_shot_iterator().get_next()
        return features, labels

    return _input_fn

