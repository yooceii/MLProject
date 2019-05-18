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

from model_fn import model_fn
from input_fn import get_num_classes, get_input_fn

if __name__ == "__main__":

    FLAGS = { 
        "training_data": "data/training.tfrecord-00000-of-00001",
        "eval_data": "data/eval.tfrecord-00000-of-00001",
        "classes_file": "data/training.tfrecord.classes",
        "num_layers": 3,
        "num_nodes": 128,
        "num_conv": "[48, 64, 96]",
        "conv_len": "[5, 5, 3]",
        "batch_norm": False,
        "learning_rate": 0.0001,
        "gradient_clipping_norm": 9.0,
        "dropout": 0.3,
        "steps": 100000,
        "batch_size": 8,
        "model_dir": "./checkpoint",
        "self_test": False
    }

    
    config = tf.estimator.RunConfig(
          model_dir=FLAGS.model_dir,
          save_checkpoints_secs=300,
          save_summary_steps=100)

    model_params = tf.contrib.training.HParams(
        num_layers=FLAGS.num_layers,
        num_nodes=FLAGS.num_nodes,
        batch_size=FLAGS.batch_size,
        num_conv=ast.literal_eval(FLAGS.num_conv),
        conv_len=ast.literal_eval(FLAGS.conv_len),
        num_classes=get_num_classes(FLAGS.classes_file),
        learning_rate=FLAGS.learning_rate,
        gradient_clipping_norm=FLAGS.gradient_clipping_norm,
        cell_type=FLAGS.cell_type,
        batch_norm=FLAGS.batch_norm,
        dropout=FLAGS.dropout)


    train_spec = tf.estimator.TrainSpec(input_fn=get_input_fn(
        mode=tf.estimator.ModeKeys.TRAIN,
        tfrecord_pattern=FLAGS.training_data,
        batch_size=FLAGS.batch_size), max_steps=FLAGS.steps)

    eval_spec = tf.estimator.EvalSpec(input_fn=get_input_fn(
        mode=tf.estimator.ModeKeys.EVAL,
        tfrecord_pattern=FLAGS.eval_data,
        batch_size=FLAGS.batch_size))

    estimator = tf.estimator.Estimator(
        model_fn,
        config=config,
        params=model_params)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    testEstimator = tf.estimator.Estimator(model_fn=model_fn,warm_start_from="checkpoint/", config=config, params=model_params)

    pred_input_fn = lambda: get_input_fn(mode=tf.estimator.ModeKeys.PREDICT, tfrecord_pattern = FLAGS.test_data, batch_size=FLAGS.batch_size)
    testEstimator.predict(pred_input_fn)