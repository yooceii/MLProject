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


def my_model_fn(features,labels,mode,params):
    def create_input_model(features,labels):
        shapes=features["shape"]
        ink1=features["ink"]
        length_of_tensors = tf.squeeze(tf.slice(shapes,begin=[0,0],size = [params.batch_size,1]))
        inks=tf.reshape(ink1,[params.batch_size,-1,3])
        if labels is not None:
            labels=tf.squeeze(labels)
        return inks,labels,length_of_tensors
    
    def main_path(inks,lengths):
        #First,get into convolution layer:
        conv=inks
        train = tf.estimator.ModeKeys.TRAIN
        for i in range(len(params.num_conv)):
            convolved_input=conv
            if params.batch_norm:
                conv = tf.layers.batch_normalization(
                    conv,training=(mode==train))
            if i > 0 and params.dropout:
                convolved_input = tf.layers.dropout(
                    convolved_input,
                    rate = params.dropout,
                    training=(mode == train)
                )
                convolved = tf.layers.conv1d(
                    convolved_input,
                    filters=params.num_conv[i],
                    kernel_size=params.conv_len[i],
                    activation=None,
                    strides=1,
                    padding="same",
                    name="conv1d_%d" % i)
            
        #Second, we make use of a RNN model, continue using the result from CNN
        convolved = tf.transpose(convolved, [1, 0, 2])
        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=params.num_layers,
        num_units=params.num_nodes,
        dropout=params.dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0,
        direction="bidirectional")
        outputs, _ = lstm(convolved)
        outputs = tf.transpose(outputs, [1, 0, 2])
        mask = tf.tile(
            tf.expand_dims(tf.sequence_mask(lengths,tf.shape(outputs)[1]),2),
            [1,1,tf.shape(outputs)[2]])
        zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))
        outputs = tf.reduce_sum(zero_outside, axis=1)
        return outputs
    
    
    inks,length,labels = create_input_model(features,labels)
    state = main_path(inks,length)
    #add a layer of fully connected layer
    logits=tf.layers.dense(state,params.num_classes)
    
    #loss_computing
    cross_entropy = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    )
    #optimizer
    train_op = tf.contrib.layers.optimize_loss(
        loss=cross_entropy,
        global_step=tf.train.get_global_step(),
        learning_rate=params.learning_rate,
        optimizer="Adam",
      clip_gradients=params.gradient_clipping_norm,
      summaries=["learning_rate", "loss", "gradients", "gradient_norm"])

    predictions = tf.argmax(logits, axis=1)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"logits": logits, "predictions": predictions},
        loss=cross_entropy,
        train_op=train_op,
        eval_metric_ops={"accuracy": tf.metrics.accuracy(labels, predictions)})
    
    