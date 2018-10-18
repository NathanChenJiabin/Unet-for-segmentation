#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 13:50:51 2018

@author: Jiabin CHEN
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os.path
# import re
#import time

import numpy as np
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import palmier_input

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def conv_conv_pool(input_,
                   n_filters,
                   training,
                   flags,
                   name,
                   pool=True,
                   activation=tf.nn.relu):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}
    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions
    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(
                net,
                F, (3, 3),
                activation=None,
                padding='same',
                kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg),
                name="conv_{}".format(i + 1))
            net = tf.layers.batch_normalization(
                net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(
            net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool


def upconv_2D(tensor, n_filter, flags, name):
    """Up Convolution `tensor` by 2 times
    Args:
        tensor (4-D Tensor): (N, H, W, C)
        n_filter (int): Filter Size
        name (str): name of upsampling operations
    Returns:
        output (4-D Tensor): (N, 2 * H, 2 * W, C)
    """

    return tf.layers.conv2d_transpose(
        tensor,
        filters=n_filter,
        kernel_size=2,
        strides=2,
        padding = 'same',
        kernel_regularizer=tf.contrib.layers.l2_regularizer(flags.reg),
        name="upsample_{}".format(name))


def upconv_concat(inputA, input_B, n_filter, flags, name):
    """Upsample `inputA` and concat with `input_B`
    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation
    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    up_conv = upconv_2D(inputA, n_filter, flags, name)

    return tf.concat(
        [up_conv, input_B], axis=-1, name="concat_{}".format(name))


def make_unet(X, training, flags=None):
    """Build a U-Net architecture
    Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batch_normalization layers
    Returns:
        output (4-D Tensor): (N, H, W, 1)
            
    Notes:
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    """
    # net = X / 127.5 - 1
    conv1, pool1 = conv_conv_pool(X, [8, 8], training, flags, name=1)
    conv2, pool2 = conv_conv_pool(pool1, [16, 16], training, flags, name=2)
    conv3, pool3 = conv_conv_pool(pool2, [32, 32], training, flags, name=3)
    conv4, pool4 = conv_conv_pool(pool3, [64, 64], training, flags, name=4)
    conv5 = conv_conv_pool(
        pool4, [128, 128], training, flags, name=5, pool=False)

    up6 = upconv_concat(conv5, conv4, 64, flags, name=6)
    conv6 = conv_conv_pool(up6, [64, 64], training, flags, name=6, pool=False)

    up7 = upconv_concat(conv6, conv3, 32, flags, name=7)
    conv7 = conv_conv_pool(up7, [32, 32], training, flags, name=7, pool=False)

    up8 = upconv_concat(conv7, conv2, 16, flags, name=8)
    conv8 = conv_conv_pool(up8, [16, 16], training, flags, name=8, pool=False)

    up9 = upconv_concat(conv8, conv1, 8, flags, name=9)
    conv9 = conv_conv_pool(up9, [8, 8], training, flags, name=9, pool=False)

    return tf.layers.conv2d(
        conv9,
        1, (1, 1),
        name='final_layer',
        activation=tf.nn.sigmoid,
        padding='same')


def dice_score(y_pred, y_true):
    """Returns a (approx) dice score
    intesection = y_pred.flatten() * y_true.flatten()
    Then, dice = 2 * intersection  + 1e-7 / (y_pred.sum() + y_true.sum() + 1e-7)
    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)
    Returns:
        float: dice score
    """
    H, W, _ = y_true.get_shape().as_list()[1:]

    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])

    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = tf.reduce_sum(
        pred_flat, axis=1) + tf.reduce_sum(
            true_flat, axis=1) + 1e-7

    return tf.reduce_mean(intersection / denominator)

def iou(y_pred, y_true):
    """Returns a (approx) iou score
    intesection = y_pred.flatten() * y_true.flatten()
    Then, iou = intersection+ 1e-7 / (y_pred.sum() + y_true.sum() - intersection + 1e-7) 
    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)
    Returns:
        float: iou score
    """
    H, W, _ = y_true.get_shape().as_list()[1:]

    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])

    intersection =  tf.reduce_sum(pred_flat * true_flat, axis=1) 
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) - intersection

    return tf.reduce_mean((intersection+ 1e-7) / (denominator + 1e-7))


def l_2norm(y_pred, y_true):
    return tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
    #pred_flat = tf.reshape(y_pred, [y_pred.get_shape()[0], -1])
    #true_flat = tf.reshape(y_true, [y_pred.get_shape()[0], -1])
    #return tf.reduce_mean(tf.reduce_sum((pred_flat-true_flat)**2, axis=1))


def focal_loss(y_pred,y_true):
    gamma = 2
    alpha = 0.25
    pred_flat = tf.reshape(y_pred, [y_pred.get_shape()[0], -1])
    true_flat = tf.reshape(y_true, [y_pred.get_shape()[0], -1])
    val1=tf.multiply(tf.log(pred_flat+1e-10),true_flat)
    val2=tf.multiply(tf.log(1.0-pred_flat+1e-10),1.0-true_flat)
    val_tot=tf.multiply(-alpha*((1-pred_flat)**gamma),val1) - tf.multiply((1-alpha)*((pred_flat)**gamma),val2)
    return tf.reduce_mean(tf.reduce_sum(val_tot, axis=1))


def huber_loss(y_pred, y_true):
    return tf.losses.huber_loss(labels=y_true, predictions=y_pred)


def make_train_op(y_pred, y_true):
    """Returns a training operation
    Loss function = - iou(y_pred, y_true)
    IOU is
        (the area of intersection)
        --------------------------
        (the area of two boxes)
    Args:
        y_pred (4-D Tensor): (N, H, W, 1)
        y_true (4-D Tensor): (N, H, W, 1)
    Returns:
        train_op: minimize operation
    """
    #loss = -IOU_(y_pred, y_true)
    #loss = huber_loss(y_pred, y_true)
    loss = focal_loss(y_pred, y_true)

    global_step = tf.train.get_or_create_global_step()

    optim = tf.train.AdamOptimizer(learning_rate=0.0001)
    return optim.minimize(loss, global_step=global_step)


def read_flags():
    """Returns flags"""

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--epochs", default=20, type=int, help="Number of epochs")

    parser.add_argument("--batch-size", default=16, type=int, help="Batch size")

    parser.add_argument(
        "--logdir", default="logdir-v3", help="Tensorboard log directory")

    parser.add_argument(
        "--reg", type=float, default=0.1, help="L2 Regularizer Term")

    parser.add_argument(
        "--ckdir", default="models-v7", help="Checkpoint directory")
    
    parser.add_argument(
        "--ckdir_target", default="models-v7", help="Checkpoint directory")

    flags = parser.parse_args()
    return flags


def main(flags, retrain):
    
    n_train = palmier_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
   
    n_test = palmier_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
   
    train_logdir = os.path.join(flags.logdir, "train-focalloss")

    tf.reset_default_graph()
   
    train_file_list = ['data/train45.tfrecords', 'data/train149_14.tfrecords', 'data/train149_9.tfrecords']
    images, labels = palmier_input.distorted_inputs(file_list=train_file_list,batch_size=flags.batch_size)
    mode = tf.placeholder(tf.bool, name="mode")
    pred = make_unet(images, mode, flags)
    

    tf.summary.histogram("Predicted Mask", pred)
    tf.summary.image("Predicted Mask", pred)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_op = make_train_op(pred, labels)

    loss_op = l_2norm(pred, labels)
    IOU_op = iou(pred, labels)
    
    tf.summary.scalar("IOU", IOU_op)
    tf.summary.scalar('l2norm_loss', loss_op)

    test_file_list = ['data/test149.tfrecords','data/test149_5.tfrecords']
    images_test, labels_test = palmier_input.inputs(file_list=test_file_list, batch_size=flags.batch_size)
    tf.get_variable_scope().reuse_variables()
    pred_test = make_unet(images_test, mode, flags)
    IOU_test = iou(pred_test, labels_test)
    l2loss_test = l_2norm(pred_test, labels_test)

    summary_op = tf.summary.merge_all()
    
    saver = tf.train.Saver()
    
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        train_summary_writer = tf.summary.FileWriter(train_logdir, sess.graph)
       
        if retrain:
            latest_check_point = tf.train.latest_checkpoint(flags.ckdir)
            saver.restore(sess, latest_check_point)
            print('restore the model successufully !!!')

        else:
            try:
                os.rmdir(flags.ckdir)
            except :
                pass
            #os.mkdir(flags.ckdir_target)
            sess.run(init)
            print('initialization the model !')

        try:
            global_step = tf.train.get_global_step(sess.graph)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for epoch in range(flags.epochs):
                count = 0
                total_average_l2loss_per_epoch = 0
                total_average_iou_per_epoch = 0
                for step in range(0, n_train, flags.batch_size):
                    _, v_iou, v_loss, step_summary, global_step_value = sess.run(
                        [train_op, IOU_op, loss_op, summary_op, global_step], feed_dict={mode: True})
                    train_summary_writer.add_summary(step_summary, global_step_value)
                    count += 1
                    total_average_l2loss_per_epoch += v_loss
                    total_average_iou_per_epoch += v_iou
                    
                    if step % (1000*flags.batch_size) == 0:                        
                        format_str = ('epoch %d step %d, current_IOU = %.8f, current_l_2_loss = %.8f')                
                        print(format_str % (epoch+1, step, v_iou, v_loss))
                  
                        
                format_str1 = ('average l2 loss of epoch%d: %.8f, and average iou: %.8f')
                print(format_str1 % (epoch+1, total_average_l2loss_per_epoch/count, total_average_iou_per_epoch/count))

                # Evaluation
                count = 0
                average_test_iou = 0
                average_test_l2loss = 0
                for step in range(0, n_test, flags.batch_size):
                    step_iou_test, step_l2loss_test = sess.run([IOU_test, l2loss_test], feed_dict={mode: False})
                    average_test_iou += step_iou_test 
                    average_test_l2loss += step_l2loss_test 
                    count+=1
                    
                format_str2 = ('epoch %d , average_iou = %.8f, average_l_2_loss = %.8f')                
                print(format_str2 % (epoch+1, average_test_iou/count, average_test_l2loss/count))                
                
                saver.save(sess, "models/epoch%d-model.ckpt"%epoch)                

        finally:
            coord.request_stop()
            coord.join(threads)
            saver.save(sess, "models/model.ckpt")
            print('save successufully the model !')


if __name__ == '__main__':
    flags = read_flags()
    retrain = False
    main(flags, retrain)