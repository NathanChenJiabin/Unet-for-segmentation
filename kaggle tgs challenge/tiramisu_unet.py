#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 16:09:20 2018

@author: Jiabin Chen
"""

import numpy as np
import os
import tensorflow as tf
import keras
from keras import backend as K
from keras import layers
from keras.callbacks import Callback
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, Cropping2D
from keras.layers import BatchNormalization
from keras.optimizers import adam
from keras.callbacks import History
from keras.backend import binary_crossentropy
import datetime
from keras.applications.densenet import preprocess_input
import pandas as pd
from keras.utils import multi_gpu_model
import palmier_input
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

smooth = 1e-12
num_channels = 3
num_mask_channels = 1
IMAGE_SIZE = 256

class EvaluateInputTensor(Callback):
    """ Validate a model which does not expect external numpy data during training.
    Keras does not expect external numpy data at training time, and thus cannot
    accept numpy arrays for validation when all of a Keras Model's
    `Input(input_tensor)` layers are provided an  `input_tensor` parameter,
    and the call to `Model.compile(target_tensors)` defines all `target_tensors`.
    Instead, create a second model for validation which is also configured
    with input tensors and add it to the `EvaluateInputTensor` callback
    to perform validation.
    It is recommended that this callback be the first in the list of callbacks
    because it defines the validation variables required by many other callbacks,
    and Callbacks are made in order.
    # Arguments
        model: Keras model on which to call model.evaluate().
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring the evaluation round finished.
            Ignored with the default value of `None`.
    """

    def __init__(self, model, val_model, steps, metrics_prefix='val', verbose=1):
        # parameter of callbacks passed during initialization
        # pass evalation mode directly
        super(EvaluateInputTensor, self).__init__()
        self.val_model = val_model
        self.train_model = model
        self.num_steps = steps
        self.verbose = verbose
        self.metrics_prefix = metrics_prefix

    def on_epoch_end(self, epoch, logs={}):
        self.val_model.set_weights(self.train_model.get_weights())
        results = self.val_model.evaluate(None, None, steps=int(self.num_steps),
                                          verbose=self.verbose)
        metrics_str = '\n'
        for result, name in zip(results, self.val_model.metrics_names):
            metric_name = self.metrics_prefix + '_' + name
            logs[metric_name] = result
            if self.verbose > 0:
                metrics_str = metrics_str + metric_name + ': ' + str(result) + ' '

        if self.verbose > 0:
            print(metrics_str)


def conv_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = layers.Dropout(0.2)(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = concatenate([x, x1], name=name + '_concat')
    return x


def dense_block(x, blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x



def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        require_flatten,
                        weights=None):
    """Internal utility to compute/validate a model's input shape.
    # Arguments
        input_shape: Either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_size: Default input width/height for the model.
        min_size: Minimum input width/height accepted by the model.
        data_format: Image data format to use.
        require_flatten: Whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: One of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
            If weights='imagenet' input channels must be equal to 3.
    # Returns
        An integer shape tuple (may include None entries).
    # Raises
        ValueError: In case of invalid argument values.
    """
    if weights != 'imagenet' and input_shape and len(input_shape) == 3:
        if data_format == 'channels_first':
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[0]) + ' input channels.')
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[-1]) + ' input channels.')
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if weights == 'imagenet' and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting`include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be ' +
                                 str(default_shape) + '.')
        return default_shape
    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[0] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[1] is not None and input_shape[1] < min_size) or
                   (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) +
                                     '; got `input_shape=' +
                                     str(input_shape) + '`')
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        '`input_shape` must be a tuple of three integers.')
                if input_shape[-1] != 3 and weights == 'imagenet':
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[0] is not None and input_shape[0] < min_size) or
                   (input_shape[1] is not None and input_shape[1] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) +
                                     '; got `input_shape=' +
                                     str(input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape

def DenseNet(blocks,
             include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the DenseNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        blocks: numbers of building blocks for the four dense layers.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3
    
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    
    inputs = img_input

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = Model(inputs, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = Model(inputs, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = Model(inputs, x, name='densenet201')
    else:
        model = Model(inputs, x, name='densenet')



    return model


def DenseNet121(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    return DenseNet([6, 12, 24, 16],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes)


def DenseNet169(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    return DenseNet([6, 12, 32, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes)


def make_tiramisu_unet(model_input, load_pretrain_weights=True, img_rows=256, img_cols=256, is_input_tensor=True):
    # model_input = layers.Input(tensor=x_train_batch)
# =============================================================================
#     if is_input_tensor:
#         base_model = DenseNet121(include_top=False,
#                                  weights=None,
#                                  input_tensor=model_input,
#                                  input_shape=(img_rows, img_cols, num_channels),
#                                  pooling=None)
#     else:
#         base_model = DenseNet121(include_top=False,
#                                  weights=None,
#                                  input_shape=(img_rows, img_cols, num_channels),
#                                  pooling=None)
# =============================================================================
    if is_input_tensor:
        base_model = DenseNet([6,8,16,12], include_top=False,
                              weights=None,
                              input_tensor=model_input,
                              input_shape=(img_rows, img_cols, num_channels),
                              pooling=None)    
    else:
        base_model = DenseNet([6,8,16,12], include_top=False,
                              weights=None,
                              input_shape=(img_rows, img_cols, num_channels),
                              pooling=None)  
    
    if load_pretrain_weights:
        base_model.load_weights('weight/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
    
    x = base_model.get_layer('conv5_block12_concat').output
    #x = base_model.get_layer('conv4_block24_concat').output
    blocks = [16, 8, 8, 4]
    # Block_deconv1
    media5 = base_model.get_layer('conv4_block16_concat').output
    up6 = concatenate([Conv2DTranspose(128,kernel_size=2, strides=2, padding='valid', kernel_initializer='he_uniform')(x), media5])
    conv6 = dense_block(up6, blocks[0], 'deconv1')

    # Block_deconv2
    media4 = base_model.get_layer('conv3_block8_concat').output
    up7 = concatenate([Conv2DTranspose(64,kernel_size=2, strides=2, padding='valid', kernel_initializer='he_uniform')(conv6), media4])
    conv7 = dense_block(up7, blocks[1], 'deconv2')

    # Block_deconv3
    media3 = base_model.get_layer('conv2_block6_concat').output
    up8 = concatenate([Conv2DTranspose(64,kernel_size=2, strides=2, padding='valid', kernel_initializer='he_uniform')(conv7), media3])
    conv8 = dense_block(up8, blocks[2], 'deconv3')

    # Block_deconv4
    media2 = base_model.get_layer('conv1/conv').output
    up9 = concatenate([Conv2DTranspose(32,kernel_size=2, strides=2, padding='valid', kernel_initializer='he_uniform')(conv8), media2])
    conv9 = dense_block(up9, blocks[3], 'deconv4')

    # Block_deconv5
    up10 = layers.UpSampling2D()(conv9)
    conv10 = Conv2D(16, 3, padding='same', kernel_initializer='he_uniform', activation='relu')(up10)
    conv10 = Conv2D(num_mask_channels, 1, padding='valid', kernel_initializer='he_uniform', activation='sigmoid')(conv10)
    print(K.get_variable_shape(conv10))
    model = Model(inputs=base_model.input, outputs=conv10)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional vgg16 layers
    # =============================================================================
    #     for layer in base_model.layers:
    #         layer.trainable = False
    # =============================================================================
    return model, base_model


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    return iou(y_pred_pos, y_true)


def iou(y_pred, y_true):
    """Returns a (approx) IOU score
    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7
    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)
    Returns:
        float: IOU score
    """
    (_,H, W, _) = K.get_variable_shape(y_true)
    print(K.get_variable_shape(y_pred))
    pred_flat = K.reshape(y_pred, [-1, H * W])
    true_flat = K.reshape(y_true, [-1, H * W])

    intersection = K.sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = K.sum(
        pred_flat, axis=1) + K.sum(
        true_flat, axis=1) - intersection + 1e-7

    return intersection / denominator


def jaccard_coef_loss(y_true, y_pred):
    # return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)
    return -K.log(iou(y_pred, y_true)) + binary_crossentropy(y_true, y_pred)


def save_model(model):
    model_json = model.to_json()
    with open("tiramisu_unet_logdir/model_v1608.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("tiramisu_unet_logdir/model_v1608.h5")
    print("Saved model to disk")


def save_history(history, suffix):
    filename = 'history/history_tirmisu' + suffix + '.csv'
    pd.DataFrame(history.history).to_csv(filename, index=False)


def main():
    sess = K.get_session()
    train_batch_size = 8
    train_num_examples = palmier_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    train_file_list = ['data/train45.tfrecords', 'data/train149_14.tfrecords', 'data/train149_9.tfrecords']
    x_train_batch, y_train_batch = palmier_input.distorted_inputs(file_list=train_file_list,batch_size=train_batch_size)
    #x_train_batch_proc = preprocess_input(x_train_batch)
    model_input = Input(tensor=x_train_batch)
    train_model, _ = make_tiramisu_unet(model_input, load_pretrain_weights=False)
    train_model.compile(optimizer=adam(lr=1e-5), loss=jaccard_coef_loss,
                           metrics=[jaccard_coef_int],
                           target_tensors=[y_train_batch])
    #parallel_model = multi_gpu_model(train_model, gpus=4)
    # Pass the target tensor `y_train_batch` to `compile`
    # via the `target_tensors` keyword argument:
# =============================================================================
#     parallel_model.compile(optimizer=adam(lr=1e-6), loss=jaccard_coef_loss,
#                            metrics=[jaccard_coef_int],
#                            target_tensors=[y_train_batch])
# =============================================================================
    # train_model.summary()
    test_batch_size = 32
    test_file_list = ['data/test149.tfrecords','data/test149_5.tfrecords']
    x_test_batch, y_test_batch = palmier_input.inputs(file_list=test_file_list, batch_size=test_batch_size)
    #x_test_batch_proc = preprocess_input(x_test_batch)
    # Create a separate test model
    # to perform validation during training
    test_model_input = Input(tensor=x_test_batch)
    test_model, _ = make_tiramisu_unet(test_model_input, load_pretrain_weights=False)

    # Pass the target tensor `y_test_batch` to `compile`
    # via the `target_tensors` keyword argument:
    test_model.compile(optimizer=adam(lr=1e-5),
                       loss=jaccard_coef_loss,
                       metrics=[jaccard_coef_int],
                       target_tensors=[y_test_batch])

    print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))
    # Fit the model using data from the TFRecord data tensors.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    checkpoint_path = "tiramisu_unet_logdir/v1308.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1, save_weights_only=True)
    # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20)
    # learning rate callbacks
    #reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1)
    #tb = tf.keras.callbacks.TensorBoard(log_dir='tiramisulog/', histogram_freq=10, write_graph=False,
    #                                    write_grads=True,write_images=True)
    history = History()
    
    train_model.fit(epochs=50,
                    steps_per_epoch=int(np.ceil(train_num_examples / float(train_batch_size))),
                    callbacks=[EvaluateInputTensor(train_model, test_model, steps=100), history, cp_callback])
    suffix = 'v1608'
    save_history(history, suffix)

    # for layer in base_model.layers:
    #     layer.trainable = False
    # train_model.compile(optimizer=adam(lr=1e-4), loss=jaccard_coef_loss,
    #                     metrics=['binary_crossentropy', jaccard_coef_int],
    #                     target_tensors=[y_train_batch])
    # history2 = History()
    # train_model.fit(epochs=70,
    #                 steps_per_epoch=int(np.ceil(train_num_examples / float(batch_size))),
    #                 callbacks=[EvaluateInputTensor(train_model, test_model, steps=100), history2, cp_callback])
    # suffix = 'second'
    # save_history(history2, suffix)

    # Save the model weights.
    save_model(train_model)
    # Clean up the TF session.
    coord.request_stop()
    coord.join(threads)
    K.clear_session()


if __name__ == '__main__':
    main()
