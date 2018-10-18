
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd

import os

import random
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')

import cv2
from sklearn.model_selection import StratifiedKFold

from keras.models import Model
from keras.layers import Input,  Activation
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import Callback, LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers
from keras import layers
from keras.losses import binary_crossentropy

import tensorflow as tf

from keras.preprocessing.image import array_to_img, img_to_array, load_img

cv_total = 5

version = 1

img_size_ori = 101
img_size_target = 101


def upsample(img):
    return img


def downsample(img):
    return img


print('Load data ...')
# Loading of training/testing ids and depths
train_df = pd.read_csv("../input/tgs-salt-identification-challenge/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../input/tgs-salt-identification-challenge/depths.csv", index_col="id")
# train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
# depths_df = pd.read_csv("../input/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

len(train_df)

train_df["images"] = [np.array(
    load_img("../input/tgs-salt-identification-challenge/train/images/{}.png".format(idx), grayscale=True)) / 255 for
                      idx in train_df.index]

train_df["masks"] = [np.array(
    load_img("../input/tgs-salt-identification-challenge/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx
                     in train_df.index]

# train_df["images"] = [np.array(load_img("../input/train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]

# train_df["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in train_df.index]

print('Data loaded finish !')


#### Reference  from Heng's discussion
# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/63984#382657
def get_mask_type(mask):
    border = 10
    outer = np.zeros((101 - 2 * border, 101 - 2 * border), np.float32)
    outer = cv2.copyMakeBorder(outer, border, border, border, border, borderType=cv2.BORDER_CONSTANT, value=1)

    cover = (mask > 0.5).sum()
    if cover < 8:
        return 0  # empty
    if cover == ((mask * outer) > 0.5).sum():
        return 1  # border
    if np.all(mask == mask[0]):
        return 2  # vertical

    percentage = cover / (101 * 101)
    if percentage < 0.15:
        return 3
    elif percentage < 0.25:
        return 4
    elif percentage < 0.50:
        return 5
    elif percentage < 0.75:
        return 6
    else:
        return 7


def histcoverage(coverage):
    histall = np.zeros((1, 8))
    for c in coverage:
        histall[0, c] += 1
    return histall


train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_target, 2)

train_df["coverage_class"] = train_df.masks.map(get_mask_type)

train_all = []
evaluate_all = []
skf = StratifiedKFold(n_splits=cv_total, random_state=1234, shuffle=True)
for train_index, evaluate_index in skf.split(train_df.index.values, train_df.coverage_class):
    train_all.append(train_index)
    evaluate_all.append(evaluate_index)
    print(train_index.shape, evaluate_index.shape)


def get_cv_data(cv_index):
    train_index = train_all[cv_index - 1]
    evaluate_index = evaluate_all[cv_index - 1]
    x_train = np.array(train_df.images[train_index].map(upsample).tolist()).reshape(-1, img_size_target,
                                                                                    img_size_target, 1)
    y_train = np.array(train_df.masks[train_index].map(upsample).tolist()).reshape(-1, img_size_target, img_size_target,
                                                                                   1)
    x_valid = np.array(train_df.images[evaluate_index].map(upsample).tolist()).reshape(-1, img_size_target,
                                                                                       img_size_target, 1)
    y_valid = np.array(train_df.masks[evaluate_index].map(upsample).tolist()).reshape(-1, img_size_target,
                                                                                      img_size_target, 1)
    return x_train, y_train, x_valid, y_valid


def convBn2d(input_tensor, filters, stage, block, kernel_size=(3, 3)):
    bn_axis = 3
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    conv_name_base = 'conv2d' + str(stage) + block + '_branch'

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base)(input_tensor)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base)(x)

    return x


def identity_block(input_tensor, filters, stage, block, kernel_size=(3, 3)):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2 = [filters] * 2

    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(input_tensor)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters1, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)
    # x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)

    x = layers.add([x, input_tensor])

    return x


def conv_block(input_tensor,
               filters,
               stage,
               block,
               strides=(2, 2),
               kernel_size=(3, 3)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = [filters] * 3
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters1, (2, 2), strides=strides,
                      padding='valid',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)
    # x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)

    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(input_tensor)
    shortcut = layers.Activation('relu')(shortcut)
    shortcut = layers.Conv2D(filters3, (2, 2), strides=strides,
                             padding='valid',
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])

    return x


def cse_block(prevlayer, prefix):
    mean = layers.Lambda(lambda xin: K.mean(xin, axis=[1, 2]))(prevlayer)
    # mean = layers.Dropout(0.1)(mean)
    lin1 = layers.Dense(K.int_shape(prevlayer)[3] // 2, name=prefix + 'cse_lin1', activation='relu')(mean)
    # lin1 = layers.Dropout(0.1)(lin1)
    lin2 = layers.Dense(K.int_shape(prevlayer)[3], name=prefix + 'cse_lin2', activation='sigmoid')(lin1)
    x = layers.Multiply()([prevlayer, lin2])
    return x


def sse_block(prevlayer, prefix):
    #     conv = layers.Conv2D(K.int_shape(prevlayer)[3], (1, 1), padding="same", kernel_initializer="he_normal", activation='sigmoid', strides=(1, 1),
    #                   name=prefix + "_conv")(prevlayer)

    conv = layers.Conv2D(1, (1, 1), padding="same", kernel_initializer="he_normal", activation='sigmoid',
                         strides=(1, 1),
                         name=prefix + "_conv")(prevlayer)
    conv = layers.Multiply(name=prefix + "_mul")([prevlayer, conv])
    return conv


def csse_block(x, prefix):
    '''
    Implementation of Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
    https://arxiv.org/abs/1803.02579
    '''
    cse = cse_block(x, prefix)
    sse = sse_block(x, prefix)
    x = layers.Add(name=prefix + "_csse_mul")([cse, sse])

    return x


def build_UResNet(input_layer, start_neurons):
    '''
    input_layer is designed to (128,128,3)
    '''
    # 128 -> 128
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(input_layer)

    conv1 = identity_block(conv1, filters=start_neurons * 1, stage=1, block='a')
    conv1 = identity_block(conv1, filters=start_neurons * 1, stage=1, block='b')
    conv1 = identity_block(conv1, filters=start_neurons * 1, stage=1, block='c')
    conv1 = csse_block(conv1, 'stage1')

    # 128 -> 64
    conv2 = conv_block(conv1, filters=start_neurons * 2, stage=2, block='a')
    conv2 = identity_block(conv2, filters=start_neurons * 2, stage=2, block='b')
    conv2 = identity_block(conv2, filters=start_neurons * 2, stage=2, block='c')
    conv2 = identity_block(conv2, filters=start_neurons * 2, stage=2, block='d')
    conv2 = csse_block(conv2, 'stage2')

    # 64 -> 32
    conv3 = conv_block(conv2, filters=start_neurons * 4, stage=3, block='a')
    conv3 = identity_block(conv3, filters=start_neurons * 4, stage=3, block='b')
    conv3 = identity_block(conv3, filters=start_neurons * 4, stage=3, block='c')
    conv3 = identity_block(conv3, filters=start_neurons * 4, stage=3, block='d')
    conv3 = identity_block(conv3, filters=start_neurons * 4, stage=3, block='e')
    conv3 = identity_block(conv3, filters=start_neurons * 4, stage=3, block='f')
    conv3 = csse_block(conv3, 'stage3')

    # 32 -> 16
    conv4 = conv_block(conv3, filters=start_neurons * 8, stage=4, block='a')
    conv4 = identity_block(conv4, filters=start_neurons * 8, stage=4, block='b')
    conv4 = identity_block(conv4, filters=start_neurons * 8, stage=4, block='c')
    conv4 = identity_block(conv4, filters=start_neurons * 8, stage=4, block='d')
    conv4 = csse_block(conv4, 'stage4')

    # 16 -> 8
    conv5 = conv_block(conv4, filters=start_neurons * 16, stage=-1, block='a')
    # dilation convolution
    # dilated_layer1 = Conv2D(start_neurons*8, (3,3), activation='relu', padding='same', dilation_rate=2)(conv5)
    # dilated_layer2 = Conv2D(start_neurons*8, (3,3), activation='relu', padding='same', dilation_rate=4)(dilated_layer1)
    # dilated_layer = layers.add([dilated_layer1, dilated_layer2])
    conv5 = convBn2d(conv5, start_neurons * 16, stage=-1, block='b')
    conv5 = convBn2d(conv5, start_neurons * 16, stage=-1, block='c')

    # 8 -> 16
    deconv5 = layers.UpSampling2D(size=(2, 2))(conv5)
    uconv5 = concatenate([deconv5, conv4])
    uconv5 = convBn2d(uconv5, start_neurons * 16, stage=0, block='a')
    uconv5 = convBn2d(uconv5, 64, stage=0, block='b')
    uconv5 = csse_block(uconv5, 'stage0')

    # 16 -> 32
    deconv4 = layers.UpSampling2D(size=(2, 2))(uconv5)
    uconv4 = concatenate([deconv4, conv3])
    uconv4 = convBn2d(uconv4, start_neurons * 8, stage=5, block='a')
    uconv4 = convBn2d(uconv4, 64, stage=5, block='b')
    uconv4 = csse_block(uconv4, 'stage5')

    # 32 -> 64
    # deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3 = layers.UpSampling2D(size=(2, 2))(uconv4)
    uconv3 = concatenate([deconv3, conv2])
    uconv3 = convBn2d(uconv3, start_neurons * 4, stage=6, block='a')
    uconv3 = convBn2d(uconv3, 64, stage=6, block='b')
    uconv3 = csse_block(uconv3, 'stage6')

    # 64 -> 128
    deconv2 = layers.UpSampling2D(size=(2, 2))(uconv3)
    uconv2 = concatenate([deconv2, conv1])
    uconv2 = convBn2d(uconv2, start_neurons * 2, stage=7, block='a')
    uconv2 = convBn2d(uconv2, 64, stage=7, block='b')
    uconv2 = csse_block(uconv2, 'stage7')

    # hypercolumn
    hypercolumn = concatenate([uconv2,
                               deconv2,
                               layers.UpSampling2D(size=(4, 4))(uconv4),
                               layers.UpSampling2D(size=(8, 8))(uconv5)])

    hypercolumn = layers.Dropout(0.4)(hypercolumn)
    logits = Conv2D(64, (3, 3), padding='same', activation='relu')(hypercolumn)

    # output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(logits)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    return output_layer


def get_iou_vector(A, B):
    A = np.squeeze(A)  # new added
    B = np.squeeze(B)  # new added
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch] > 0, B[batch] > 0
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
            metric.append(1)
            continue

        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0)) / (np.sum(union > 0))
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)


def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label > 0.5, pred > 0.5], tf.float64)


def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label > 0.5, pred > 0], tf.float64)


# code download from: https://github.com/bermanmaxim/LovaszSoftmax
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)

        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.math.add(tf.nn.elu(errors_sorted), tf.ones_like(errors_sorted)), tf.stop_gradient(grad),
                            1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels


def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1) > 0.5, 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    # logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred  # Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image=True, ignore=None)
    return loss


def build_model(start_neurones):
    h = 128
    w = 128
    input_layer = Input((h, w, 3))
    output_layer = build_UResNet(input_layer, start_neurones)

    model = Model(input_layer, output_layer)

    # c = optimizers.sgd(lr=lr, momentum=0.9, decay=0.0001)

    return model


def plot_history(history, metric_name):
    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15, 5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_loss.legend()
    ax_score.plot(history.epoch, history.history[metric_name], label="Train score")
    ax_score.plot(history.epoch, history.history["val_" + metric_name], label="Validation score")
    ax_score.legend()


def predict_result(model, x_test, img_size_target):  # predict both orginal and reflect x an example of TTA
    x_test_reflect = np.array([np.fliplr(x) for x in x_test])
    preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([np.fliplr(x) for x in preds_test2_refect])
    return preds_test / 2


def save_history(history, suffix):
    filename = 'history_' + suffix + '.csv'
    pd.DataFrame(history.history).to_csv(filename, index=False)


class SnapshotModelCheckpoint(Callback):
    """Callback that saves the snapshot weights of the model.
    Saves the model weights on certain epochs (which can be considered the
    snapshot of the model at that epoch).
    Should be used with the cosine annealing learning rate schedule to save
    the weight just before learning rate is sharply increased.
    # Arguments:
        nb_epochs: total number of epochs that the model will be trained for.
        nb_snapshots: number of times the weights of the model will be saved.
        fn_prefix: prefix for the filename of the weights.
    """

    def __init__(self, nb_epochs, nb_snapshots, fn_prefix='Model'):
        super(SnapshotModelCheckpoint, self).__init__()

        self.check = nb_epochs // nb_snapshots
        self.fn_prefix = fn_prefix

    def on_epoch_end(self, epoch, logs={}):
        if epoch != 0 and (epoch + 1) % self.check == 0:
            filepath = self.fn_prefix + '-%d.ckpt' % ((epoch + 1) // self.check)
            self.model.save_weights(filepath, overwrite=True)
            print("Saved snapshot at ../%s_%d.ckpt" % (self.fn_prefix, epoch))


class SnapshotCallbackBuilder:
    """Callback builder for snapshot ensemble training of a model.
    From the paper "Snapshot Ensembles: Train 1, Get M For Free" (https://openreview.net/pdf?id=BJYwwY9ll)
    Creates a list of callbacks, which are provided when training a model
    so as to save the model weights at certain epochs, and then sharply
    increase the learning rate.
    """

    def __init__(self, nb_epochs, nb_snapshots, init_lr):
        """
        Initialize a snapshot callback builder.
        # Arguments:
            nb_epochs: total number of epochs that the model will be trained for.
            nb_snapshots: number of times the weights of the model will be saved.
            init_lr: initial learning rate
        """
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr

    def get_callbacks(self, model_prefix='Model'):
        """
        Creates a list of callbacks that can be used during training to create a
        snapshot ensemble of the model.
        Args:
            model_prefix: prefix for the filename of the weights.
        Returns: list of 3 callbacks [ModelCheckpoint, LearningRateScheduler,
                 SnapshotModelCheckpoint] which can be provided to the 'fit' function
        """
        # if not os.path.exists('weights/'):
        #     os.makedirs('weights/')

        callback_list = [ModelCheckpoint('%s.ckpt' % model_prefix, monitor='val_my_iou_metric_2',
                                         mode='max', save_best_only=True, verbose=1),
                         LearningRateScheduler(schedule=self._cosine_anneal_schedule),
                         SnapshotModelCheckpoint(self.T, self.M, fn_prefix='%s' % model_prefix)]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)


def lr_scheduler(epoch):
    if epoch <= 10:
        lr = 0.01
    elif (epoch > 10) and (epoch <= 20):
        lr = 0.005
    else:
        lr = 0.001

    return lr


# training
cv_index = 2
basic_name = 'UResnet_cv'+str(cv_index+1)
print('############################################\n', basic_name)
save_model_name = basic_name + '.ckpt'

x_train, y_train, x_valid, y_valid = get_cv_data(cv_index + 1)
number_examples_for_train = x_train.shape[0]


# Data augmentation
# x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
# y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
def add_depth_channels(image_tensor):
    h, w = image_tensor.shape
    image3c = np.zeros((h, w, 3))
    image3c[:, :, 0] = image_tensor
    for row, const in enumerate(np.linspace(0, 1, h)):
        image3c[row, :, 1] = const
    image3c[:, :, 2] = image_tensor * image3c[:, :, 1]
    return image3c


def random_crop_resize(img, msk):
    limit = 0.1
    h, w = img.shape[:2]
    dl = int(h * limit)
    l = np.random.randint(int(0.8 * h), int(0.9 * h))

    y0 = np.random.randint(0, dl)
    y1 = y0 + l

    x0 = np.random.randint(0, dl)
    x1 = x0 + l

    img_crop = img[y0:y1, x0:x1]
    msk_crop = img[y0:y1, x0:x1]
    return cv2.resize(img_crop, (128, 128), interpolation=cv2.INTER_CUBIC), cv2.resize(msk_crop, (128, 128),
                                                                                       interpolation=cv2.INTER_CUBIC)


def single_img_aug(img, msk):
    img = np.squeeze(img, axis=-1)
    msk = np.squeeze(msk, axis=-1)
    if np.random.rand() <= 0.5:
        img = np.fliplr(img)
        msk = np.fliplr(msk)
        pass
    if np.random.rand() < 0.5:
        img, msk = random_crop_resize(img, msk)
        ## img = (128,128) msk=(128,128)
    else:
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
        msk = cv2.resize(msk, (128, 128), interpolation=cv2.INTER_CUBIC)

    ## add depth information
    img = add_depth_channels(img)
    # now img = (128,128,3)
    msk = np.expand_dims(msk, axis=-1)
    # now msk = (128,128,1)
    return img, msk


def do_augmentation(X_train, y_train):
    X_train_aug = []
    y_train_aug = []
    for index in range(X_train.shape[0]):
        img, msk = single_img_aug(X_train[index], y_train[index])
        X_train_aug.append(img)
        y_train_aug.append(msk)

    return np.array(X_train_aug), np.array(y_train_aug)


def generator(features, labels, batch_size):
    # create empty arrays to contain batch of features and labels
    batch_features = np.zeros((batch_size, 128, 128, 3))
    batch_labels = np.zeros((batch_size, 128, 128, 1))

    while True:
        # Fill arrays of batch size with augmented data taken randomly from full passed arrays
        indexes = random.sample(range(len(features)), batch_size)
        # Perform the exactly the same augmentation for X and y
        random_augmented_images, random_augmented_labels = do_augmentation(features[indexes], labels[indexes])
        batch_features[:, :, :, :] = random_augmented_images[:, :, :, :]
        batch_labels[:, :, :, :] = random_augmented_labels[:, :, :, :]

        yield batch_features, batch_labels

    # Resize valid data images to (128,128) and add depth information


X_valid = []
Y_valid = []
for index in range(x_valid.shape[0]):
    img = np.squeeze(x_valid[index])
    msk = np.squeeze(y_valid[index])
    msk = cv2.resize(msk, (128, 128), interpolation=cv2.INTER_CUBIC)
    Y_valid.append(np.expand_dims(msk, axis=-1))
    X_valid.append(add_depth_channels(cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)))

x_valid = np.array(X_valid)
y_valid = np.array(Y_valid)

## Build model and train
print('preparning ...')
model = build_model(start_neurones=32)
print('Bulid successfully model !')

# init_lr = 0.001
# #c = optimizers.sgd(lr=init_lr, momentum=0.9, decay=0.0001)
# c = optimizers.adam(lr=init_lr)
# model.compile(loss=binary_crossentropy, optimizer=c, metrics=[my_iou_metric])

# early_stopping = EarlyStopping(monitor='val_my_iou_metric', mode = 'max',patience=20, verbose=1)
# model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric',
#                               mode='max', save_best_only=True, verbose=1)
# reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode='max',
#                               factor=0.5, patience=5, min_lr=0.0000001, verbose=1)
# # scheduler = LearningRateScheduler(lr_scheduler)

# epochs = 110
# batch_size = 32
# steps_per_epoch = int(number_examples_for_train / batch_size)

# history = model.fit_generator(generator(x_train, y_train, batch_size),
#                               epochs=epochs,
#                               steps_per_epoch=steps_per_epoch,
#                               validation_data=[x_valid, y_valid],
#                               verbose = 2,
#                               callbacks =[reduce_lr, early_stopping, model_checkpoint])

# save_history(history, basic_name)

print('preparing...')
model.load_weights("../input/myuresnet3/UResnet_v0_cv3.ckpt")
print('load pretrained weights!')
input_x = model.layers[0].input

output_layer = model.layers[-1].input
model2 = Model(input_x, output_layer)
# model2.load_weights("../input/weights12345/my_resnet_v2_cv4.ckpt")
# print('load pretrained weights!')

init_lr = 0.0001
c = optimizers.adam(lr=init_lr)
# c = optimizers.sgd(lr=init_lr, momentum=0.9, decay=0.0001)
# lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation
# Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
model2.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])

# early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=16, verbose=1)
model_checkpoint = ModelCheckpoint(save_model_name, monitor='val_my_iou_metric_2',
                                   mode='max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode='max', factor=0.5, patience=8, min_lr=0.000000001,
                              verbose=1)

epochs = 75
batch_size = 32
# o = SnapshotCallbackBuilder(epochs, 1, init_lr)
# callback_list = o.get_callbacks(model_prefix=basic_name)

steps_per_epoch = int(number_examples_for_train / batch_size)

history = model2.fit_generator(generator(x_train, y_train, batch_size),
                               epochs=epochs,
                               steps_per_epoch=steps_per_epoch,
                               validation_data=[x_valid, y_valid],
                               verbose=2,
                               callbacks=[reduce_lr, model_checkpoint])

save_history(history, basic_name)

# model2.load_weights(save_model_name)

# thresholds_ori = np.linspace(0.2, 0.8, 100)
# thresholdslist = np.log(thresholds_ori/(1-thresholds_ori))
# thresholds = [0]*cv_total

# preds_valid = predict_result(model2 , x_valid, img_size_target)
# iouslist = np.array([get_iou_vector(y_valid, preds_valid > threshold) for threshold in thresholdslist])
# threshold_best_index = np.argmax(iouslist)
# iou_best = iouslist[threshold_best_index]
# threshold_best = thresholdslist[threshold_best_index]

# plt.plot(thresholdslist, iouslist)
# plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
# plt.xlabel("Threshold")
# plt.ylabel("IoU")
# plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
# plt.legend()
# plt.show()
# thresholds[cv_index] = threshold_best
# print(thresholds)
# print(threshold_best)
# print(iou_best)




