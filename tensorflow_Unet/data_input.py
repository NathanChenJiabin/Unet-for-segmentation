# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Attention: this file is modified directly in base of Tensorflow official code.
"""Routine for decoding the binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os
#
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np


IMAGE_SIZE = 256

# Global constants describing the data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 26153
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 8609


def read_data(filename_queue):
    """Reads and parses examples from palmier data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (256)
      width: number of columns in the result (256)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: a [height, width, depth] float tensor represent mask
      image: a [height, width, depth] float Tensor with the image data
  """

    class PALMIERRecord(object):
        pass

    result = PALMIERRecord()
    # input format.
    result.height = IMAGE_SIZE
    result.width = IMAGE_SIZE
    result.depth = 3

    reader = tf.TFRecordReader()
    result.key, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([], tf.string)
                                       })

    result.image = tf.reshape(tf.decode_raw(features['image'], tf.float32), [IMAGE_SIZE, IMAGE_SIZE, 3])
    result.label = tf.reshape(tf.decode_raw(features['label'], tf.float32), [IMAGE_SIZE, IMAGE_SIZE, 1])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 3-D Tensor of [height, width, 1] of type.floar32.
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 4D tensor of [batch_size, height, width, 1] size.
  """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 4
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)
    tf.summary.image('masks', label_batch)

    return images, label_batch


def distort_color(image, random_proc=0):
    if random_proc == 0:
        return image
    elif random_proc == 4:
        return tf.image.random_brightness(image, 0.5)
    elif random_proc == 2:
        return tf.image.random_contrast(image, 0.0, 1.0)
    elif random_proc == 3:
        return tf.image.random_hue(image, 0, 0.2)
    elif random_proc == 1:
        return tf.image.random_saturation(image, 0, 5.0)


def distorted_inputs(file_list, batch_size):
    """Construct distorted input for palmiers training using the Reader ops.

  Args:
    file_list: List path to the palmiers data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
  """
    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(file_list)

    with tf.name_scope('read_data'):
        # Read examples from files in the filename queue.
        read_input = read_data(filename_queue)
        # Random ajust color 
        distorted_img = distort_color(read_input.image, np.random.randint(5))
        # Random rotation Attention: must rotate image and mask together
        alea = np.random.randint(2)
        if alea == 0:
            alea2 = np.random.randint(3)+1
            img = tf.image.rot90(distorted_img, k=alea2)
            mask = tf.image.rot90(read_input.label, k=alea2)
        else:
            img = distorted_img
            mask = read_input.label
        # Random flip right and left or up and down Attention: must flip image and mask together
        alea = np.random.randint(3)
        if alea == 0:
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)
        if alea == 1:
            img = tf.image.flip_up_down(img)
            mask = tf.image.flip_up_down(mask)
        
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                 min_fraction_of_examples_in_queue)
        print('Filling queue with %d palmiers images before starting to train. '
              'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(img, mask,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def inputs(file_list, batch_size):
    """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    file_list: List path to the palmiers test data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
  """


    with tf.name_scope('input_testdata'):
        # Create a queue that produces the filenames to read.
        filename_queue_test = tf.train.string_input_producer(file_list)

        # Read examples from files in the filename queue.
        read_input_test = read_data(filename_queue_test)

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples_test = int(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL *
                                 min_fraction_of_examples_in_queue)
        print('Filling queue with %d palmiers images before starting to evaluation. '
              'This will take a few minutes.' % min_queue_examples_test)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(read_input_test.image, read_input_test.label,
                                           min_queue_examples_test, batch_size,
                                           shuffle=False)
