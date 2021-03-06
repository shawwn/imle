# coding=utf-8
# Copyright 2020 The Mesh TensorFlow Authors.
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

#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""tf.data.Dataset interface to the MNIST dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import shutil
import tempfile

import numpy as np
from six.moves import urllib
import tensorflow.compat.v1 as tf


def read32(bytestream):
  """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
  """Validate that filename corresponds to images for the MNIST dataset."""
  with tf.gfile.Open(filename, 'rb') as f:
    magic = read32(f)
    read32(f)  # num_images, unused
    rows = read32(f)
    cols = read32(f)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                     f.name))
    if rows != 28 or cols != 28:
      raise ValueError(
          'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
          (f.name, rows, cols))


def check_labels_file_header(filename):
  """Validate that filename corresponds to labels for the MNIST dataset."""
  with tf.gfile.Open(filename, 'rb') as f:
    magic = read32(f)
    read32(f)  # num_items, unused
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                     f.name))


def download(directory, filename):
  """Download (and unzip) a file from the MNIST dataset if not already done."""
  filepath = os.path.join(directory, filename)
  if tf.gfile.Exists(filepath):
    return filepath
  if not tf.gfile.Exists(directory):
    tf.gfile.MakeDirs(directory)
  url = 'http://yann.lecun.com/exdb/mnist/' + filename + '.gz'
  _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
  print('Downloading %s to %s' % (url, zipped_filepath))
  urllib.request.urlretrieve(url, zipped_filepath)
  with gzip.open(zipped_filepath, 'rb') as f_in, \
      tf.gfile.Open(filepath, 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)
  os.remove(zipped_filepath)
  return filepath


def dataset(directory, images_file, labels_file, resolution, channels):
  """Download and parse MNIST dataset."""

  if not isinstance(resolution, list):
    resolution = [resolution, resolution]

  images_file = download(directory, images_file)
  labels_file = download(directory, labels_file)

  check_image_file_header(images_file)
  check_labels_file_header(labels_file)

  def decode_image(image):
    # Normalize from [0, 255] to [0.0, 1.0]
    image = tf.decode_raw(image, tf.uint8)
    image = tf.reshape(image, [1, 1, 28, 28])
    image = tf.transpose(image, [0,2,3,1])
    #import pdb; pdb.set_trace()
    image = tf.image.resize_area(image, resolution)
    #image = tf.image.resize_bilinear(image, resolution)
    image = tf.concat((image,)*channels, axis=-1) # convert to RGB
    #import pdb; pdb.set_trace()
    image = tf.transpose(image, [0,3,1,2])
    image = tf.cast(image, tf.float32)
    #image = tf.reshape(image, [channels*np.prod(resolution)])
    return image / 255.0

  def decode_label(label):
    label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
    label = tf.reshape(label, [])  # label is a scalar
    return tf.to_int32(label)

  images = tf.data.FixedLengthRecordDataset(
      images_file, 28 * 28, header_bytes=16).map(decode_image)
  labels = tf.data.FixedLengthRecordDataset(
      labels_file, 1, header_bytes=8).map(decode_label)
  return tf.data.Dataset.zip((images, labels))


def train(directory, resolution=28, channels=1):
  """tf.data.Dataset object for MNIST training data."""
  return dataset(directory, 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte', resolution=resolution, channels=channels)


def test(directory, resolution=28, channels=1):
  """tf.data.Dataset object for MNIST test data."""
  return dataset(directory, 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte', resolution=resolution, channels=channels)

def get_mnist(n, only_labels=None, resolution=28, channels=1):
  if only_labels is not None and not isinstance(only_labels, list):
    only_labels = [only_labels]
  with tf.Session() as sess:
    t = train('MNIST', resolution=resolution, channels=channels)
    it = t.make_initializable_iterator()
    sess.run(it.initializer)
    nxt = it.get_next()
    for i in range(n):
      while True:
        image, label = sess.run(nxt)
        if not only_labels or label in only_labels:
          yield image, label
          break

def get_mnist_images(n, **kws):
  resolution = 28
  if 'resolution' in kws:
    resolution = kws.pop('resolution')
  resolution = [resolution, resolution]
  channels = 1
  if 'channels' in kws:
    channels = kws.pop('channels')
  return np.array([x[0] for x in get_mnist(n, resolution=resolution, channels=channels, **kws)]).reshape([-1,channels] + resolution)

def save_mnist_image(sample, outfile):
  with tf.Session() as sess:
    image_out = sess.run(tf.io.encode_jpeg(sample))
    with open(outfile, 'wb') as f:
      f.write(image_out)

