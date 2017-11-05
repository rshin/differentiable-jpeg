import io
import sys

import numpy as np
import tensorflow as tf
import PIL.Image

sys.path.insert(0, 'slim')
from nets import nets_factory
from preprocessing import preprocessing_factory

import jpeg


def vgg_normalization(image):
  return image - [123.68, 116.78, 103.94]


def inception_normalization(image):
  return ((image / 255.) - 0.5) * 2


normalization_fn_map = {
    'inception': inception_normalization,
    'inception_v1': inception_normalization,
    'inception_v2': inception_normalization,
    'inception_v3': inception_normalization,
    'inception_v4': inception_normalization,
    'inception_resnet_v2': inception_normalization,
    'mobilenet_v1': inception_normalization,
    'nasnet_mobile': inception_normalization,
    'nasnet_large': inception_normalization,
    'resnet_v1_50': vgg_normalization,
    'resnet_v1_101': vgg_normalization,
    'resnet_v1_152': vgg_normalization,
    'resnet_v1_200': vgg_normalization,
    'resnet_v2_50': vgg_normalization,
    'resnet_v2_101': vgg_normalization,
    'resnet_v2_152': vgg_normalization,
    'resnet_v2_200': vgg_normalization,
    'vgg': vgg_normalization,
    'vgg_a': vgg_normalization,
    'vgg_16': vgg_normalization,
    'vgg_19': vgg_normalization,
}


def batch(iterable, size):
  iterator = iter(iterable)
  batch = []
  while True:
    try:
      batch.append(next(iterator))
    except StopIteration:
      if batch:
        yield batch
      return

    if len(batch) == size:
      yield batch
      batch = []


def load_image(fn, image_size):
  # Resize the image appropriately first
  image = PIL.Image.open(fn)
  image = image.convert('RGB')
  image = image.resize((image_size, image_size), PIL.Image.BILINEAR)
  image = np.array(image, dtype=np.float32)
  return image


def jpeg_defense_numpy(images, quality):
  images = list(images.round().astype(np.uint8))
  new_images = []
  for image in images:
    image = PIL.Image.fromarray(image)
    buf = io.BytesIO()
    image.save(buf, 'jpeg', quality=int(quality))
    buf.seek(0)
    new_images.append(np.array(PIL.Image.open(buf), dtype=np.float32))
  return np.array(new_images)


def jpeg_defense(images, quality):
  return tf.py_func(
      jpeg_defense_numpy, [images, quality], [tf.float32], stateful=False)[0]


def jpeg_defense_tf(images, quality):
  with tf.device('/cpu:0'):
    result = tf.map_fn(
        lambda image: tf.image.decode_jpeg(
          tf.image.encode_jpeg(image, quality=quality)),
        tf.cast(tf.round(images), tf.uint8),
        parallel_iterations=64,
        back_prop=False)

    result = tf.cast(result, tf.float32)
    result.set_shape(images.shape.as_list())
    return result


def differentiable_jpeg(image, quality):
  return jpeg.jpeg_compress_decompress(
      image, rounding=jpeg.diff_round, factor=jpeg.quality_to_factor(quality))


def create_model(name):
  offset = {
      'inception_resnet_v2': 1,
  }[name]
  num_classes = 1000 + offset

  normalization_fn = normalization_fn_map[name]
  network_fn = nets_factory.get_network_fn(
      name, num_classes=num_classes, is_training=False)
  image_size = network_fn.default_image_size

  return normalization_fn, network_fn, image_size, offset
