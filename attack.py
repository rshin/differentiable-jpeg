import argparse
import glob
import io
import itertools
import os
import sys

from tensorflow.python.framework import function
import PIL.Image
import numpy as np
import tensorflow as tf
import tqdm

sys.path.insert(0, 'slim')
import cleverhans.attacks
import cleverhans.model
from nets import nets_factory
from preprocessing import preprocessing_factory

import jpeg

def make_network_fn(network_fn, input_shape, output_shape):

  #@function.Defun(tf.float32, shape_func=lambda _: [output_shape])
  def fn(images):
    images.set_shape(input_shape)
    logits, _ = network_fn(images)
    return logits

  return fn
  #return tf.make_template('network', fn)


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


def jpeg_defense(images, quality):
  def fn(images, quality):
    images = list(images.round().astype(np.uint8))
    new_images = []
    for image in images:
      image = PIL.Image.fromarray(image)
      buf = io.BytesIO()
      image.save(buf, 'jpeg', quality=int(quality))
      buf.seek(0)
      new_images.append(np.array(PIL.Image.open(buf), dtype=np.float32))
    return np.array(new_images)
  return tf.py_func(fn, [images, quality], [tf.float32], stateful=False)[0]


def differentiable_jpeg(image, quality):
  return jpeg.jpeg_compress_decompress(
      image, rounding=jpeg.diff_round, factor=jpeg.quality_to_factor(quality))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-name')
  parser.add_argument('--checkpoint')
  args = parser.parse_args()

  tf.get_variable_scope()._reuse = tf.AUTO_REUSE

  synsets = {
      line.strip(): i
      for i, line in enumerate(open('imagenet-val/imagenet_lsvrc_2015_synsets.txt', 'r'))
  }
  offset = {
      'inception_resnet_v2': 1,
  }[args.model_name]
  num_classes = 1000 + offset

  normalization_fn = normalization_fn_map[args.model_name]
  network_fn = nets_factory.get_network_fn(
      args.model_name, num_classes=num_classes, is_training=False)
  image_size = network_fn.default_image_size

  logits_fn = make_network_fn(lambda image: network_fn(normalization_fn(image)),
      [None, image_size, image_size, 3], [None, num_classes])
  pred_fn = lambda image: tf.argmax(logits_fn(image), axis=1)

  ch_model = cleverhans.model.CallableModelWrapper(logits_fn, 'logits')
  ch_jpeg_model = cleverhans.model.CallableModelWrapper(
      lambda image: logits_fn(differentiable_jpeg(image, 25)), 'logits')

  image_ph = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
  y_pred = {
      'orig': pred_fn(image_ph),
      'orig-25': pred_fn(jpeg_defense(image_ph, 25)),
#      'orig-50': pred_fn(jpeg_defense(image_ph, 50)),
#      'orig-75': pred_fn(jpeg_defense(image_ph, 75)),
  }

  model_vars = tf.contrib.framework.get_variables_to_restore()
  saver = tf.train.Saver(model_vars)

  sess = tf.InteractiveSession()
  print('Restoring parameters...')
  saver.restore(sess, args.checkpoint)
  print('Done.')

  attack = cleverhans.attacks.FastGradientMethod(ch_model)
  x_attack = attack.generate(image_ph, clip_min=0, clip_max=255, eps=3)
  y_pred.update({
    'attack': pred_fn(x_attack),
    'attack-25': pred_fn(jpeg_defense(x_attack, 25)),
#    'attack-50': pred_fn(jpeg_defense(x_attack, 50)),
#    'attack-75': pred_fn(jpeg_defense(x_attack, 75)),
  })

  jpeg_attack = cleverhans.attacks.FastGradientMethod(ch_jpeg_model)
  x_jpeg_attack = jpeg_attack.generate(image_ph, clip_min=0, clip_max=255, eps=3)
  y_pred.update({
    'jpeg-attack': pred_fn(x_jpeg_attack),
    'jpeg-attack-25': pred_fn(jpeg_defense(x_jpeg_attack, 25)),
  })


  count = 0
  correct = {k: 0 for k in y_pred}

  fns = glob.glob('imagenet-val/*/*.JPEG')
  all_labels = [synsets[os.path.basename(os.path.dirname(fn))] + offset for fn in fns]
  all_images = [load_image(fn, image_size) for fn in tqdm.tqdm(fns)]

  batch_size = 32
  for labels, images in itertools.izip(
      batch(all_labels, batch_size), batch(tqdm.tqdm(all_images), batch_size)):
    #y = sess.run(y_pred, {image_ph: images})
    for name, pred in y_pred.iteritems():
      y = sess.run(pred, {image_ph: images})
      correct[name] += sum(y == labels)
    count += len(images)

  print correct
  #print 'Accuracy: {}'.format(correct / float(count))
  #print 'Attacked accuracy: {}'.format(attack_correct / float(count))
