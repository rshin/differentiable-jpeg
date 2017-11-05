import argparse
import collections
import csv
import glob
import io
import itertools
import os
import re
import sys
import time

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

  def fn(images):
    images.set_shape(input_shape)
    logits, _ = network_fn(images)
    return logits

  return fn


'''
def make_network_fn(network_fn, input_shape, output_shape):

  @function.Defun(tf.float32, tf.float32, shape_func=lambda _: [input_shape])
  def fn_grad(images, grad_logits):
    images.set_shape(input_shape)
    grad_logits.set_shape(output_shape)
    #with tf.device('/cpu:0'):
    #  grad_logits = tf.identity(grad_logits)
    #  images = tf.Print(images, [tf.shape(images)], 'images shape: ',
    #      summarize=9999)
    #  grad_logits = tf.Print(grad_logits, [tf.shape(grad_logits)], 'grad_logits shape: ', summarize=9999)
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      logits, _  = network_fn(images)
    return tf.gradients(ys=[logits], xs=[images], grad_ys=[grad_logits])

  # Can't use fn_grad directly because of a bug:
  # fn has images + as many inputs as the number of variables used inside.
  # fn_grad gets images + variables + grad_logits as input.
  # So the grad_logits argument is set to the first variable.
  def fn_py_grad(op, grad_logits):
    return [fn_grad(op.inputs[0], grad_logits)] + [tf.zeros_like(inp) for inp in
      op.inputs[1:]]

  @function.Defun(tf.float32, python_grad_func=fn_py_grad, shape_func=lambda _: [output_shape])
  #@function.Defun(tf.float32, shape_func=lambda _: [output_shape])
  def fn(images):
    images.set_shape(input_shape)
    logits, _ = network_fn(images)
    #logits = tf.Print(logits, [tf.shape(logits)], 'logits shape: ',
    #    summarize=9999)
    return logits

  return fn
'''


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


def jpeg_defense(images, quality):
  return tf.py_func(fn, [images, quality], [tf.float32], stateful=False)[0]


def differentiable_jpeg(image, quality):
  return jpeg.jpeg_compress_decompress(
      image, rounding=jpeg.diff_round, factor=jpeg.quality_to_factor(quality))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-name')
  parser.add_argument('--checkpoint')
  parser.add_argument('--attacks', default='.*')
  parser.add_argument('--targets', default='.*')
  parser.add_argument('--models', default='.*')
  parser.add_argument('--batch-size', default=32, type=int)
  parser.add_argument('--output')
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

  #ch_model = cleverhans.model.CallableModelWrapper(logits_fn, 'logits')
  #ch_jpeg_model = cleverhans.model.CallableModelWrapper(
  #    lambda image: logits_fn(differentiable_jpeg(image, 25)), 'logits')

  ch_models = collections.OrderedDict([
      ('orig', cleverhans.model.CallableModelWrapper(logits_fn, 'logits')),
      ('jpeg-25', cleverhans.model.CallableModelWrapper(
        lambda x: logits_fn(differentiable_jpeg(x, 25)), 'logits')),
  #    'jpeg-50': cleverhans.model.CallableModelWrapper(
  #      lambda x: logits_fn(differentiable_jpeg(x, 50)), 'logits'),
      ('jpeg-75', cleverhans.model.CallableModelWrapper(
        lambda x: logits_fn(differentiable_jpeg(x, 75)), 'logits')),
  ])
  models = collections.OrderedDict([
      ('orig', pred_fn),
      ('def-25', lambda x: pred_fn(jpeg_defense(x, 25))),
      ('def-75', lambda x: pred_fn(jpeg_defense(x, 75))),
  ])

  image_ph = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
  label_ph = tf.placeholder(tf.int32, [None])
  label_onehot = tf.one_hot(label_ph, num_classes)
  combos = collections.OrderedDict()
  for model_name, model in models.iteritems():
    combos[('none', 'none', model_name)] = model(image_ph)

  model_vars = tf.contrib.framework.get_variables_to_restore()
  saver = tf.train.Saver(model_vars)

  sess = tf.InteractiveSession()
  print('Restoring parameters...')
  saver.restore(sess, args.checkpoint)
  print('Done.')

  attack_methods = {
      'fgm': cleverhans.attacks.FastGradientMethod,
      'iterative': cleverhans.attacks.BasicIterativeMethod,
      'deepfool': cleverhans.attacks.DeepFool,
  }

  attacks = collections.OrderedDict()
  attacks.update([('fgm-inf-{}'.format(eps), (attack_methods['fgm'], {
      'eps': eps,
      'ord': np.inf
  })) for eps in [1, 3, 5]])
  #attacks.update([('fgm-l2-{}'.format(eps), (attack_methods['fgm'], {
  #    'eps': eps,
  #    'ord': 2
  #})) for eps in [1, 3, 5]])
  attacks.update([('iter-inf-{}'.format(eps), (attack_methods['iterative'], {
      'eps': eps,
      'eps_iter': eps / 6.,
      'ord': np.inf
  })) for eps in [1, 3, 5]])

  now = time.time()
  for attack_name, (method_class, params) in attacks.items():
    if re.search(args.attacks, attack_name) is None:
      continue

    for target_name, ch_model in ch_models.items():
      if re.search(args.targets, target_name) is None:
        continue

      method = method_class(ch_model)
      for model_name, model in models.items():
        if re.search(args.models, model_name) is None:
          continue

        identifier = (attack_name, target_name, model_name)
        combos[identifier] = model(
            method.generate(image_ph, y=label_onehot, clip_min=0, clip_max=255, **params))
        print identifier
  print('{} seconds.'.format(time.time() - now))

  fns = glob.glob('imagenet-val/*/*.JPEG')
  all_labels = [synsets[os.path.basename(os.path.dirname(fn))] + offset for fn in fns]
  all_images = [load_image(fn, image_size) for fn in tqdm.tqdm(fns)]

  if args.output is None:
    output = 'results/attacks={},targets={},models={},{}.csv'.format(
        args.attacks, args.targets, args.models, int(time.time()))
  else:
    output = args.output

  with open(output, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['attack', 'target', 'model', 'correct', 'count'])
    batch_size = args.batch_size
    for name, pred in combos.items():
      correct, count = 0, 0
      for labels, images in itertools.izip(
          batch(all_labels, batch_size),
          batch(tqdm.tqdm(all_images, desc=str(name)), batch_size)):
        y = sess.run(pred, {image_ph: images, label_ph: labels})
        correct += sum(y == labels)
        count += len(labels)
      result = list(name) + [str(correct), str(count)]
      print result
      writer.writerow(result)
      f.flush()

  #print 'Accuracy: {}'.format(correct / float(count))
  #print 'Attacked accuracy: {}'.format(attack_correct / float(count))
