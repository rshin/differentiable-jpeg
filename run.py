import argparse
import collections
import csv
import functools
import glob
import io
import itertools
import os
import re
import sys
import time

from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import function, device as pydev
from tensorflow.python.client import device_lib
import PIL.Image
import numpy as np
import tensorflow as tf
import tqdm

sys.path.insert(0, 'slim')
import cleverhans.attacks
import cleverhans.model
from nets import nets_factory
from preprocessing import preprocessing_factory

import attacks
import jpeg
import utils


def get_available_gpus():
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']


def variables_on_cpu(var_device, other_device):
  var_ops = set(["Variable", "VariableV2", "VarHandleOp"])
  # Avoid https://github.com/tensorflow/tensorflow/issues/11484
  var_device = pydev.canonical_name(var_device)
  other_device = pydev.canonical_name(other_device)
  def device_function(op):
    node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
    if node_def.op in var_ops:
      return var_device
    # Keep existing device if one has already been specified.
    if op.device:
      return op.device
    if (node_def.op == 'Gather' and
        node_def.attr['Tparams'].type == tf.int32.as_datatype_enum):
      return var_device
    if (node_def.op == 'Const' and
        node_def.attr['dtype'].type == tf.string.as_datatype_enum):
      return var_device
    if node_def.op in ('Assert', 'ListDiff'):
      return var_device
    return other_device

  return device_function


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


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model-name')
  parser.add_argument('--checkpoint')
  parser.add_argument('--limit', type=int)
  parser.add_argument('--attacks', default='.*')
  parser.add_argument('--targets', default='.*')
  parser.add_argument('--models', default='.*')
  parser.add_argument('--batch-size', default=25, type=int)
  parser.add_argument('--output')
  args = parser.parse_args()

  tf.get_variable_scope()._reuse = tf.AUTO_REUSE

  synsets = {
      line.strip(): i
      for i, line in enumerate(
          open('imagenet-val/imagenet_lsvrc_2015_synsets.txt', 'r'))
  }
  normalization_fn, network_fn, image_size, offset = utils.create_model(
      args.model_name)
  num_classes = 1000 + offset
  image_size = network_fn.default_image_size

  logits_fn = make_network_fn(lambda image: network_fn(normalization_fn(image)),
                              [None, image_size, image_size,
                               3], [None, num_classes])
  pred_fn = lambda image: tf.argmax(logits_fn(image), axis=1)

  #
  # Define all graphs
  #
  image_ph = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
  label_ph = tf.placeholder(tf.int32, [None])
  label_onehot = tf.one_hot(label_ph, num_classes)

  defenses = collections.OrderedDict([
      ('none', lambda x: x),
      ('jpeg-25', functools.partial(utils.differentiable_jpeg, quality=25)),
      ('jpeg-50', functools.partial(utils.differentiable_jpeg, quality=50)),
      ('jpeg-75', functools.partial(utils.differentiable_jpeg, quality=75)),
  ])
  def_choice_ph = tf.placeholder(tf.int32, [])

  #gpus = get_available_gpus()
  #split_images = tf.split(image_ph, len(gpus))
  #logits_by_tower = []

  #for i, (dev, tower_images) in enumerate(zip(gpus, split_images)):
  #  with tf.device(variables_on_cpu('/cpu:0', dev)), \
  #      tf.name_scope('tower{}'.format(i)):

  # The default argument in the lambda is VERY imporatnt.
  # Otherwise all lambdas will end up with the same value for defense,
  # because the creation of the lambda doesn't capture the value of defense,
  # only its name.
  tests = [tf.equal(def_choice_ph, i) for i in range(len(defenses))]
  def_images = tf.case(
      [(test, lambda d=defense: d(image_ph))
       for test, defense in zip(tests, defenses.values())])
  logits = logits_fn(def_images)
  #logits_by_tower.append(logits)
  #logits = tf.concat(logits_by_tower, axis=0)
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=label_ph, logits=logits)

  grads, = tf.gradients(ys=[loss], xs=[image_ph])

  eval_defenses = collections.OrderedDict([
      ('none', lambda x: x),
      ('round', tf.round),
      ('jpeg-def-25', functools.partial(utils.jpeg_defense_tf, quality=25)),
      ('jpeg-def-50', functools.partial(utils.jpeg_defense, quality=50)),
      ('jpeg-def-75', functools.partial(utils.jpeg_defense_tf, quality=75)),
  ])
  eval_choice_ph = tf.placeholder(tf.int32, [])
  eval_images = tf.case(
      [(tf.equal(eval_choice_ph, i), lambda d=defense: d(image_ph))
       for i, (_, defense) in enumerate(eval_defenses.items())],
      exclusive=True)
  eval_preds = pred_fn(eval_images)

  attack_defs = collections.OrderedDict()
  attack_defs['none'] = lambda x, *args, **kwargs: x
  #attack_defs.update([('fgm-inf-{}'.format(eps), functools.partial(
  #    attacks.fgm, ord=np.inf, eps=eps, eps_iter=eps, nb_iter=1))
  #                    for eps in [1, 3, 5]])
  #attack_defs.update([('fgm-l2-{}'.format(eps), functools.partial(
  #    attacks.fgm, ord=2, eps=eps, eps_iter=eps, nb_iter=1))
  #                    for eps in [1, 3, 5]])
  #attack_defs.update([('iter-inf-{}-{}-{}'.format(eps, eps / 10., 10),
  #                     functools.partial(
  #                         attacks.fgm,
  #                         ord=np.inf,
  #                         eps=eps,
  #                         eps_iter=eps / 10.,
  #                         nb_iter=10)) for eps in [1, 3, 5]])
  attack_defs.update([('iter-l2-{}-{}-{}'.format(eps, eps_iter, nb_iter),
                       functools.partial(
                           attacks.fgm,
                           ord=2,
                           eps=eps,
                           eps_iter=eps_iter,
                           nb_iter=nb_iter))
                      for eps in [128., 256., 512.,]
                      for nb_iter in [10, 20]
                      for eps_iter in [eps / nb_iter * 2, eps / nb_iter]])
  #attack_defs.update([('iter-l2-{}'.format(eps), functools.partial(
  #    attacks.fgm, ord=2, eps=eps, eps_iter=eps / 5., nb_iter=10))
  #                    for eps in [1, 3, 5]])

  #ch_fgm = cleverhans.attacks.FastGradientMethod(
  #    cleverhans.model.CallableModelWrapper(logits_fn, 'logits'))
  #ch_fgm_out = ch_fgm.generate(
  #    image_ph, eps=1, ord=np.inf, y=label_onehot, clip_min=0, clip_max=255)

  #
  # End of defining graphs
  #
  model_vars = tf.contrib.framework.get_variables_to_restore()
  saver = tf.train.Saver(model_vars)

  print('Restoring parameters...')
  sess = tf.Session()
  saver.restore(sess, args.checkpoint)
  print('Done.')

  eval_fn = sess.make_callable(eval_preds, [image_ph, eval_choice_ph])
  #eval_fn = sess.make_callable(preds, [image_ph])
  grad_fn = sess.make_callable(grads, [def_choice_ph, image_ph, label_ph])
  grad_loss_fn = sess.make_callable([grads, loss],
                                    [def_choice_ph, image_ph, label_ph])
  #attack_defs['ch-fgm-inf-1'] = lambda images, labels, *args, **kwargs: \
  #    sess.run(ch_fgm_out, {image_ph: images, label_ph: labels})

  #
  # Load images
  #
  fns = sorted(
      glob.glob('imagenet-val/*/*.JPEG'), key=lambda fn: os.path.basename(fn))
  if args.limit:
    fns = fns[:args.limit]
  all_labels = [
      synsets[os.path.basename(os.path.dirname(fn))] + offset for fn in fns
  ]
  all_images = [utils.load_image(fn, image_size) for fn in tqdm.tqdm(fns)]

  if args.output is None:
    #output = 'results/attacks={},targets={},models={},{}.csv'.format(
    #    args.attacks, args.targets, args.models, int(time.time()))
    output = 'results/{}.csv'.format(int(time.time()))
  else:
    output = args.output

  metric_names = ['l2', 'norm_l2', 'linf']
  with open(output, 'w') as f:
    writer = csv.writer(f)
    header = ['attack', 'defense', 'model', 'correct', 'count']
    for metric in metric_names:
      header += [ '{} {}'.format(summary, metric) for summary in ('min', 'avg',
        'max')]
    writer.writerow(header)

    batch_size = args.batch_size
    for def_i, def_name in enumerate(defenses.keys()):
      grad_fn_partial = functools.partial(grad_fn, def_i)

      for attack_name, attack in attack_defs.items():
        if attack_name == 'none' and def_name != 'none':
          continue

        total = 0
        correct = collections.defaultdict(int)
        metrics = tuple([] for _ in metric_names)

        for labels, images in itertools.izip(
            utils.batch(all_labels, batch_size),
            utils.batch(tqdm.tqdm(all_images), batch_size)):

          images = np.array(images)
          attacked_images = attack(
              images, labels, grad_fn_partial, clip_min=0, clip_max=255)

          orig_l2_norm = attacks.batchwise_norm_np(images, 2) / 255
          diff_l2_norm = attacks.batchwise_norm_np(images - attacked_images,
                                                       2) / 255

          metrics[0].extend(diff_l2_norm.tolist())
          metrics[1].extend((diff_l2_norm / orig_l2_norm).tolist())
          metrics[2].extend((attacks.batchwise_norm_np(images - attacked_images,
                                                       np.inf) / 255).tolist())

          total += len(labels)
          for eval_i, (eval_name, defense) in enumerate(eval_defenses.items()):
            correct[eval_name] += sum(
                eval_fn(attacked_images, eval_i) == labels)

        metrics_summarized = [
            s(m)for m in metrics for s in (np.min, np.mean, np.max)
        ]
        for eval_name in eval_defenses.keys():
          row = ([attack_name, def_name, eval_name, correct[eval_name], total] +
                 metrics_summarized)
          print row
          writer.writerow(row)
          f.flush()
