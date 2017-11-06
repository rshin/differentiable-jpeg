import numpy as np

def batchwise_norm_np(x, ord):
  dims = tuple(range(1, len(x.shape)))
  if ord == np.inf:
    return np.amax(np.abs(x), axis=dims, keepdims=True)
  elif ord == 1:
    return np.sum(np.abs(x), axis=dims, keepdims=True)
  elif ord == 2:
    return np.sqrt(np.sum(np.square(x), axis=dims, keepdims=True))
  else:
    raise ValueError(ord)


def fgm(images, labels, backward,
    ord, eps, eps_iter, nb_iter, clip_min, clip_max):
  orig_images = images
  images = np.copy(orig_images)

  for i in range(nb_iter):
    grad_images = backward(images, labels)

    if ord == np.inf:
      grad_images = np.sign(grad_images)
    elif ord in (1, 2):
      grad_images /= batchwise_norm_np(grad_images, ord)

    # Take eps_iter step
    grad_images *= eps_iter
    images += grad_images

    # Ensure |images - orig_images| < eps
    factor = np.maximum(batchwise_norm_np(images - orig_images, ord) / eps, 1)
    images /= factor
    images += orig_images * (1 - 1./factor)

    np.clip(images, clip_min, clip_max, out=images)

  return images
