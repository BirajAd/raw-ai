import numpy as np

def affine_backward(dout, cache):
  """
  Backward propagation for affine forward
    
  Input:
    - dout: (N, D)
    - cache

  Output:
    - dx: gradient w.r.t x
    - dw: gradient w.r.t w
    - db: gradient w.r.t b
  """
  x, w, b = cache

  N = x.shape[0]
  x = x.reshape(N, -1)

  dx = dout @ w.T
  dw = x.T @ dout
  db = dout.sum(axis=0)

  return dx, dw, db


def relu_backward(dout, cache):
  """
  Backward pass for relu

  Input:
    - dout: downward flowing gradient
    - cache

  Output:
    - dx: gradient w.r.t x
  """
  x = cache
  return dout * np.where(x > 0, 1, 0)

def word_embedding_backward(dout, cache):
  """Backward pass for word embeddings.

    We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    we will use np.add.at for this exercise

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D)
  """
  x, W = cache

  dW = np.zeros_like(W)

  # update dW at indices from x, with values from dout
  np.add.at(dW, x, dout)

  return dW

def tanh_backward(dout, cache):
  out = cache

  return dout * (1 - np.square(out))


def batchnorm_backward(dout, cache):
  """Alternative backward pass for batch normalization.

  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  See the jupyter notebook for more hints.

  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.

  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  ###########################################################################
  # TODO: Implement the backward pass for batch normalization. Store the    #
  # results in the dx, dgamma, and dbeta variables.                         #
  #                                                                         #
  # After computing the gradient with respect to the centered inputs, you   #
  # should be able to compute gradients with respect to the inputs in a     #
  # single statement; our implementation fits on a single 80-character line.#
  ###########################################################################
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

  x, gamma, beta, x_hat, sample_mean, sample_var, eps, inv_var = cache

  dgamma = np.sum(dout * x_hat, axis=0)
  dbeta = np.sum(dout, axis=0)

  m, D = dout.shape
  dl_xhat = dout * gamma
  dx = (
    (1 / m)
    * inv_var
    * (m * dl_xhat - np.sum(dl_xhat, axis=0) - x_hat * np.sum(dl_xhat * x_hat, axis=0))
  )

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  ###########################################################################
  #                             END OF YOUR CODE                            #
  ###########################################################################

  return dx, dgamma, dbeta