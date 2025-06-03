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
