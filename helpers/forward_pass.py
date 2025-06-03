import numpy as np

def affine_forward(x, w, b):
  """
    It computes the w*x + b

    Input:
      - x: Input matrix of size: (N, D1, D2, ...., Dk)
      - w: weight matrix of size: (D, H)
      - b: matrix of size (H,)
    
    Output:
      - out: result of w*x + b, size of (N, H)
      - cache: cache needed for backward pass of affine_forward
  """
  N = x.shape[0]
  out = x.reshape(N, -1) @ w + b
  cache = (x, w, b)
  return out, cache

def relu_forward(x):
  """
  Relu function:

    { x if x > 0 }
    { 0 otherwise }

  Input:
    - x: input matrix
  """
  out = np.where(x > 0, x, 0)
  cache = x
  return out, cache

def softmax_loss(scores, y=None):
  """
  given the scores and labels, it calculates the softmax loss and returns the gradient

  Input:
    - scores: scores for each category
    - y: labels

  Output:
    - loss: softmax loss
    - dlogits: gradients of loss w.r.t logits
  """
  # to avoid overflow
  N, num_classes = scores.shape
  scores -= scores.max(axis=1, keepdims=True)
  scores = np.exp(scores)
  exp_sum = scores.sum(axis=1, keepdims=True)
  softmax = scores / exp_sum

  if y is None:
    return softmax

  # loss = -np.log(e_fyi / exp_sum)
  correct_category_scores = softmax[np.arange(N), y]
  logs = np.log(correct_category_scores)
  loss = -np.sum(logs)
  # average loss
  loss /= N

  # backprop w.r.t logits
  mask = np.zeros_like(softmax)
  mask[np.arange(N), y] = -1
  dlogits = softmax + mask

  return loss, dlogits

def word_embedding_forward(x, W):
  """Forward pass for word embeddings.

  We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  word to a vector of dimension D.

  Inputs:
  - x: Integer array of shape (N, T) giving indices of words at each timestamp
  - W: Weight matrix of shape (V, D) giving some weight(embedding) for the vocabulary

  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  out = W[x]

  cache = (x, W)
  
  return out, cache

def tanh_forward(x):
  """
  Calculates tanh of x

  Input:
    x: 

  Output:
    out: output of tanh(x)
    cache: output basically
  """
  out = np.tanh(x)

  cache = (out)

  return out, cache
