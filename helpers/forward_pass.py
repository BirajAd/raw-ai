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


def batchnorm_forward(x, gamma, beta, bn_param):
  """Forward pass for batch normalization.

  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the
  mean and variance of each feature, and these averages are used to normalize
  data at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7
  implementation of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param["mode"]
  eps = bn_param.get("eps", 1e-5)
  momentum = bn_param.get("momentum", 0.1)

  N, D = x.shape
  running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get("running_var", np.ones(D, dtype=x.dtype))

  out, cache = None, None
  sample_mean = x.mean(axis=0)
  sample_var = x.var(axis=0)

  if mode == "train":
    #######################################################################
    # TODO: Implement the training-time forward pass for batch norm.      #
    # Use minibatch statistics to compute the mean and variance, use      #
    # these statistics to normalize the incoming data, and scale and      #
    # shift the normalized data using gamma and beta.                     #
    #                                                                     #
    # You should store the output in the variable out. Any intermediates  #
    # that you need for the backward pass should be stored in the cache   #
    # variable.                                                           #
    #                                                                     #
    # You should also use your computed sample mean and variance together #
    # with the momentum variable to update the running mean and running   #
    # variance, storing your result in the running_mean and running_var   #
    # variables.                                                          #
    #                                                                     #
    # Note that though you should be keeping track of the running         #
    # variance, you should normalize the data based on the standard       #
    # deviation (square root of variance) instead!                        #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
    # might prove to be helpful.                                          #
    #######################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    inv_var = 1 / np.sqrt(sample_var + eps)
    normalized = (x - sample_mean) * inv_var  # call it x_hat
    out = gamma * normalized + beta
    cache = (x, gamma, beta, normalized, sample_mean, sample_var, eps, inv_var)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################
  elif mode == "test":
    #######################################################################
    # TODO: Implement the test-time forward pass for batch normalization. #
    # Use the running mean and variance to normalize the incoming data,   #
    # then scale and shift the normalized data using gamma and beta.      #
    # Store the result in the out variable.                               #
    #######################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    normalized = (x - running_mean) / np.sqrt(running_var + eps)
    out = gamma * normalized + beta

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #######################################################################
    #                          END OF YOUR CODE                           #
    #######################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param["running_mean"] = running_mean
  bn_param["running_var"] = running_var

  return out, cache
