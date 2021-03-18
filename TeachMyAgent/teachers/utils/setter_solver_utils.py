"""
Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


This file contains tensorflow building blocks for
`Automated curricula through setter-solver interactions`, S. Racaniere & A. K. Lampinen, ICLR 2020.

This code is provided to help with reproducibility of the results of the paper.
"""
# Modified by Clément Romac, copy of the license at TeachMyAgent/teachers/LICENSES/Setter-Solver

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import sonnet as snt
import tensorflow as tf


_MIN_SCALE = 1e-2


def _highway_block(inputs, hidden_size=32):
  """A simple highway block with a single gate.

  The block was previously used in Zilly et al, "Recurrent Highway Networks":
  https://arxiv.org/abs/1607.03474

  Args:
    inputs: An input `Tensor` with shape `[batch_size, hidden_size]`.
    hidden_size: The size on the residual connections.

  Returns:
    A `Tensor` with the same shape as the input.
  """
  inputs.get_shape().assert_has_rank(2)

  to_gates = snt.Linear(2 * hidden_size)(inputs)
  to_tanh, to_gate = tf.split(to_gates, 2, axis=-1)
  gate = tf.sigmoid(to_gate)
  return inputs + (tf.tanh(to_tanh) - inputs) * gate


class Highway(snt.AbstractModule):
  """A linear network with multiple highway blocks.

  This architecture was introduced in Zilly et al, "Recurrent Highway Networks":
  https://arxiv.org/abs/1607.03474
  """

  def __init__(self,
               output_size,
               num_blocks=4,
               hidden_size=32,
               zero_init_last_layer=False,
               name='highway'):
    """Constructs the network.

    Args:
      output_size: The wanted number of outputs.
      num_blocks: The number of highway blocks.
      hidden_size: The size on the residual connections.
      zero_init_last_layer: Whether to initialize the last convolution with 0's.
        This makes the transform an identity op in a RVNP-like architecture.
        See GLOW (https://arxiv.org/abs/1807.03039) for more details.
      name: Name of the module.
    """
    super(Highway, self).__init__(name=name)
    self._output_size = output_size
    self._num_blocks = num_blocks
    self._hidden_size = hidden_size
    self._zero_init_last_layer = zero_init_last_layer

  def _build(self, inputs):
    """Runs the network on the given input.

    Args:
      inputs: A `Tensor` with shape `[batch_size, hidden_size]`.

    Returns:
      A `Tensor` with shape `[batch_size, output_size]`.
    """

    def _block(y):
      return _highway_block(y, hidden_size=self._hidden_size)

    y = snt.Linear(self._hidden_size)(inputs)
    # Starting with tanh.
    y = tf.tanh(y)
    for _ in range(self._num_blocks):
      y = _block(y)

    if self._zero_init_last_layer:
      final_layer_initializers = {'w': tf.zeros_initializer()}
    else:
      final_layer_initializers = None

    y = snt.Linear(self._output_size, initializers=final_layer_initializers)(y)
    return y


def non_zero_uniform(shape):
  """Samples in open range (0, 1).

  This avoids the value 0, which can be returned by tf.random.uniform, by
  replacing all 0 values with 0.5.

  Args:
    shape: a list or tuple of integers.

  Returns:
    A Tensor of the given shape, a dtype of float32, and all values in the open
    interval (0, 1).
  """
  rnd = tf.random.uniform(shape, dtype=tf.float32)
  return tf.where(tf.equal(rnd, 0.), tf.ones_like(rnd) / 2., rnd)


def _add_condition(z, condition):
  if condition is not None:
    return tf.concat([z, condition], axis=1)
  else:
    return z


def _inverse_sigmoid(x):
  """Inverts a sigmoid."""
  return -tf.math.log(1. / x - 1)


def _log_sigmoid(x):
  """Computes log(sigmoid(x)) while avoiding 0 or infinity on the output."""
  # If \sigma is the sigmoid function, then
  # log \sigma(x) = -log(1 + e^{-x}) = -softplus(-a)
  return -tf.math.softplus(-x)


class ClippedSigmoid(object):
  """Sigmoid function with clipped values to avoid overflows."""

  def __init__(self, clip=None):
    if clip is None:
      self._x_clip = None
      self._y_clip = None
    else:
      self._x_clip = clip
      self._y_clip = np.tanh(clip / 2) / 2 + 0.5

  def _clip(self, x):
    if self._x_clip is None:
      return x
    else:
      return tf.clip_by_value(x, -self._x_clip, self._x_clip)

  def _clip_inverse(self, y):
    if self._y_clip is None:
      return y
    else:
      return tf.clip_by_value(y, -self._y_clip, self._y_clip)

  def apply(self, x):
    x = self._clip(x)
    return tf.sigmoid(x)

  def inverse(self, y):
    y = self._clip_inverse(y)
    return _inverse_sigmoid(y)

  def log_differential(self, x):
    x = self._clip(x)
    # Recall \sigma'(x) = \sigma(x) \times \sigma(-x).
    return _log_sigmoid(x) + _log_sigmoid(-x)

class NothingActivationFunction():
  def apply(self, x):
    return x

  def inverse(self, y):
    return y

  def log_differential(self, x):
    return x


def _soft_clip(x, max_abs):
  """Clips values in range (-max_abs, max_abs) in a soft manner."""
  scale = max_abs / (np.pi / 2.)
  return scale * tf.math.atan(x / scale)


def _softplus(x):
  """A nice softplus.

  This is a monotonous function that is equivalent to x at +inf and
  -1 / (3PI x^2) at -inf. To prove this, use two facts:
  1. For negative x, atan(x) = -PI/2 - artan(1/x)
  2. Near 0, atan(x) = x - x^3/3 + O(x^5)
  From these two things, you can deduce that as x -> -inf:
    atan(x) = -PI/2 - 1/x + 1 / 3x^3 + O(1/x^5).
  This is enough to prove the statement.

  This function decays much slower than the softplus
  from tf.math.softplus at -inf, which decays as exp(x).

  Args:
    x: a Tensor.

  Returns:
    A Tensor of the same type and shape as x.
  """
  x = _soft_clip(x, 1e6)
  y = x * (tf.math.atan(x) / np.pi + 0.5) + 1 / np.pi
  # Mathematically, y above is equivalent to -1/ (3PI x^2) as x goes to -inf,
  # but because of numerical instabilities, the value of y sometimes end up
  # negative.
  # For example, at the time of writing, y is equal to -0.00113806 when
  # x = -33920.55078125. We fix this with the tf.where below.
  equiv_y = 1 / (3 * np.pi * tf.square(x))
  return tf.where(x < -30., equiv_y, y)


def _interlace(z1, z2):
  """Interlace batched flat Tensors.

  See test for an example.

  This is basically the reverse operation of extracting odd and even position
  values. This method satisfies that for any Tensor z:
  z = _interlace(z[:, ::2], z[:, 1::2]).

  Args:
    z1: a Tensor of shape [B, N]
    z2: a Tensor of shape [B, N] or [B, N - 1]

  Returns:
    A Tensor.

  Raises:
    ValueError: if the shapes of z1 and z2 are wrong.
  """
  n1 = z1.get_shape()[1].value
  n2 = z2.get_shape()[1].value
  if n2 not in (n1, n1 - 1):
    raise ValueError(
        'z2 ({}) should be the same shape or 1 shorter than z1 ({}).'.format(
            z2.get_shape().as_list(),
            z1.get_shape().as_list()))
  if n2 < n1:
    # Add some padding.
    z2 = tf.concat([z2, tf.zeros_like(z2[:, :1])], axis=-1)
  z = tf.stack([z1, z2], axis=1)
  z = tf.transpose(z, perm=[0, 2, 1])
  z = tf.reshape(z, shape=[-1, n1 * 2])
  if n2 < n1:
    # Remove the padding.
    z = z[:, :-1]
  return z

class FlatRnvp(snt.AbstractModule):
  """An R-NVP for flat Tensors.

  The main methods are `sample` and `infer`. The `sample(batch_size)` method
  returns a batch of Tensors and their respective log probabilities. The
  `infer(samples)` method returns the inferred samples from the base
  distribution as well as the log probabilities of the input samples.
  """

  def __init__(self,
               latent_size,
               num_blocks,
               num_layers_per_block,
               tf_session,
               judge_output_op,
               hidden_size=None,
               custom_getter=None,
               activation=tf.nn.leaky_relu,
               final_non_linearity=ClippedSigmoid,
               loss_noise_ub=0.0,
               random_state=np.random.RandomState(seed=24),
               name='flat_rnvp'):
    super(FlatRnvp, self).__init__(custom_getter=custom_getter, name=name)
    self.tf_session = tf_session
    self._latent_size = latent_size
    self._num_blocks = num_blocks
    self._num_layers_per_block = num_layers_per_block
    self._hidden_size = hidden_size
    self._activation = activation
    # Note that latent_size might not be even
    self._n1 = latent_size // 2
    # Layers created during `sample` are reused during `infer`.
    self._layers = None
    self._logit_goals = final_non_linearity == ClippedSigmoid
    self._final_non_linearity = final_non_linearity()
    self._loss_noise_ub = loss_noise_ub
    self._random_state = random_state

    self()
    self._judge_output_op = judge_output_op
    self._loss = self._set_loss()
    self._optimizer = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(self._loss, var_list=self._local_variables)

  def _split(self, z):
    z1 = z[:, :self._n1]
    z2 = z[:, self._n1:]
    return z1, z2

  def _build_sample(self, latent, condition):
    self._layers = []
    log_p = -0.5 * tf.reduce_sum(tf.square(latent), axis=1)
    for _ in range(self._num_blocks):
      latent, log_p = self._block_transform(latent, log_p, condition)
    logit = tf.debugging.check_numerics(latent, 'goal latent')
    log_p = log_p - tf.reduce_sum(
      self._final_non_linearity.log_differential(logit), axis=1)
    return self._final_non_linearity.apply(logit), log_p

  def _build_infer(self, samples, condition):
    z = self._final_non_linearity.inverse(samples)
    log_p = -tf.reduce_sum(self._final_non_linearity.log_differential(z),
                           axis=1)
    log_p = tf.debugging.check_numerics(log_p, 'initial_log_p')
    for block_layers in self._layers[::-1]:
      z, log_p = self._inverse_block_transform(
        z, log_p, condition, block_layers)
    log_p = tf.debugging.check_numerics(log_p, 'reverse_log_p')
    log_p -= 0.5 * tf.reduce_sum(tf.square(z), axis=1)
    log_p = tf.debugging.check_numerics(log_p, 'final_log_p')
    return z, log_p

  def _build(self):
    # We rebuild the layers at each call. This does not matter because of the
    # automatic variables sharing, and it simplifies the code to always rebuild.
    self._latent = tf.placeholder(tf.float64, shape=(None, self._latent_size))
    self._samples = tf.placeholder(tf.float64, shape=(None, self._latent_size))
    self._sample_condition = tf.placeholder(tf.float64, shape=(None, 1))
    self._infer_condition = tf.placeholder(tf.float64, shape=(None, 1))

    self.sample_output, self.sample_log_p = self._build_sample(self._latent, self._sample_condition)
    self.infer_output, self.infer_log_p = self._build_infer(self._samples, self._infer_condition)
    self._local_variables = tf.get_collection( # used to avoid training judge's variables
      tf.GraphKeys.TRAINABLE_VARIABLES,
      self._original_name
    )

  def _set_loss(self):
    feasibility_loss = tf.math.reduce_mean(
      tf.math.squared_difference(self._judge_output_op(self.sample_output),
                                 _inverse_sigmoid(self._sample_condition)))
    validity_loss = tf.math.reduce_mean(-self.infer_log_p)
    cover_loss = tf.math.reduce_mean(self.sample_log_p)

    return feasibility_loss + validity_loss + cover_loss

  def sample(self, batch_size, condition=None):
    latent = self._random_state.normal(size=[batch_size, self._latent_size]) # mean: 0, std: 1
    return self.tf_session.run([self.sample_output, self.sample_log_p],
                               feed_dict={self._latent: latent, self._sample_condition: condition})

  def infer(self, samples, condition=None):
    if self._layers is None:
      raise RuntimeError(
          'You need to have called `sample()` at least once before calling '
          '`infer()`')

    return self.tf_session.run([self.infer_output, self.infer_log_p],
                               feed_dict={self._samples: samples, self._infer_condition: condition})

  def train(self, samples, feasibilities, returns):
    reshaped_returns = returns.reshape((len(returns),))
    succeeded_goals = samples[reshaped_returns == 1]
    succeeded_feasibilities = feasibilities[reshaped_returns == 1]

    noise = self._random_state.uniform(low=0, high=self._loss_noise_ub, size=succeeded_goals.shape)
    noisy_succeeded_goals = succeeded_goals+noise
    if self._logit_goals:
      noisy_succeeded_goals = np.clip(noisy_succeeded_goals, 1e-4, 1 - 1e-4) # clip to match sigmoid

    latent = self.tf_session.run(self.infer_output, feed_dict={self._samples: samples, self._infer_condition:feasibilities})
    return self.tf_session.run([self._loss, self._optimizer],
                               feed_dict={
                                 self._samples: noisy_succeeded_goals,
                                 self._infer_condition: succeeded_feasibilities,
                                 self._sample_condition: feasibilities,
                                 self._latent: latent # np.clip(latent , 1e-4, 1 - 1e-4)
                               })


  def _block_transform(self, z, log_p, condition):
    """Builds a block used inside an RNVP."""
    self._layers.append([])
    z, log_p = self._left_to_right_transform(z, log_p, condition,
                                             self._transform)
    z, log_p = self._right_to_left_transform(z, log_p, condition,
                                             self._transform)
    z, log_p = self._interlaced_transform(z, log_p, condition, self._transform)
    return z, log_p

  def _inverse_block_transform(self, z, log_p, condition, block_layers):
    left_to_right, right_to_left, interlace = block_layers
    # Go in reverse order from the transforms in `_block_transform`
    s_net, t_net = interlace
    transform = functools.partial(
        self._inverse_transform, s_net=s_net, t_net=t_net)
    z, log_p = self._interlaced_transform(z, log_p, condition, transform)
    s_net, t_net = right_to_left
    transform = functools.partial(
        self._inverse_transform, s_net=s_net, t_net=t_net)
    z, log_p = self._right_to_left_transform(z, log_p, condition, transform)
    s_net, t_net = left_to_right
    transform = functools.partial(
        self._inverse_transform, s_net=s_net, t_net=t_net)
    z, log_p = self._left_to_right_transform(z, log_p, condition, transform)
    return z, log_p

  def _transform(self, z1, z2, log_p, condition):
    # Implement the update from the RNVP paper, using their notation. Below,
    # replace the exp(s) in the original paper with _softplus in
    # order to get more stable learning.
    zz = _add_condition(z1, condition)
    n = z2.get_shape()[1].value
    if self._hidden_size is None:
      hidden_size = n
    else:
      hidden_size = self._hidden_size
    s_net = Highway(
        output_size=n,
        hidden_size=hidden_size,
        num_blocks=self._num_layers_per_block,
        name='scale')
    t_net = Highway(
        output_size=n,
        hidden_size=hidden_size,
        num_blocks=self._num_layers_per_block,
        name='translation')
    self._layers[-1].append((s_net, t_net))
    s = s_net(zz)
    t = t_net(zz)
    scale = _MIN_SCALE + _softplus(s)
    z2_out = z2 * scale + t
    log_p_out = log_p - tf.reduce_sum(tf.log(scale), axis=-1)

    return z2_out, log_p_out

  def _inverse_transform(self, z1, z2, log_p, condition, s_net, t_net):
    zz = _add_condition(z1, condition)
    s = s_net(zz)
    t = t_net(zz)
    # The division below can lead to very large numbers if we allow the scale
    # to get very small. This effect can compound over multiple layers. That's
    # why we add _MIN_SCALE below.
    scale = _MIN_SCALE + _softplus(s)
    z2_out = (z2 - t) / scale
    log_p_out = log_p - tf.reduce_sum(tf.log(scale), axis=-1)
    return z2_out, log_p_out

  def _left_to_right_transform(self, z, log_p, condition, transform_fn):
    """Affine coupling layer from the RNVP paper."""
    z.get_shape().assert_has_rank(2)
    log_p.get_shape().assert_has_rank(1)
    z1, z2 = self._split(z)
    z2_out, log_p_out = transform_fn(z1, z2, log_p, condition)
    return tf.concat([z1, z2_out], axis=-1), log_p_out

  def _right_to_left_transform(self, z, log_p, condition, transform_fn):
    """Affine coupling layer from the RNVP paper."""
    z.get_shape().assert_has_rank(2)
    log_p.get_shape().assert_has_rank(1)
    z1, z2 = self._split(z)
    # reverse the order of z1 and z2 to get a right to left transform
    z1_out, log_p_out = transform_fn(z2, z1, log_p, condition)
    return tf.concat([z1_out, z2], axis=-1), log_p_out

  def _interlaced_transform(self, z, log_p, condition, transform_fn):
    """Affine coupling layer from the RNVP paper."""
    z.get_shape().assert_has_rank(2)
    log_p.get_shape().assert_has_rank(1)
    z1 = z[:, ::2]
    z2 = z[:, 1::2]
    z1_out, log_p_out = transform_fn(z2, z1, log_p, condition)
    return _interlace(z1_out, z2), log_p_out


class Judge(snt.AbstractModule):
  """Computes logits for probability of a goal being solvable."""

  def __init__(self, hidden_sizes, tf_session, goal_size, custom_getter=None, name='judge'):
    super(Judge, self).__init__(custom_getter=custom_getter, name=name)
    self._layer_sizes = hidden_sizes + [1]
    self.tf_session = tf_session
    self._goal_size = goal_size
    self()
    self._loss = self._set_loss()
    self._optimizer = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(self._loss)

  def _build(self):
    self._mlp = snt.nets.MLP(self._layer_sizes[1:])
    self._goal = tf.placeholder(tf.float64, shape=(None, self._goal_size))
    goal_flat = snt.BatchFlatten()(self._goal)
    self.output = self._mlp(goal_flat) # get non-activated output

  def _set_loss(self):
    self._loss_returns = tf.placeholder(tf.float64, shape=(None, 1))
    return tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(labels=self._loss_returns, logits=self.output))

  def train(self, returns, samples):
    return self.tf_session.run([self._loss, self._optimizer],
                               feed_dict={
                                 self._loss_returns: returns,
                                 self._goal: samples
                               })

  def calculate_feasibility(self, goal):
    return self.tf_session.run(self.output, feed_dict={self._goal: goal})