#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3(d): Grooving with GRUs
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import tensorflow as tf
import numpy as np

class LSTMCell(tf.nn.rnn_cell.LSTMCell):
    """Wrapper around our LSTM cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, input_size, state_size):
        self.input_size = input_size
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def __call__(self, inputs, state, scope=None):
        """Updates the state using the previous @state and @inputs.

        Args:
            inputs: is the input vector of size [None, self.input_size]
            state: is the previous state vector of size [None, self.state_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector pair.
        """
        scope = scope or type(self).__name__

        with tf.variable_scope(scope):

            c_t0, h_t0 = state
            
            xinit = tf.contrib.layers.xavier_initializer(dtype=tf.float64)
            W_i = tf.get_variable("W_i", shape=[self.input_size, self._state_size], initializer=xinit, dtype=tf.float64)
            U_i = tf.get_variable("U_i", shape=[self._state_size, self._state_size], initializer=xinit, dtype=tf.float64)
            b_i = tf.get_variable("b_i", shape=[self._state_size], initializer=tf.constant_initializer(0.0), dtype=tf.float64)

            W_f = tf.get_variable("W_f", shape=[self.input_size, self._state_size], initializer=xinit, dtype=tf.float64)
            U_f = tf.get_variable("U_f", shape=[self._state_size, self._state_size], initializer=xinit, dtype=tf.float64)
            b_f = tf.get_variable("b_f", shape=[self._state_size], initializer=tf.constant_initializer(0.0), dtype=tf.float64)

            W_o = tf.get_variable("W_o", shape=[self.input_size, self._state_size], initializer=xinit, dtype=tf.float64)
            U_o = tf.get_variable("U_o", shape=[self._state_size, self._state_size], initializer=xinit, dtype=tf.float64)
            b_o = tf.get_variable("b_o", shape=[self._state_size], initializer=tf.constant_initializer(0.0), dtype=tf.float64)

            W_u = tf.get_variable("W_u", shape=[self.input_size, self._state_size], initializer=xinit, dtype=tf.float64)
            U_u = tf.get_variable("U_u", shape=[self.input_size, self._state_size], initializer=xinit, dtype=tf.float64)
            b_u = tf.get_variable("b_u", shape=[self._state_size], initializer=tf.constant_initializer(0.0), dtype=tf.float64)
            
            i_t = tf.sigmoid(tf.matmul(inputs,W_i) + tf.matmul(h_t0, U_i) + b_i)
            f_t = tf.sigmoid(tf.matmul(inputs,W_f) + tf.matmul(h_t0, U_f) + b_f)
            o_t = tf.sigmoid(tf.matmul(inputs,W_o) + tf.matmul(h_t0, U_o) + b_o)
            u_t = tf.tanh(tf.matmul(inputs,W_u) + tf.matmul(h_t0, U_u) + b_u)

            c_t = i_t*u_t + f_t*c_t0

            h_t = o_t * tf.tanh(c_t)

        output = h_t
        new_state = (c_t, h_t)
        return output, new_state
