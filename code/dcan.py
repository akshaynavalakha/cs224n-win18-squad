# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
from operator import mul


class LSTMEncoder(object):

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size #200
        self.keep_prob = keep_prob
        self.lstm = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size) #BasicLSTMCell LSTMBlockCell
        # TODO not sure if I want to add dropout here
        #self.lstm = tf.nn.rnn_cell.DropoutWrapper(cell=self.lstm, input_keep_prob=self.keep_prob)


    def build_graph(self, inputs, masks, type):

        with vs.variable_scope("LSTMEncoder"):
            # get length of Q and C
            input_lens = tf.reduce_sum(masks, reduction_indices=1)
            inputs_size = inputs.get_shape().as_list()
            inputs_temp = inputs

            # TODO replace this with bidirectional LSTM to get more hidden states
            # 1) get encoding from LSTM
            C_or_Q, _ = tf.nn.dynamic_rnn(self.lstm, inputs_temp, dtype=tf.float32)


            if type=="question":
                # IF it is Question_hidden
                # Calculate q_dash = tanh(W q + b)
                q_dash = tf.layers.dense(inputs_temp, C_or_Q.get_shape()[2], activation=tf.tanh)
                inputs_temp = q_dash
                # TODO if BiLSTM used then hideen size should change here
                sentinel = tf.get_variable("sentinel_q", [1, 1, self.hidden_size], initializer=tf.random_normal_initializer())
            else:
                # IF it is Context_hidden
                sentinel = tf.get_variable("sentinel_c", [1, 1, self.hidden_size], initializer=tf.random_normal_initializer()) # 1,200
                inputs_temp = C_or_Q
            #
            # reshape sentinel to add batch
            sentinel_tile = tf.tile(sentinel, [tf.shape(inputs_temp)[0], 1, 1]) #?, 1, 200(h)
            # add sentinel at beginning!!!
            out = tf.concat([sentinel_tile, inputs_temp], axis=1) # ?, 601, 200(h)

            out.get_shape().as_list()

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class CoAttention(object):

    def __init__(self, keep_prob, context_hidden_size, query_hidden_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.context_hidden_size = context_hidden_size
        self.query_hidden_size = query_hidden_size


    def build_graph(self, question_hiddens, qn_mask, context_hiddens, context_mask):

        with vs.variable_scope('Coattention') as scope:
            question_hiddens.get_shape().as_list() #? , 31, 200
            context_hiddens.get_shape().as_list()  #? ,601, 200

            #Update masks add sentinel of ones at beginning
            q_sent = tf.ones([tf.shape(qn_mask)[0], 1], dtype=tf.int32)
            c_sent = tf.ones([tf.shape(context_mask)[0], 1], dtype=tf.int32)

            qn_mask = tf.concat([q_sent, qn_mask], axis=1)
            context_mask = tf.concat([c_sent, context_mask], axis=1)

            question_length = tf.shape(question_hiddens)[1]
            context_length = tf.shape(context_hiddens)[1]
            keys_dim = tf.shape(context_hiddens)[2]


            Q_tranpose = tf.transpose(question_hiddens, perm=[0, 2, 1]) #?, 200, 31

            # L = D.T * Q = C.T * Q
            L = tf.matmul(context_hiddens, Q_tranpose)  #?, 601, 31

            L_transpose = tf.transpose(L, perm=[0, 2, 1]) #?, 31, 601

            A_D = tf.map_fn(lambda x: tf.nn.softmax(x), L_transpose, dtype=tf.float32) #?, 31, 601

            A_Q = tf.map_fn(lambda x: tf.nn.softmax(x), L, dtype=tf.float32) #?, 601, 31

            C_Q = tf.matmul(tf.transpose(context_hiddens, perm=[0, 2, 1]), A_Q)  #?, 200, 31

            Q_concat_CQ = tf.concat([Q_tranpose, C_Q], axis=1) #?,  400(2h), 31

            C_D = tf.matmul(Q_concat_CQ, A_D) #?, 400(2h), 601

            CO_ATT = tf.concat([context_hiddens, tf.transpose(C_D, perm=[0, 2, 1])], axis=2) #?, 601, 600(3h)

            with tf.variable_scope('Coatt_encoder'):
                # LSTM for coattention encoding
                cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.query_hidden_size)
                cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.query_hidden_size)
                input_lens = tf.reduce_sum(context_mask, reduction_indices=1)
                #?, 601, 400
                (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, CO_ATT,
                                                       dtype=tf.float32, sequence_length=input_lens)
                U_1 = tf.concat([fw_out, bw_out], axis=2)
                
                dims = U_1.get_shape().as_list()
                # Remove the sentinel vector from beginning
                U_2 = U_1[:, 1:]


            #?, 601, 400
            out = tf.nn.dropout(U_2, self.keep_prob)
            #?, 601, 400(2h)
            return out


