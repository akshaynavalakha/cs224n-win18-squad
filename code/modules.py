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
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell



class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output

class BiDafAttn(object):
    """Module for bidirectional attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the bidaf model, the keys are the context hidden states
    and the values are the question hidden states for C2Q attention.

    the keys are the question hidden states and values are context hidden states
    and the values are the question hidden states for C2Q attention.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys, keys_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BiDafAttn"):

            # Calculate the similarity matrix
            num_values = tf.shape(values)[1]
            num_keys = tf.shape(keys)[1]
            keys_dim = tf.shape(keys)[2]



            # Expand the dimension of keys and values so they have the same number of dimension
            # We can use them for broadcasting
            keys_expand = tf.expand_dims(keys,2) # (batch_size, num_keys, 1,key_vec_size)
            values_expand = tf.expand_dims(values, 1)  # (batch_size, 1, num_values,key_vec_size)

            temp = tf.multiply(keys_expand, values_expand) #(batch_size, num_keys, num_values, key_vec_size)

            #The similarity matrix is S = wsim.T[ci ; qj ; ci * qj]
            # Because of broadcasting the dot product can be expressed as the sum of the
            #Individual dot product

            # wsim.T dot c
            logits_keys = tf.layers.dense(keys_expand, 1, activation=None) # (batch_size, num_keys, 1, 1)

            #wsim.T dot q
            logits_values = tf.layers.dense(values_expand, 1, activation=None)  # (batch_size, 1, num_values, 1)

            #wsim.t dot (c * q)
            logits_temp =  tf.layers.dense(temp, 1, activation=None) #(batch_size, num_keys, num_value, 1)

            similarity_matrix = logits_keys + logits_values + logits_temp  # # (batch_size, num_keys, num_values, 1)
            similarity_matrix = tf.squeeze(similarity_matrix, axis=[3])  # (batch_size, num_keys, num_values)

            values_mask_expand = tf.expand_dims(values_mask,1)


            m_vector = tf.reduce_max(similarity_matrix, axis=2) # (batch_size, num_keys)
            _, c_hat_attn = masked_softmax(m_vector, keys_mask, 1)
            c_hat = tf.reduce_sum(tf.multiply(tf.expand_dims(c_hat_attn,2),keys),1) #batch_size, vec_size
            c_hat_expand = tf.expand_dims(c_hat, 1)   # USE for broadcasting batch_size, 1 vec_size

            _, a_hat_attn = masked_softmax(similarity_matrix, values_mask_expand, 2) #(batch_size, num_keys, num_values)
            # Use attention distribution to take weighted sum of values
            a_hat = tf.matmul(a_hat_attn, values)  # shape (batch_size, num_keys, value_vec_size)

            output = tf.concat([keys, a_hat, tf.multiply(keys,a_hat),tf.multiply(keys, c_hat_expand)], axis=2) # shape (batch_size, num_keys, 4*value_vec_size)


            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)


            return  output




def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist


class RNNEncoder_LSTM(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder_LSTM"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class MODEL_LAYER_BIDAF(object):
    """
    This is a 2 layer LSTM network

    It takes in the input from the attention layer

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob

        # layer 1
        self.rnn_cell_fw_1 = rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
        self.rnn_cell_fw_1 = DropoutWrapper(self.rnn_cell_fw_1, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw_1 = rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
        self.rnn_cell_bw_1 = DropoutWrapper(self.rnn_cell_bw_1, input_keep_prob=self.keep_prob)

        # layer 2
        self.rnn_cell_fw_2 = rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
        self.rnn_cell_fw_2 = DropoutWrapper(self.rnn_cell_fw_2, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw_2 = rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
        self.rnn_cell_bw_2 = DropoutWrapper(self.rnn_cell_bw_2, input_keep_prob=self.keep_prob)


    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("MODEL_LAYER"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out_1, bw_out_1), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw_1, self.rnn_cell_bw_1, inputs, input_lens, dtype=tf.float32, scope="layer1")

            # Concatenate the forward and backward hidden states
            layer_1_out = tf.concat([fw_out_1, bw_out_1], 2)

            # Apply dropout
            # Is this needed ?????
            layer_1_out = tf.nn.dropout(layer_1_out, self.keep_prob)


            (fw_out_2, bw_out_2), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw_2, self.rnn_cell_bw_2, layer_1_out, input_lens, dtype=tf.float32, scope="layer2")

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out_2, bw_out_2], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

        return out


class END_WORD_LAYER(object):
    """
    This is a 2 layer LSTM network

    It takes in the input from the attention layer

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob

        # layer 1
        self.rnn_cell_fw_1 = rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
        self.rnn_cell_fw_1 = DropoutWrapper(self.rnn_cell_fw_1, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw_1 = rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
        self.rnn_cell_bw_1 = DropoutWrapper(self.rnn_cell_bw_1, input_keep_prob=self.keep_prob)



    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("END_WORD"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out_1, bw_out_1), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw_1, self.rnn_cell_bw_1, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out_1, bw_out_1], 2)

            # Apply dropout

            out = tf.nn.dropout(out, self.keep_prob)

        return out


class ANSWER_DECODER(object):
    """
    This is a 2 layer LSTM network

    It takes in the input from the attention layer

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob, max_iteration, max_pool, batch_size):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.max_iteration = max_iteration
        self.max_pool = max_pool
        self.batch_size = batch_size

        self.rnn_cell_fw = rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)

    def build_graph(self, U_matrix, masks, u_s, u_e):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """

        # us has a shape of [batch_size, 1, dim]
        # ue has a shape of [batch_size, 1, dim]
        # U has a shape of [batch_size, seq_len, dim]
        # masks has a dimension [batch_size, seq_len]

       # batch_size = U_matrix.get_shape().as_list()[0]
        u_start = u_s
        u_end = u_e
        pos = tf.convert_to_tensor(np.arange(self.batch_size), dtype=tf.int32)

        highway_alpha = HNM(self.hidden_size, self.keep_prob, self.max_iteration, self.max_pool, "alpha")
        highway_beta  = HNM(self.hidden_size, self.keep_prob, self.max_iteration, self.max_pool, "beta")
        start = []
        end   = []
        alpha_logits = []
        beta_logits = []
        cell = self.rnn_cell_fw

        init_state = self.rnn_cell_fw.zero_state(self.batch_size, dtype=tf.float32)
        with vs.variable_scope("ANSWER_DECODER"):
            for time_step in range(self.max_iteration):
                input_lstm = tf.concat([u_start, u_end], axis=1)  # (batch_size, 4*hidden_szie)
                #input_lstm = tf.expand_dims(input_lstm, 1) #(batch_size, 1, 4*hidden_size)
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                    reuse = True

                output, hi =  cell(input_lstm, init_state)  #(batch_size,1, hidden_size)

                alpha = highway_alpha.build_graph(U_matrix, masks, output, u_start, u_end, time_step ) #(batch_size, context_len)
                s_indx = tf.argmax(alpha, 1)  # (batch_size, context_len)
                # Update the u_start and u_end for the next iteration
                fn_s = lambda position: index(U_matrix, s_indx, position)
                u_start = tf.map_fn(lambda position: fn_s(position), pos, dtype=tf.float32)



                beta  = highway_beta.build_graph(U_matrix, masks, output, u_start, u_end, time_step ) #(batch_size , context_len)
                e_indx = tf.argmax(beta, 1) #(batch_size, context_len)

                # Update the u_start and u_end for the next iteration
                fn_s = lambda position: index(U_matrix, s_indx, position)
                u_start = tf.map_fn(lambda position: fn_s(position), pos, dtype=tf.float32)

                fn_e = lambda position: index(U_matrix, e_indx, position)
                u_end = tf.map_fn(lambda position: fn_e(position), pos, dtype=tf.float32)

                # update the init_state for the next iteration
                init_state = hi

                if time_step != 0:
                    start.append(s_indx)
                    end.append(e_indx)
                    alpha_logits.append(alpha)
                    beta_logits.append(beta)

        return start, end, alpha_logits, beta_logits



class HNM(object):
    " The function is used to calcualte the maxout given a Attention matrix , initial_guess, at hidden state"

    def __init__(self, hidden_size, keep_prob, max_iteration, max_pool, scope):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.max_iteration = max_iteration
        self.max_pool = max_pool
        self.scope = scope

    def build_graph(self, U_matrix, masks, hidden_state, u_start, u_end, time_step):


        with vs.variable_scope(self.scope):

            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
                use = True
            else:
                use = None

            input_r = tf.concat([hidden_state, u_start, u_end], axis=1) # (batch_size, hidden_size + 2 * dims)
            input_r = tf.expand_dims(input_r, 1)
            r = tf.contrib.layers.fully_connected(input_r, num_outputs=self.hidden_size,activation_fn=tf.nn.tanh, scope=self.scope +'r', reuse=use)  # (batch_size , 1, hidden_size)

            W_u = tf.contrib.layers.fully_connected(U_matrix, num_outputs=self.hidden_size*self.max_pool,activation_fn=None,scope=self.scope +'W_u', reuse=use)  # (batch_size, seq_len , hidden_size)
            W_r = tf.contrib.layers.fully_connected(r, num_outputs=self.hidden_size*self.max_pool, activation_fn=None, scope=self.scope +'W_r', reuse=use)  # (bacth_szie, 1 , hidden_size)

            m1_maxout_input = W_u + W_r  # (bacth_size , seq_len, hidden_size*max_pool)

            m1 = tf.contrib.layers.maxout(m1_maxout_input, num_units=self.hidden_size, name=self.scope + 'm1')  # (bacth_size, seq_len, hidden_size)
            m2_input = tf.contrib.layers.fully_connected(m1, num_outputs=self.hidden_size*self.max_pool, activation_fn=None, scope=self.scope +'m2_input', reuse=use)
            m2 = tf.contrib.layers.maxout(m2_input, num_units=self.hidden_size, name=self.scope + 'm2')  # (bacth_size, seq_len, hidden_size)

            out_in = tf.contrib.layers.fully_connected(tf.concat([m1, m2], axis=2), num_outputs=self.max_pool, activation_fn=None, scope=self.scope +'out_in', reuse=use)
            out = tf.contrib.layers.maxout(out_in, num_units=1, name=self.scope + 'out')  # (bacth_size, seq_len, 1)
            out = tf.squeeze(out, axis=2) #(batch_size, seq_len)

        return out



def index(U, s , position):
    u_words = tf.gather(U, position)
    s_indx = tf.gather(s, position)
    word = tf.gather(u_words, s_indx)

    return word











