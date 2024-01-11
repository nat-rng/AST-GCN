import tensorflow as tf
from utils import calculate_laplacian
from tensorflow.keras.regularizers import l1_l2

class TGCNCell(tf.keras.layers.Layer):
    def __init__(self, num_units, adj, num_nodes, l1, l2, act=tf.nn.tanh, **kwargs):
        super(TGCNCell, self).__init__(**kwargs)
        self._act = act
        self._nodes = num_nodes
        self._units = num_units
        self._adj = [calculate_laplacian(adj)]
        self.l1 = l1
        self.l2 = l2
        self.build_weights()

    def build_weights(self):
        regularizer = l1_l2(l1=self.l1, l2=self.l2)
        self.gates_weights = self.add_weight(shape=(self._units + 1, 2 * self._units),
                                             initializer='glorot_uniform',
                                             regularizer=regularizer,
                                             name='gates_weights')
        self.gates_biases = self.add_weight(shape=(2 * self._units,),
                                            initializer='zeros',
                                            name='gates_biases')
        self.candidate_weights = self.add_weight(shape=(self._units + 1, self._units),
                                                 initializer='glorot_uniform',
                                                 regularizer=regularizer,
                                                 name='candidate_weights')
        self.candidate_biases = self.add_weight(shape=(self._units,),
                                                initializer='zeros',
                                                name='candidate_biases')

    @property
    def state_size(self):
        return self._nodes * self._units

    @property
    def output_size(self):
        return self._units

    def call(self, inputs, state):
        state = tf.reshape(state, [-1, self._nodes, self._units])

        value = tf.nn.sigmoid(
            self._gc(inputs, state, 2 * self._units) + self.gates_biases)

        r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
        r = tf.reshape(r, [-1, self._nodes, self._units])
        u = tf.reshape(u, [-1, self._nodes, self._units])
        r_state = r * state

        c = self._act(self._gc(inputs, r_state, self._units) + self.candidate_biases)
        c = tf.reshape(c, [-1, self._nodes, self._units])
        new_h = u * state + (1 - u) * c
        new_h = tf.reshape(new_h, [-1, self._nodes * self._units])

        return new_h, new_h

    def _gc(self, inputs, state, output_size):
        inputs = tf.expand_dims(inputs, 2)
        state = tf.reshape(state, (-1, self._nodes, self._units))
        x_s = tf.concat([inputs, state], axis=2)
        input_size = x_s.get_shape()[2]
        x0 = tf.transpose(x_s, perm=[1, 2, 0])
        x0 = tf.reshape(x0, shape=[self._nodes, -1])

        for m in self._adj:
            x1 = tf.sparse.sparse_dense_matmul(m, x0)

        x = tf.reshape(x1, shape=[self._nodes, input_size, -1])
        x = tf.transpose(x, perm=[2, 0, 1])
        x = tf.reshape(x, shape=[-1, input_size])

        weights = self.gates_weights if output_size == 2 * self._units else self.candidate_weights
        x = tf.matmul(x, weights)
        biases = self.gates_biases if output_size == 2 * self._units else self.candidate_biases
        x = tf.nn.bias_add(x, biases)
        return x
    
class TGCNModel(tf.keras.Model):
    def __init__(self, num_nodes, gru_units, adj, pre_len, l1, l2):
        super(TGCNModel, self).__init__()
        self.num_nodes = num_nodes
        self.gru_units = gru_units
        self.adj = adj
        self.pre_len = pre_len
        self.tgcn_cell = TGCNCell(gru_units, adj, num_nodes, l1, l2)
        self.rnn = tf.keras.layers.RNN(self.tgcn_cell, return_sequences=True)
        self.dense_out = tf.keras.layers.Dense(num_nodes * pre_len)

    def call(self, inputs):
        timesteps = inputs.shape[1]
        x = self.rnn(inputs)
        x = tf.reshape(x, [-1, self.num_nodes*timesteps*self.gru_units])

        x = self.dense_out(x)

        x = tf.reshape(x, [-1, self.pre_len, self.num_nodes])
        return x