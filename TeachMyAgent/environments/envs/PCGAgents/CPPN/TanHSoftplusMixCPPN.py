import tensorflow as tf
from TeachMyAgent.environments.envs.PCGAgents.CPPN.BaseCPPN import BaseCPPN

class TanHSoftplusMixCPPN(BaseCPPN):
    '''
    Feedforward CPNN with 4 layers of 64 units alternating TanH/Softplus activation functions.
    '''
    def generator(self):
        tf.reset_default_graph()
        # inputs to cppn
        self.input = tf.placeholder(tf.float32, [self.x_dim, self.input_dim + 1])

        self.h1_weights = tf.Variable(tf.truncated_normal([self.input_dim + 1, 64], mean=0.0, stddev=2.0))
        h1 = tf.nn.tanh(tf.matmul(self.input, self.h1_weights))

        self.h2_weights = tf.Variable(tf.truncated_normal([64, 64], mean=0.0, stddev=2.0))
        h2 = tf.nn.softplus(tf.matmul(h1, self.h2_weights))

        self.h3_weights = tf.Variable(tf.truncated_normal([64, 64], mean=0.0, stddev=2.0))
        h3 = tf.nn.tanh(tf.matmul(h2, self.h3_weights))

        self.h4_weights = tf.Variable(tf.truncated_normal([64, 64], mean=0.0, stddev=2.0))
        h4 = tf.nn.softplus(tf.matmul(h3, self.h4_weights))

        self.output_weights = tf.Variable(tf.truncated_normal([64, self.output_dim], mean=0.0, stddev=2.0))
        output = tf.matmul(h4, self.output_weights)
        result = tf.reshape(output, [self.x_dim, self.output_dim])

        return result