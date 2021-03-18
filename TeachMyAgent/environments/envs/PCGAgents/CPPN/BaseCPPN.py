import tensorflow as tf
import numpy as np

class BaseCPPN():
    '''
    Simple Base CPPN class wich taks a input vector in addition of a vector defining the x position.
    '''
    def __init__(self, x_dim, input_dim, batch_size=1, output_dim=1, weights_path=None):
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.x_dim = x_dim
        self.input_dim = input_dim

        # builds the generator network
        self.G = self.generator()

        self.init()

        if weights_path is not None:
            saver = tf.train.Saver()
            saver.restore(self.sess, weights_path)

    def init(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def generator(self):
        tf.reset_default_graph()
        # inputs to cppn
        self.input = tf.placeholder(tf.float32, [self.x_dim, self.input_dim + 1])

        output_weights = tf.Variable(tf.truncated_normal([self.input_dim + 1, self.output_dim]))
        output = tf.matmul(self.input, output_weights)
        result = tf.reshape(output, [self.x_dim])

        return result

    def generate(self, input_vector):
        x = np.arange(self.x_dim)
        scaled_x = x / (self.x_dim - 1)
        x_vec = scaled_x.reshape((self.x_dim, 1))
        reshaped_input_vector = np.ones((self.x_dim, self.input_dim)) * input_vector
        final_input = np.concatenate((x_vec, reshaped_input_vector), axis=1)
        return self.sess.run(self.G, feed_dict={self.input: final_input})

    def close(self):
        self.sess.close()