import tensorflow as tf
import numpy as np

class BaseCPPN():
    '''
        Simple Base CPPN class wich takes an input vector in addition of a vector defining the x position.
    '''
    def __init__(self, x_dim, input_dim, batch_size=1, output_dim=1, weights_path=None):
        '''
            Builds the computational graph.

            Args:
                x_dim: How many times the CPPN should slide on the x axis
                input_dim: Size of the input vector controlling the generated pattern
                batch_size: Size of batch provided
                output_dim: Size of the output vector
                weights_path: Path to load weights. If None, weights are randomly sampled
        '''
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
        '''
            Initialize the tf session.
        '''
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def generator(self):
        '''
            Build the computational graph.
        '''
        tf.reset_default_graph()
        # inputs to cppn
        self.input = tf.placeholder(tf.float32, [self.x_dim, self.input_dim + 1])

        output_weights = tf.Variable(tf.truncated_normal([self.input_dim + 1, self.output_dim]))
        output = tf.matmul(self.input, output_weights)
        result = tf.reshape(output, [self.x_dim])

        return result

    def generate(self, input_vector):
        '''
            Generate an output of size `batch_size*x_dim*output_size` given the input vector
        '''
        x = np.arange(self.x_dim)
        scaled_x = x / (self.x_dim - 1)
        x_vec = scaled_x.reshape((self.x_dim, 1))
        reshaped_input_vector = np.ones((self.x_dim, self.input_dim)) * input_vector
        final_input = np.concatenate((x_vec, reshaped_input_vector), axis=1)
        return self.sess.run(self.G, feed_dict={self.input: final_input})

    def close(self):
        '''
            Close the tf session.
        '''
        self.sess.close()