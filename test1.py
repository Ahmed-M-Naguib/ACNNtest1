from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

class Layer(object, metaclass=ABCMeta):
    def __init__(self):
        pass
    @abstractmethod
    def call(self, *args, **kwargs):
        raise NotImplementedError
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

class FullyConnected(Layer):
    def __init__(self,
                 output_dim,
                 weights=None,
                 bias=None,
                 activation=None,
                 scope=''):
        Layer.__init__(self)
        self.output_dim = output_dim
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.scope = scope
    def build(self, input_tensor):
        num_batch, input_dim = input_tensor.get_shape()
        # build weights
        if self.weights:
            assert self.weights.get_shape() == (input_dim.value, self.output_dim)
        else:
            self.weights = tf.Variable(tf.truncated_normal((input_dim.value, self.output_dim), stddev=0.1),
                                       name='weights')
        # build bias
        if self.bias:
            assert self.bias.get_shape() == (self.output_dim, )
        else:
            self.bias = tf.Variable(tf.constant(0.1, shape=[self.output_dim]), name='bias')
        # fully connected layer
        fc = tf.matmul(input_tensor, self.weights) + self.bias
        # activation
        if self.activation:
            return self.activation(fc)
        return fc
    def call(self, input_tensor):
        if self.scope:
            with tf.variable_scope(self.scope) as scope:
                return self.build(input_tensor)
        else:
            return self.build(input_tensor)

class ACNN:
    def start_new_session(self):
        saver = tf.train.Saver()  # create a saver
        global_step = 0
        self.sess.run(tf.global_variables_initializer())
        print('started a new session')
        return saver, global_step

    def continue_previous_session(self, ckpt_file):
        saver = tf.train.Saver()  # create a saver
        with open(ckpt_file) as file:  # read checkpoint file
            line = file.readline()  # read the first line, which contains the file name of the latest checkpoint
            ckpt = line.split('"')[1]
            global_step = int(ckpt.split('-')[1])
        saver.restore(self.sess, 'saver/' + ckpt)
        print('restored from checkpoint ' + ckpt)

        return saver, global_step

    def __init__(self):
        self.sess=tf.Session()
        self.input=tf.placeholder(dtype=tf.float32, shape=[None, 2], name='input')
        self.fc1 = FullyConnected(2, activation=tf.nn.relu, scope='encode')(self.input)
        self.reconstruct = FullyConnected(2, activation=tf.nn.sigmoid, scope='decode')(self.fc1)
        self.loss = tf.nn.l2_loss(self.input - self.reconstruct)  # L2 loss
        self.training = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        self.start_new_session()

    def get_batch(self, batch_size, data):
        return data[np.random.randint(data.shape[0], size=batch_size)]

    def train(self, data, batch_size, passes, new_training=True):
        if new_training:
            saver, global_step = self.start_new_session()
        else:
            saver, global_step = self.continue_previous_session(ckpt_file='saver/checkpoint')
        for step in range(1 + global_step, 1 + passes + global_step):
            self.training.run(session=self.sess, feed_dict={self.input: self.get_batch(batch_size, data)})
            if step % 10 == 0:
                loss = self.loss.eval(session=self.sess, feed_dict={self.input: self.get_batch(batch_size, data)})
                print("pass {}, training loss {}".format(step, loss))
            if step % 1000 == 0:  # save weights
                saver.save(self.sess, 'saver/cnn', global_step=step)
                print('checkpoint saved')



data = np.random.rand(320,2)
model = ACNN()
model.train(data, 32, 1000)