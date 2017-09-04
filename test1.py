from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.framework import ops


def binary(x):
    if x <= 0:
        return 0
    else:
        return 1
np_binary = np.vectorize(binary)

def d_binary(x):
    a=1000
    a*np.exp(-a*x) / ((1 + np.exp(-a*x))*(1 + np.exp(-a*x)))
    r = x % 1
    if r <= 0.5:
        return 1
    else:
        return 0
np_d_binary = np.vectorize(d_binary)

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

np_binary_32 = lambda x: np_binary(x).astype(np.float32)
def tf_binary_activation(x, name=None):
    with ops.op_scope([x], name, "binary") as name:
        y = py_func(np_binary_32,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=binarygrad)  # <-- here's the call to the gradient
        return y[0]

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
                 scope='',
                 input_tensor=[]):
        Layer.__init__(self)
        self.output_dim = output_dim
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.scope = scope
        self.build()
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
        self.fc = tf.matmul(input_tensor, self.weights) + self.bias
        # activation
        if self.activation:
            self.fca = self.activation(self.fc)
        else:
            self.fca=self.fc

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
        self.fc1 = FullyConnected(2, activation=tf.nn.sigmoid, scope='encode', self.input)
        self.reconstruct = FullyConnected(2, activation=tf.nn.sigmoid, scope='decode', self.fc1.fca)
        self.loss = tf.nn.l2_loss(self.input - self.reconstruct.fca)  # L2 loss
        self.training = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        self.start_new_session()

    def get_batch(self, batch_size, data):
        return data[np.random.randint(data.shape[0], size=batch_size)]

    def getLineFromWeight(self, grid_image, nnum):
        w = self.fc1.weights.eval(session=self.sess)
        b = self.fc1.bias.eval(session=self.sess)
        for u in range(grid_image.shape[0]):
            for i in range(w.shape[0])
                v = b[i]+u*w[i]
                if(v>=0 and v<grid_image.shape[1])
                grid_image[u,v]=255
        return grid_image


    def train(self, data, batch_size, passes, new_training=True):
        if new_training:
            saver, global_step = self.start_new_session()
        else:
            saver, global_step = self.continue_previous_session(ckpt_file='saver/checkpoint')

        max_data = np.max(data, axis=0)
        min_data = np.min(data, axis=0)
        normdata = np.int16((data - min_data)*999 / (max_data - min_data))
        grid_image = np.zeros(shape=(1000, 1000))
        for i in range(normdata.shape[0]):
            grid_image[normdata[i,0],normdata[i,1]]=255

        for step in range(1 + global_step, 1 + passes + global_step):
            self.training.run(session=self.sess, feed_dict={self.input: self.get_batch(batch_size, data)})
            if step % 10 == 0:
                loss = self.loss.eval(session=self.sess, feed_dict={self.input: self.get_batch(batch_size, data)})
                print("pass {}, training loss {}".format(step, loss))
            if step % 100 == 0:  # save weights
                saver.save(self.sess, 'saver/cnn', global_step=step)
                print('checkpoint saved')
                plt.imshow(grid_image, cmap=plt.cm.gray, interpolation='nearest')
                plt.waitforbuttonpress(timeout=0.1)


N=3200
data = np.random.rand(N,2)
data[0:np.int16(N/2),:]+=np.ones(shape=(np.int16(N/2),2))*10
model = ACNN()
model.train(data, np.int16(N/10), 1000)