
import tensorflow as tf

from constants import *
from layers import *
from utils_misc import numpy2jpg

class Unet(object):

    def __init__(self, input_shape, output_shape, verbose=True):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None]+self.input_shape, name="inputs")
        self.output_placeholder = tf.placeholder(dtype=tf.float32, shape=[None]+self.output_shape, name="outputs")
        self.is_train = tf.placeholder(dtype=tf.bool, name='is_train')
        
        self.config = Config()
        self.layers = {}
        self.params = {}

        self.verbose = verbose

        if self.verbose:
            print("Completed Initializing the Unet Model.....")

    def create_model(self):
        currLayer = self.input_placeholder
        self.layers['input'] = currLayer

        for layerName in self.config.layer_keys:
            layerParams = self.config.layer_params[layerName]
            currLayer = conv_relu(currLayer, layerParams, is_train=self.is_train, name=layerName, verbose=self.verbose)

            self.layers[layerName] = currLayer

        self.layers['output'] = self.layers[layerName]

        # softmax loss prep
        # self.logits_op = tf.add(tf.matmul(average_embedding, weight), bias)
        # self.probabilities_op = tf.nn.softmax(self.logits_op)

        self.add_loss_op()

        if self.verbose:
            print("The name of the output layer is: " + layerName)
            print("Built the Unet Model.....")

    def add_loss_op(self):
        # regression loss
        self.loss_op = tf.losses.mean_squared_error(self.output_placeholder, self.layers['output'])

        # softmax loss
        # self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_op, labels=self.label_placeholder))
        
        tf.summary.scalar('Loss', self.loss_op)
        self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss_op)

    def metrics(self):
        # last_axis = len(self.probabilities_op.get_shape().as_list())
        # self.prediction_op = tf.to_int32(tf.argmax(self.probabilities_op, axis=last_axis-1))
        # difference = self.label_placeholder - self.prediction_op
        # zero = tf.constant(0, dtype=tf.int32)
        # boolean_difference = tf.cast(tf.equal(difference, zero), tf.float64)
        # self.accuracy_op = tf.reduce_mean(boolean_difference)

        tf.summary.scalar('Accuracy', self.loss_op)
        self.summary_op = tf.summary.merge_all()

    def run(self, sess, in_batch, out_batch, is_train):
        feed_dict = {
            self.is_train           : is_train,
            self.input_placeholder  : in_batch,
            self.output_placeholder : out_batch
        }

        if is_train:
            _, summary, loss = sess.run([self.train_op, self.summary_op, self.loss_op], feed_dict=feed_dict)
        else:
            summary, loss = sess.run([self.summary_op, self.loss_op], feed_dict=feed_dict)

        if self.verbose:
            # print("Average accuracy per batch {0}".format(accuracy))
            print("Batch Loss: {0}".format(loss))

        return summary, loss


    def sample(self, sess, in_batch, imgName=None):
        feed_dict = {
            self.is_train           : False,
            self.input_placeholder  : in_batch
        }

        out_img = sess.run([self.layers['output']], feed_dict=feed_dict)[0][0,:,:,:]

        if imgName!=None:
            numpy2jpg(imgName, out_img, meanVal=1, verbose=False)

        return out_img