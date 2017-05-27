
import tensorflow as tf
import numpy as np
import os

from constants import *
from layers import *
from utils_misc import numpy2jpg,h52numpy
from scipy.misc import imresize,toimage

class Unet(object):

    def __init__(self, input_shape, output_shape, verbose=True):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None]+self.input_shape, name="inputs")
        self.output_placeholder = tf.placeholder(dtype=tf.float32, shape=[None]+self.output_shape, name="outputs")
        self.is_train = tf.placeholder(dtype=tf.bool, name='is_train')
        
        self.config = UnetConfig()
        self.layers = {}
        self.params = {}
        self.visual = {}

        self.fnameMod = 0

        self.SAMPLE_INPUT, self.SAMPLE_OUTPUT, self.SAMPLE_NAMES = h52numpy(SAMPLE_DATA_FILE, checkMean=False, batch_sz=11, 
                                                                            mod_output=False, iter_val=None, shuffle=False)

        self.verbose = verbose

        if self.verbose:
            print("Completed Initializing the Unet Model.....")

    def create_model(self):
        currLayer = self.input_placeholder
        self.layers['input'] = currLayer

        for layerName in self.config.layer_keys:
            layerParams = self.config.layer_params[layerName]
            currLayer = conv_relu(currLayer, layerParams, is_train=self.is_train, name=layerName, verbose=self.verbose)

            # fuse layers if appropriate
            if self.config.use_fuse and (layerName in self.config.fuse_layers):
                fuse_target = self.config.fuse_layers[layerName]
                currLayer = tf.concat([self.layers[fuse_target], currLayer], axis=3)
                
                print('Fusing ' + layerName + ' and ' + fuse_target)
                print('Layer shape: {0}\n'.format(currLayer.shape))

            self.layers[layerName] = currLayer

        self.layers['output'] = self.layers[layerName]

        # add tensors for visual purposes
        self.visual['input'] = self.input_placeholder + LINE_MEAN
        self.visual['ground_truth'] = self.output_placeholder + [REDUCED_R_MEAN,REDUCED_G_MEAN,REDUCED_B_MEAN]
        self.visual['predicted'] = self.layers['output'] + [REDUCED_R_MEAN,REDUCED_G_MEAN,REDUCED_B_MEAN]
        self.visual['predicted_overlay'] = self.visual['predicted'] * (self.visual['input']/255.0)

        self.add_loss_op()

        if self.verbose:
            print("The name of the output layer is: " + layerName)
            print("Built the Unet Model.....")

    def add_loss_op(self):
        # regression loss
        self.loss_op = tf.losses.mean_squared_error(self.output_placeholder, self.layers['output'])
        
        self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss_op)

    def metrics(self):
        self.summary_loss = tf.summary.scalar('Loss', self.loss_op)

        summary_i = tf.summary.image('Input', self.visual['input'], max_outputs=11)
        summary_gt = tf.summary.image('Ground Truth', self.visual['ground_truth'], max_outputs=11)
        summary_p = tf.summary.image('Predicted', self.visual['predicted'], max_outputs=11)
        summary_po = tf.summary.image('Predicted with Overlay', self.visual['predicted_overlay'], max_outputs=11)

        self.summary_img = tf.summary.merge([summary_i, summary_gt, summary_p, summary_po])


    def run(self, sess, in_batch, out_batch, is_train, imgName):
        feed_dict = {
            self.is_train           : is_train,
            self.input_placeholder  : in_batch,
            self.output_placeholder : out_batch
        }

        if is_train:
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            
            _, _, summary_loss, loss, out_img = sess.run([self.train_op, extra_update_ops, self.summary_loss, 
                                                          self.loss_op, self.layers['output']], feed_dict=feed_dict)
            # out_img = out_img[0,:,:,:]
            # print(np.max(out_batch[0]))
            # print(np.min(out_batch[0]))
            # print(np.mean(out_batch[0]))
            # print('-'*10)
            # numpy2jpg(imgName.replace('.jpg','_noOverlay.jpg'), out_img, overlay=None, meanVal=1, verbose=False)
            # numpy2jpg(imgName.replace('.jpg','_overlay.jpg'), out_img, overlay=in_batch[0], meanVal=1, verbose=False)
            # numpy2jpg(imgName.replace('.jpg','_original.jpg'), in_batch[0], None, meanVal=LINE_MEAN, verbose=False)
            # exit(0)
        else:
            summary_loss, loss = sess.run([self.summary_loss, self.loss_op], feed_dict=feed_dict)

        if self.verbose:
            # print("Average accuracy per batch {0}".format(accuracy))
            print("Batch Loss: {0}".format(loss))

        return summary_loss, loss


    def sample(self, sess, in_batch=None, out_batch=None, imgName=None, out2board=False):
        if out2board:
            in_batch = self.SAMPLE_INPUT
            out_batch = self.SAMPLE_OUTPUT
            imgNames = [os.path.join(imgName, imgN) for imgN in self.SAMPLE_NAMES]

        feed_dict = {
            self.is_train           : False,
            self.input_placeholder  : in_batch,
            self.output_placeholder : out_batch
        }

        in_img, gt_img, pred_img, pred_overlay_img, summary_img, loss = sess.run([self.visual['input'], self.visual['ground_truth'], 
                                                                                  self.visual['predicted'], self.visual['predicted_overlay'], 
                                                                                  self.summary_img, self.loss_op], feed_dict=feed_dict)
        in_img = in_img[:,:,:,0]


        if imgName!=None:
            for iImg,gtImg,pImg,poImg,name in zip(in_img, gt_img, pred_img, pred_overlay_img, imgNames):
                print(name)
                print(name.replace('.jpg', '_input%d.jpg'%self.fnameMod))

                toimage(iImg, cmin=0, cmax=255).save(name.replace('.jpg', '_input%d.jpg'%self.fnameMod))
                toimage(gtImg, cmin=0, cmax=255).save(name.replace('.jpg', '_gt%d.jpg'%self.fnameMod))
                toimage(pImg, cmin=0, cmax=255).save(name.replace('.jpg', '_predicted%d.jpg'%self.fnameMod))
                toimage(poImg, cmin=0, cmax=255).save(name.replace('.jpg', '_overlay%d.jpg'%self.fnameMod))

        self.fnameMod += 1
        return summary_img
    
class ZhangNet(object):

    def __init__(self, input_shape, output_shape, verbose=True):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None]+self.input_shape, name="inputs")
        self.output_placeholder = tf.placeholder(dtype=tf.float32, shape=[None]+self.output_shape, name="outputs")
        self.is_train = tf.placeholder(dtype=tf.bool, name='is_train')
        
        self.config = ZhangNetConfig()
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

        # softmax loss prep
        self.layers['logits'] = currLayer
        self.layers['output'] = tf.nn.softmax(self.layers['logits'])

        self.add_loss_op()

        if self.verbose:
            print("The name of the output layer is: " + layerName)
            print("Built the Zhang Net Model.....")

    def add_loss_op(self):
        # softmax loss
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.layers['logits'],
                                                                              labels=self.output_placeholder))

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss_op)

    def metrics(self):
        # last_axis = len(self.probabilities_op.get_shape().as_list())
        # self.prediction_op = tf.to_int32(tf.argmax(self.probabilities_op, axis=last_axis-1))
        # difference = self.label_placeholder - self.prediction_op
        # zero = tf.constant(0, dtype=tf.int32)
        # boolean_difference = tf.cast(tf.equal(difference, zero), tf.float64)
        # self.accuracy_op = tf.reduce_mean(boolean_difference)

        self.summary_loss = tf.summary.scalar('Loss', self.loss_op)

    def run(self, sess, in_batch, out_batch, is_train):
        feed_dict = {
            self.is_train           : is_train,
            self.input_placeholder  : in_batch,
            self.output_placeholder : out_batch
        }

        if is_train:
            _, summary_loss, loss = sess.run([self.train_op, self.summary_loss, self.loss_op], feed_dict=feed_dict)
        else:
            summary_loss, loss = sess.run([self.summary_loss, self.loss_op], feed_dict=feed_dict)

        if self.verbose:
            # print("Average accuracy per batch {0}".format(accuracy))
            print("Batch Loss: {0}".format(loss))

        return summary_loss, loss


    def sample(self, sess, in_batch, imgName=None):
        feed_dict = {
            self.is_train           : False,
            self.input_placeholder  : in_batch
        }

        eps = 1e-4

        probs = sess.run([self.layers['output']], feed_dict=feed_dict)[0][0,:,:,:]
        logits = np.log(np.reshape(probs, ([(IMG_DIM/4)**2, 512])) + eps)
        
        unnormalized = np.exp((logits - np.max(logits, axis=1)[:,np.newaxis]) / TEMPERATURE)
        probabilities = unnormalized / np.sum(unnormalized, axis=1).astype(float)[:,np.newaxis]

        CLASS_MAP_R = np.asarray([32*i+16 for i in range(8)]*64)
        CLASS_MAP_G = np.asarray([32*int(i/8)+16 for i in range(64)]*8)
        CLASS_MAP_B = np.asarray([32*int(i/64)+16 for i in range(512)])
        
        out_img = np.stack((np.sum(CLASS_MAP_R * probabilities, axis=1), 
                            np.sum(CLASS_MAP_G * probabilities, axis=1), 
                            np.sum(CLASS_MAP_B * probabilities, axis=1)), axis=1)
        
        out_img = np.reshape(out_img, [IMG_DIM/4, IMG_DIM/4, 3])
        out_img = imresize(out_img, size=(in_batch[0].shape[0], in_batch[0].shape[1])).astype(float)

        if imgName!=None:
            print(imgName)
            numpy2jpg(imgName, out_img, overlay=in_batch[0], meanVal=None, verbose=False)
    
