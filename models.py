
import tensorflow as tf
import numpy as np
import os

from constants import *
from layers import *
from utils_misc import numpy2jpg,h52numpy
from scipy.misc import imresize,toimage

class Model(object):

    def __init__(self, input_shape, output_shape, verbose=True):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None]+self.input_shape, name="inputs")
        self.output_placeholder = tf.placeholder(dtype=tf.float32, shape=[None]+self.output_shape, name="outputs")
        self.is_train = tf.placeholder(dtype=tf.bool, name='is_train')
        
        self.layers = {}
        self.params = {}
        self.visual = {}

        self.fnameMod = 0

        self.verbose = verbose

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
            
            _, _, summary_loss, loss = sess.run([self.train_op, extra_update_ops, self.summary_loss, 
                                                 self.loss_op], feed_dict=feed_dict)
        else:
            summary_loss, loss = sess.run([self.summary_loss, self.loss_op], feed_dict=feed_dict)

        if self.verbose:
            #print("Batch Loss: {0}".format(loss))
            pass

        return summary_loss, loss


    def sample(self, sess, in_batch=None, out_batch=None, imgName=None, out2board=False):
        if out2board:
            in_batch = self.SAMPLE_INPUT
            out_batch = self.SAMPLE_OUTPUT
            imgName = [os.path.join(imgName, imgN) for imgN in self.SAMPLE_NAMES]

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
            for iImg,gtImg,pImg,poImg,name in zip(in_img, gt_img, pred_img, pred_overlay_img, imgName):
                toimage(iImg, cmin=0, cmax=255).save(name.replace('.jpg', '_input%d.jpg'%self.fnameMod))
                toimage(gtImg, cmin=0, cmax=255).save(name.replace('.jpg', '_gt%d.jpg'%self.fnameMod))
                toimage(pImg, cmin=0, cmax=255).save(name.replace('.jpg', '_predicted%d.jpg'%self.fnameMod))
                toimage(poImg, cmin=0, cmax=255).save(name.replace('.jpg', '_overlay%d.jpg'%self.fnameMod))

        self.fnameMod += 1
        return summary_img


class Unet(Model):

    def __init__(self, input_shape, output_shape, verbose=True):
        super().__init__(input_shape, output_shape, verbose=verbose)

        # load sample dataset
        self.SAMPLE_INPUT, self.SAMPLE_OUTPUT, self.SAMPLE_NAMES = h52numpy(SAMPLE_DATA_FILE, checkMean=False, batch_sz=11, 
                                                                            mod_output=False, iter_val=None, shuffle=False)

        self.config = UnetConfig()

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

    
class ZhangNet(Model):

    def __init__(self, input_shape, output_shape, verbose=True):
        super().__init__(input_shape, output_shape, verbose=verbose)

        # load sample dataset
        self.SAMPLE_INPUT, self.SAMPLE_OUTPUT, self.SAMPLE_NAMES = h52numpy(SAMPLE_DATA_FILE_CLASSIFICATION, checkMean=False, batch_sz=11, 
                                                                            mod_output=True, iter_val=None, shuffle=False)

        self.config = ZhangNetConfig()

        self.CLASS_MAP_R = tf.constant(np.asarray([32*i+16 for i in range(8)]*64), dtype=tf.float32)
        self.CLASS_MAP_G = tf.constant(np.asarray([32*int(i/8)+16 for i in range(64)]*8), dtype=tf.float32)
        self.CLASS_MAP_B = tf.constant(np.asarray([32*int(i/64)+16 for i in range(512)]), dtype=tf.float32)

        if self.verbose:
            print("Completed Initializing the Zhang Net Model.....")

    def prob2img(self, probTensor):
        # converts probability distribution to an image
        eps = 1e-4

        out_img_dim = int(IMG_DIM/4)

        batch_sz = tf.shape(probTensor)[0]
        logits = tf.log(tf.reshape(probTensor, shape=[batch_sz, out_img_dim**2, 512]) + eps)

        unnormalized = tf.exp((logits - tf.reduce_max(logits, axis=2, keep_dims=True)) / TEMPERATURE)
        probabilities = unnormalized / tf.reduce_sum(unnormalized, axis=2, keep_dims=True)

        out_img = tf.stack((tf.reduce_sum(self.CLASS_MAP_R * probabilities, axis=2), 
                            tf.reduce_sum(self.CLASS_MAP_G * probabilities, axis=2), 
                            tf.reduce_sum(self.CLASS_MAP_B * probabilities, axis=2)), axis=2)
        
        out_img = tf.reshape(out_img, shape=[batch_sz, out_img_dim, out_img_dim, 3])
        out_img = tf.image.resize_images(out_img, size=[IMG_DIM,IMG_DIM], method=tf.image.ResizeMethod.BILINEAR)

        return out_img

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

        # add tensors for visual purposes
        self.visual['input'] = self.input_placeholder + LINE_MEAN
        self.visual['ground_truth'] = self.prob2img(self.output_placeholder)
        self.visual['predicted'] = self.prob2img(self.layers['output'])
        self.visual['predicted_overlay'] = self.visual['predicted'] * (self.visual['input']/255.0)

        self.add_loss_op()

        if self.verbose:
            print("The name of the output layer is: " + layerName)
            print("Built the Zhang Net Model.....")

    def add_loss_op(self):
        xloss = tf.nn.softmax_cross_entropy_with_logits(logits=self.layers['logits'], labels=self.output_placeholder)

        if self.config.use_class_imbalance:
            class_weights = tf.constant(np.load(CLASS_IMBALANCE_FILE), dtype=tf.float32)
            max_indx = tf.argmax(self.output_placeholder, axis=2)

            weight_vec = tf.gather(params=class_weights, indices=max_indx)
            xloss = tf.reshape(xloss, shape=[tf.shape(xloss)[0], int((IMG_DIM)/4)**2])

            self.loss_op = tf.reduce_mean(tf.multiply(xloss, weight_vec))
        else:
            self.loss_op = tf.reduce_mean(xloss)

        self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss_op)
    
