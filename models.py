
import tensorflow as tf
import numpy as np
import os

from constants import *
from layers import *
from utils_misc import numpy2jpg,h52numpy,lch2rgb_batch
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
            self.input_placeholder  : in_batch
        }

        if out_batch is None:
            in_img, pred_img, pred_overlay_img = sess.run([self.visual['input'], self.visual['predicted'], 
                                                           self.visual['predicted_overlay']], feed_dict=feed_dict)
            summary_img = None
        else:
            feed_dict[self.output_placeholder] = out_batch
            in_img, gt_img, pred_img, pred_overlay_img, \
                    summary_img, loss = sess.run([self.visual['input'], self.visual['ground_truth'], 
                                                  self.visual['predicted'], self.visual['predicted_overlay'], 
                                                  self.summary_img, self.loss_op], feed_dict=feed_dict)


        in_img = in_img[:,:,:,0]

        if imgName!=None:
            for i in range(len(in_img)):
                name = imgName[i]
                toimage(in_img[i], cmin=0, cmax=255).save(name.replace('.jpg', '_input%d.jpg'%self.fnameMod))
                toimage(pred_img[i], cmin=0, cmax=255).save(name.replace('.jpg', '_predicted%d.jpg'%self.fnameMod))
                toimage(pred_overlay_img[i], cmin=0, cmax=255).save(name.replace('.jpg', '_overlay%d.jpg'%self.fnameMod))
                if out_batch is not None:
                    toimage(gt_img[i], cmin=0, cmax=255).save(name.replace('.jpg', '_gt%d.jpg'%self.fnameMod))

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

        self.config = ZhangNetConfig()
        
        # load sample dataset
        sample_data_dir = SAMPLE_DATA_FILE_CLASSIFICATION if self.config.color_space=='rgb' else SAMPLE_DATA_FILE_CLASSIFICATION_LCH
        self.SAMPLE_INPUT, self.SAMPLE_OUTPUT, self.SAMPLE_NAMES = h52numpy(sample_data_dir, checkMean=False, batch_sz=11, 
                                                                            mod_output=True, iter_val=None, shuffle=False)

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

        if self.config.color_space=='rgb':
            out_img = tf.stack((tf.reduce_sum(self.CLASS_MAP_R * probabilities, axis=2), 
                                tf.reduce_sum(self.CLASS_MAP_G * probabilities, axis=2), 
                                tf.reduce_sum(self.CLASS_MAP_B * probabilities, axis=2)), axis=2)
            
            out_img = tf.reshape(out_img, shape=[batch_sz, out_img_dim, out_img_dim, 3])

        elif self.config.color_space=='lch':
            # convert from 255 scale to the appropriate scale for lch, and then convert to rgb
            # treat H differently because it is a circular value (ref: https://en.wikipedia.org/wiki/Mean_of_circular_quantities)
            hAngles = self.CLASS_MAP_B * (2*np.pi)/255.0
            H = atan2(tf.reduce_sum(tf.cos(hAngles) * probabilities, axis=2), 
                      tf.reduce_sum(tf.sin(hAngles) * probabilities, axis=2))

            out_img = tf.stack((tf.reduce_sum(self.CLASS_MAP_R * probabilities, axis=2) * 100/255.0, 
                                tf.reduce_sum(self.CLASS_MAP_G * probabilities, axis=2) * CHROMA_MAX/255.0, 
                                H), axis=2)

            out_img = tf.reshape(out_img, shape=[batch_sz, out_img_dim, out_img_dim, 3])

            out_img = tf.reshape(tf.py_func(lch2rgb_batch, [out_img], Tout=tf.float32), shape=tf.shape(out_img))

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
            imbalance_fname = CLASS_IMBALANCE_FILE if self.config.color_space=='rgb' else CLASS_IMBALANCE_FILE_LCH
            class_weights = tf.constant(np.load(imbalance_fname), dtype=tf.float32)
            max_indx = tf.argmax(self.output_placeholder, axis=2)

            weight_vec = tf.gather(params=class_weights, indices=max_indx)
            xloss = tf.reshape(xloss, shape=[tf.shape(xloss)[0], int((IMG_DIM)/4)**2])

            chroma_weights = tf.constant(np.load(CHROMA_MATRIX_FILE), dtype=tf.float32)
            outPShape = tf.shape(self.output_placeholder)
            outShape = tf.shape(self.layers['output'])
            outputP_reshaped = tf.reshape(self.output_placeholder, shape=[outPShape[0]*outPShape[1], outPShape[2]])
            output_reshaped = tf.reshape(self.layers['output'], shape=[outShape[0]*outShape[1]*outShape[2], outShape[3]])
            
            chroma_loss = tf.reduce_sum(tf.matmul(outputP_reshaped, chroma_weights) * output_reshaped, axis=1)

            total_loss = xloss + LOSS_MIX_TERM*chroma_loss
            self.loss_op = tf.reduce_mean(tf.multiply(total_loss, weight_vec))
        else:
            self.loss_op = tf.reduce_mean(xloss)

        self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss_op)
    
