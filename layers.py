
import tensorflow as tf

#-----------------------Convolution Layers Implementations-------------------------

def conv_layer(inLayer, numC, filterSz, stride, dilation, name=None):
    dil = (1,1) if dilation==None else (dilation, dilation)
    conv_out = tf.layers.conv2d(inLayer, filters=numC, kernel_size=filterSz, strides=(stride,stride), 
                                padding='SAME', activation=tf.nn.relu, dilation_rate=dil, name=name)
    return conv_out

def deconv_layer(inLayer, numC, filterSz, stride, name=None):
    deconv_out = tf.layers.conv2d_transpose(inLayer, filters=numC, kernel_size=filterSz,
                                            strides=[stride, stride], padding='SAME',
                                            activation=tf.nn.relu)
    return deconv_out

def conv_relu(inLayer, layerParams, is_train, name=None, verbose=True):
    # The order is "NHWC"
    _,H,W,last_C = inLayer.get_shape().as_list()

    filterSz, numFilters, stride, bnorm, dilation = layerParams['filterSz'], layerParams['numFilters'], \
                                                    layerParams['stride'], layerParams['bnorm'], layerParams['dilation']

    # choose the correct convolution layer
    if stride>=1:
        # normal conv layer
        convLayer = conv_layer(inLayer, numFilters, filterSz, stride, dilation, name=name)
    else:
        # deconv (transpose, fractionally strided) layer
        convLayer = deconv_layer(inLayer, numFilters, filterSz, int(1/stride), name=name)

    if bnorm:
        convLayer = tf.layers.batch_normalization(convLayer, center=True, scale=True, training=is_train)

    if verbose:
        print('Layer name:' + name)
        print(layerParams)
        print('Layer shape: {0}\n'.format(convLayer.shape))

    return convLayer

#---------------------------Batchnorm Implementation-------------------------
def batch_norm(prevLayer, axes, is_train, var_eps=1e-6, name=''):
    def update_bnorm():
        ema_apply_op = ema.apply([mu, sigma])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(mu), tf.identity(sigma)

    outShape = prevLayer.get_shape().as_list()[-1]

    mu, sigma = tf.nn.moments(prevLayer, axes)
    beta = tf.get_variable(name+'_beta', shape=outShape, initializer=tf.contrib.layers.xavier_initializer())
    scale = tf.get_variable(name+'_scale', shape=outShape, initializer=tf.contrib.layers.xavier_initializer())

    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    mean, var = tf.cond(is_train,
                        update_bnorm,
                        lambda: (ema.average(mu), ema.average(sigma)))

    next_layer = tf.nn.batch_normalization(prevLayer, mean, var, beta, scale, var_eps, name=name)
    return next_layer


