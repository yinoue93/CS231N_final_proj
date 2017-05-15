
import tensorflow as tf

#-----------------------Convolution Layers Implementations-------------------------

def conv_layer(inLayer, filterSz, last_C, curr_C, stride, name=None):
    filterVar = tf.get_variable(name=name+'_weight', shape=[filterSz, filterSz, last_C, curr_C],
                                initializer=tf.contrib.layers.xavier_initializer())
    strides = [1, stride, stride, 1]

    convLayer = tf.nn.conv2d(inLayer, filter=filterVar, strides=strides, padding='SAME')

    return convLayer, filterVar

def atrous_layer(inLayer, filterSz, last_C, curr_C, stride, dilation, name=None):
    filterVar = tf.get_variable(name=name+'_weight', shape=[filterSz, filterSz, last_C, curr_C],
                                initializer=tf.contrib.layers.xavier_initializer())
    strides = [1, stride, stride, 1]

    inLayerShape = inLayer.get_shape().as_list()
    output_shape = [tf.shape(inLayer)[0], inLayerShape[1], inLayerShape[2], curr_C]

    convLayer = tf.nn.atrous_conv2d(inLayer, filters=filterVar, rate=dilation, padding='SAME')
    convLayer = tf.reshape(convLayer, shape=output_shape)

    return convLayer, filterVar

def deconv_layer(inLayer, filterSz, last_C, curr_C, H, W, stride, name=None):
    filterVar = tf.get_variable(name=name+'_weight', shape=[filterSz, filterSz, curr_C, last_C],
                                initializer=tf.contrib.layers.xavier_initializer())
    expansion_rate = int(1/stride)
    strides = [1, expansion_rate, expansion_rate, 1]

    inLayerShape = inLayer.get_shape().as_list()
    output_shape = [tf.shape(inLayer)[0], inLayerShape[1]*expansion_rate, inLayerShape[2]*expansion_rate, curr_C]

    convLayer = tf.nn.conv2d_transpose(inLayer, filter=filterVar, output_shape=output_shape, 
                                       strides=strides, padding='SAME')
    convLayer = tf.reshape(convLayer, shape=output_shape)

    return convLayer, filterVar

def conv_relu(inLayer, layerParams, is_train, name=None, verbose=True):
    # The order is "NHWC"
    _,H,W,last_C = inLayer.get_shape().as_list()

    filterSz, numFilters, stride, bnorm, dilation = layerParams['filterSz'], layerParams['numFilters'], \
                                                    layerParams['stride'], layerParams['bnorm'], layerParams['dilation']

    # choose the correct convolution layer
    if dilation is not None:
        # atrous (dilation) conv layer
        convLayer,W = atrous_layer(inLayer, filterSz, last_C, numFilters, stride, dilation, name=name)
    elif stride<1:
        # deconv (transpose, fractionally strided) layer
        convLayer,W = deconv_layer(inLayer, filterSz, last_C, numFilters, H, W, stride, name=name)
    else:
        # normal conv layer
        convLayer,W = conv_layer(inLayer, filterSz, last_C, numFilters, stride, name=name)

    # add the bias term
    bias = tf.get_variable(name=name+'_bias', shape=numFilters)
    convLayer = tf.nn.bias_add(convLayer, bias, name=name)

    if bnorm:
        convLayer = batch_norm(convLayer, [0,1,2], is_train, name=name+'_bnorm')

    nextLayer = tf.nn.relu(convLayer)

    if verbose:
        print('Layer name:' + name)
        print(layerParams)
        print('Layer shape: {0}\n'.format(nextLayer.shape))

    return nextLayer, W, bias

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


