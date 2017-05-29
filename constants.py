# multi processing constant
POOL_WORKER_COUNT = 1

# dataset image mean constants
REDUCED_R_MEAN = 167
REDUCED_G_MEAN = 153
REDUCED_B_MEAN = 147
LINE_MEAN = 219
BINARY_MEAN = 225

# network constants
BATCH_SIZE = 4
IMG_DIM = 256
NUM_EPOCHS = 10
NUM_EPOCHS_HYPERPARAM = 2

# dataset constants
DIR_MODIFIER = '../data'
TRAIN_DATA = DIR_MODIFIER + '/home/tbonerocksyinoue/data/line_classification'
TEST_DATA = DIR_MODIFIER + '/small'
VALIDATION_DATA = DIR_MODIFIER + '/small'

CKPT_DIR_DEFAULT = DIR_MODIFIER + '/temp_ckpt'
SAMPLE_DATA_FILE = 'sample_data_line'
SAMPLE_DATA_FILE_CLASSIFICATION = 'sample_data_line_classification'
SAMPLE_OUT_DIR = DIR_MODIFIER + '/sample_out_imgs'
SUMMARY_DIR = DIR_MODIFIER + '/dev_summary2'

SAVE_CKPT_COUNTER = 1
DATA_LOAD_PARTITION = 4

# sampling constants
TEMPERATURE = 0.38

# test image stuff
TEST_IMG_NAMES = []

NUM_SAMPLES = 16

class UnetConfig(object):

    def setDefault(self, param_dict, param_name, val):
        if param_name not in param_dict:
            param_dict[param_name] = val

    def __init__(self):
        # optimizer params
        self.lr = 1e-3

        self.use_fuse = True

        # network params
        self.layer_keys = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
                           'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 
                           'conv6_1', 'conv6_2', 'conv6_3', 'conv7_1', 'conv7_2', 'conv7_3', 
                           'conv8_1', 'conv8_2', 'conv8_3', 'conv9_1', 'conv9_2', 'conv9_3', 
                           'conv10_1', 'conv10_2', 'conv10_3', 'combine_1', 'combine_2', 'combine_3']

        self.fuse_layers = {'conv8_1':'conv3_2', 'conv9_1':'conv2_1', 'conv10_1':'conv1_1'}

        self.layer_params = {}

        self.layer_params['conv1_1'] = {'numFilters':64, 'filterSz':3}
        self.layer_params['conv1_2'] = {'bnorm':True, 'numFilters':64, 'filterSz':3, 'stride':2}

        self.layer_params['conv2_1'] = {'numFilters':128, 'filterSz':3}
        self.layer_params['conv2_2'] = {'bnorm':True, 'numFilters':128, 'filterSz':3, 'stride':2}

        self.layer_params['conv3_1'] = {'numFilters':256, 'filterSz':3}
        self.layer_params['conv3_2'] = {'numFilters':256, 'filterSz':3}
        self.layer_params['conv3_3'] = {'bnorm':True, 'numFilters':256, 'filterSz':3, 'stride':2}

        self.layer_params['conv4_1'] = {'numFilters':512, 'filterSz':3}
        self.layer_params['conv4_2'] = {'numFilters':512, 'filterSz':3}
        self.layer_params['conv4_3'] = {'bnorm':True, 'numFilters':512, 'filterSz':3}

        self.layer_params['conv5_1'] = {'numFilters':512, 'filterSz':3, 'dilation':2}
        self.layer_params['conv5_2'] = {'numFilters':512, 'filterSz':3, 'dilation':2}
        self.layer_params['conv5_3'] = {'bnorm':True, 'numFilters':512, 'filterSz':3, 'dilation':2}

        self.layer_params['conv6_1'] = {'numFilters':512, 'filterSz':3, 'dilation':2}
        self.layer_params['conv6_2'] = {'numFilters':512, 'filterSz':3, 'dilation':2}
        self.layer_params['conv6_3'] = {'bnorm':True, 'numFilters':512, 'filterSz':3, 'dilation':2}

        self.layer_params['conv7_1'] = {'numFilters':512, 'filterSz':3}
        self.layer_params['conv7_2'] = {'numFilters':512, 'filterSz':3}
        self.layer_params['conv7_3'] = {'bnorm':True, 'numFilters':512, 'filterSz':3}

        self.layer_params['conv8_1'] = {'numFilters':256, 'filterSz':4, 'stride':0.5}
        self.layer_params['conv8_2'] = {'numFilters':256, 'filterSz':3}
        self.layer_params['conv8_3'] = {'bnorm':True, 'numFilters':256, 'filterSz':3}

        self.layer_params['conv9_1'] = {'numFilters':128, 'filterSz':4, 'stride':0.5}
        self.layer_params['conv9_2'] = {'numFilters':128, 'filterSz':3}
        self.layer_params['conv9_3'] = {'bnorm':True, 'numFilters':128, 'filterSz':3}

        self.layer_params['conv10_1'] = {'numFilters':64, 'filterSz':4, 'stride':0.5}
        self.layer_params['conv10_2'] = {'numFilters':64, 'filterSz':3}
        self.layer_params['conv10_3'] = {'bnorm':True, 'numFilters':64, 'filterSz':3}

        self.layer_params['combine_1'] = {'bnorm':True, 'numFilters':32, 'filterSz':3}
        self.layer_params['combine_2'] = {'bnorm':True, 'numFilters':16, 'filterSz':3}
        self.layer_params['combine_3'] = {'numFilters':3, 'filterSz':3}

        # set default layer parameters
        for _,layerParam in self.layer_params.items():
            self.setDefault(layerParam, 'bnorm', False)
            self.setDefault(layerParam, 'stride', 1)
            self.setDefault(layerParam, 'dilation', None)

            
class ZhangNetConfig(object):

    def setDefault(self, param_dict, param_name, val):
        if param_name not in param_dict:
            param_dict[param_name] = val

    def __init__(self):
        # optimizer params
        self.lr = 1e-3

        # network params
        self.layer_keys = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
                           'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 
                           'conv6_1', 'conv6_2', 'conv6_3', 'conv7_1', 'conv7_2', 'conv7_3', 
                           'conv8_1', 'conv8_2', 'conv8_3', 'conv8_313']

        self.layer_params = {}

        self.layer_params['conv1_1'] = {'numFilters':64, 'filterSz':3}
        self.layer_params['conv1_2'] = {'bnorm':True, 'numFilters':64, 'filterSz':3, 'stride':2}

        self.layer_params['conv2_1'] = {'numFilters':128, 'filterSz':3}
        self.layer_params['conv2_2'] = {'bnorm':True, 'numFilters':128, 'filterSz':3, 'stride':2}

        self.layer_params['conv3_1'] = {'numFilters':256, 'filterSz':3}
        self.layer_params['conv3_2'] = {'numFilters':256, 'filterSz':3}
        self.layer_params['conv3_3'] = {'bnorm':True, 'numFilters':256, 'filterSz':3, 'stride':2}

        self.layer_params['conv4_1'] = {'numFilters':512, 'filterSz':3}
        self.layer_params['conv4_2'] = {'numFilters':512, 'filterSz':3}
        self.layer_params['conv4_3'] = {'bnorm':True, 'numFilters':512, 'filterSz':3}

        self.layer_params['conv5_1'] = {'numFilters':512, 'filterSz':3, 'dilation':2}
        self.layer_params['conv5_2'] = {'numFilters':512, 'filterSz':3, 'dilation':2}
        self.layer_params['conv5_3'] = {'bnorm':True, 'numFilters':512, 'filterSz':3, 'dilation':2}

        self.layer_params['conv6_1'] = {'numFilters':512, 'filterSz':3, 'dilation':2}
        self.layer_params['conv6_2'] = {'numFilters':512, 'filterSz':3, 'dilation':2}
        self.layer_params['conv6_3'] = {'bnorm':True, 'numFilters':512, 'filterSz':3, 'dilation':2}

        self.layer_params['conv7_1'] = {'numFilters':512, 'filterSz':3}
        self.layer_params['conv7_2'] = {'numFilters':512, 'filterSz':3}
        self.layer_params['conv7_3'] = {'bnorm':True, 'numFilters':512, 'filterSz':3}

        self.layer_params['conv8_1'] = {'numFilters':256, 'filterSz':4, 'stride':0.5}
        self.layer_params['conv8_2'] = {'numFilters':256, 'filterSz':3}
        self.layer_params['conv8_3'] = {'numFilters':256, 'filterSz':3}

        self.layer_params['conv8_313'] = {'numFilters':512, 'filterSz':1}

        # set default layer parameters
        for _,layerParam in self.layer_params.items():
            self.setDefault(layerParam, 'bnorm', False)
            self.setDefault(layerParam, 'stride', 1)
            self.setDefault(layerParam, 'dilation', None)
                
                
                