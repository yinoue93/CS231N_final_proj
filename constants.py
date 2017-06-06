# multi processing constant
POOL_WORKER_COUNT = 1

# dataset image mean constants
REDUCED_R_MEAN = 167
REDUCED_G_MEAN = 153
REDUCED_B_MEAN = 147
LINE_MEAN = 219
BINARY_MEAN = 225

CHROMA_MAX = 100

# network constants
BATCH_SIZE = 24
IMG_DIM = 256
NUM_EPOCHS = 10
NUM_EPOCHS_HYPERPARAM = 2
LOSS_MIX_TERM = 1
USEFUL_DATA_DIR = 'useful_data/'

CLASS_IMBALANCE_FILE = USEFUL_DATA_DIR + 'class_imbalance.npy'
CLASS_IMBALANCE_FILE_LCH = USEFUL_DATA_DIR + 'class_imbalance_lch.npy'
CHROMA_MATRIX_FILE = USEFUL_DATA_DIR + 'chroma_matrix.npy'

# dataset constants
DIR_MODIFIER = '../data'
TRAIN_DATA = DIR_MODIFIER + '/home/tbonerocksyinoue/data/line_classification'
TEST_DATA = DIR_MODIFIER + '/small'
VALIDATION_DATA = DIR_MODIFIER + '/small'

CKPT_DIR_DEFAULT = DIR_MODIFIER + '/temp_ckpt'
SUMMARY_DIR = DIR_MODIFIER + '/dev_summary2'

SAVE_CKPT_COUNTER = 1
DATA_LOAD_PARTITION = 8

# sampling constants
TEMPERATURE = 0.38
SAMPLE_DATA_FILE = USEFUL_DATA_DIR + 'sample_data_line'
SAMPLE_DATA_FILE_CLASSIFICATION = USEFUL_DATA_DIR + 'sample_data_line_classification'
SAMPLE_DATA_FILE_CLASSIFICATION_LCH = USEFUL_DATA_DIR + 'sample_data_line_classification_lch'
SAMPLE_OUT_DIR = DIR_MODIFIER + '/sample_out_imgs'

# test image stuff
TEST_IMG_NAMES = []

# PAPER_IMG_NAMES = ['1239646_0', '1239647_0', '1239603_0', '1239585_0', '1239499_0', '1239616_0',
#                    '1239663_1', '1239669_0', '1239691_0', '1239684_0', '1239658_0', '1239554_0',
#                    '1239512_0', '1239506_0', '1239493_1', '1239516_1', '1239514_0', '1239558_0']
# PAPER_IMG_NAMES = ['1239479_1', '1239481_0', '1239485_0', '1239489_0', '1239493_0', 
#                    '1239495_1', '1239499_0', '1239506_3', '1239506_6', '1239510_0', 
#                    '1239512_0', '1239517_0', '1239519_0', '1239520_1', '1239522_1', 
#                    '1239523_0', '1239533_1', '1239534_0']
PAPER_IMG_NAMES = ['1239479_0', '1239479_1', '1239481_0', '1239481_1', '1239485_0', '1239489_0', '1239489_1', '1239493_0', '1239493_1', '1239495_0', 
                   '1239495_1', '1239497_0', '1239497_1', '1239499_0', '1239499_1', '1239506_0', '1239506_1', '1239506_2', '1239506_3', '1239506_4', 
                   '1239506_5', '1239506_6', '1239508_0', '1239508_1', '1239509_0', '1239509_1', '1239510_0', '1239510_1', '1239512_0', '1239512_1', 
                   '1239513_0', '1239513_1', '1239514_0', '1239514_1', '1239516_0', '1239516_1', '1239517_0', '1239517_1', '1239519_0', '1239519_1', 
                   '1239520_0', '1239520_1', '1239521_0', '1239522_0', '1239522_1', '1239523_0', '1239523_1', '1239528_0', '1239528_1', '1239529_0', 
                   '1239529_1', '1239532_0', '1239532_1', '1239533_0', '1239533_1', '1239534_0', '1239536_0', '1239536_1', '1239537_0', '1239537_1', 
                   '1239537_2', '1239541_0', '1239541_1', '1239544_0', '1239544_1', '1239545_0', '1239546_0', '1239546_1', '1239548_0', '1239548_1', 
                   '1239552_0', '1239552_1', '1239553_0', '1239553_1', '1239554_0', '1239554_1', '1239555_0', '1239555_1', '1239557_0', '1239557_1', 
                   '1239558_0', '1239559_0', '1239559_1', '1239560_0', '1239560_1', '1239561_0', '1239561_1', '1239562_0', '1239562_1', '1239563_0', 
                   '1239564_0', '1239564_1', '1239565_0', '1239565_1', '1239567_0', '1239568_0', '1239568_1', '1239569_0', '1239569_1', '1239570_0', 
                   '1239570_1', '1239571_0', '1239571_1', '1239572_0', '1239572_1', '1239574_0', '1239574_1', '1239575_0', '1239575_1', '1239576_0', 
                   '1239576_1', '1239577_0', '1239577_1', '1239579_0', '1239580_0', '1239580_1', '1239581_0', '1239581_1', '1239583_0', '1239583_1', 
                   '1239585_0', '1239585_1', '1239586_0', '1239586_1', '1239587_0', '1239587_1', '1239588_0', '1239589_0', '1239589_1', '1239590_0', 
                   '1239590_1', '1239593_0', '1239593_1', '1239594_0', '1239594_1', '1239595_0', '1239595_1', '1239596_0', '1239596_1', '1239599_0', 
                   '1239599_1', '1239601_0', '1239601_1', '1239602_0', '1239602_1', '1239603_0', '1239603_1', '1239604_0', '1239604_1', '1239605_0', 
                   '1239605_1', '1239607_0', '1239607_1', '1239608_0', '1239608_1', '1239610_0', '1239610_1', '1239611_0', '1239612_0', '1239612_1', 
                   '1239614_0', '1239614_1', '1239615_0', '1239615_1', '1239616_0', '1239616_1', '1239617_0', '1239617_1', '1239618_0', '1239618_1', 
                   '1239619_0', '1239619_1', '1239621_0', '1239621_1', '1239623_0', '1239624_0', '1239624_1', '1239628_0', '1239629_0', '1239629_1', 
                   '1239630_0', '1239630_1', '1239633_0', '1239633_1', '1239634_0', '1239634_1', '1239636_0', '1239636_1', '1239640_0', '1239640_1', 
                   '1239641_0', '1239641_1', '1239644_0', '1239644_1', '1239645_0', '1239645_1', '1239646_0', '1239646_1', '1239647_0', '1239647_1', 
                   '1239653_0', '1239653_1', '1239655_0', '1239655_1', '1239656_0', '1239656_1', '1239658_0', '1239658_1', '1239660_0', '1239660_1', 
                   '1239662_0', '1239662_1', '1239663_0', '1239663_1', '1239665_0', '1239666_0', '1239666_1', '1239667_0', '1239667_1', '1239669_0', 
                   '1239669_1', '1239670_0', '1239670_1', '1239672_0', '1239674_0', '1239674_1', '1239675_0', '1239675_1', '1239676_0', '1239677_0', 
                   '1239677_1', '1239680_0', '1239680_1', '1239681_0', '1239681_1', '1239682_0', '1239682_1', '1239684_0', '1239684_1', '1239685_0', 
                   '1239685_1', '1239686_0', '1239686_1', '1239687_0', '1239687_1', '1239688_0', '1239688_1', '1239690_0', '1239690_1', '1239691_0']
NUM_SAMPLES = 16 if PAPER_IMG_NAMES==None else len(PAPER_IMG_NAMES)

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
        self.lr = 1e-4

        self.use_class_imbalance = True
        self.color_space = 'rgb'

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
                
                
                