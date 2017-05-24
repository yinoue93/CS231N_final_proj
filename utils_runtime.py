import pickle
import os
import random
import numpy as np
import shutil
import re
import datetime

import tensorflow as tf

from utils_misc import makeDir

def get_checkpoint(override, ckpt_dir, sess, saver):
    # Checkpoint
    found_ckpt = False

    if override:
        if tf.gfile.Exists(ckpt_dir):
            # tf.gfile.DeleteRecursively(ckpt_dir)
            shutil.rmtree(ckpt_dir, ignore_errors=True)
        tf.gfile.MakeDirs(ckpt_dir)

    # check if arags.ckpt_dir is a directory of checkpoints, or the checkpoint itself
    if len(re.findall('model.ckpt-[0-9]+', ckpt_dir)) == 0:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            i_stopped = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print("Found checkpoint for epoch ({0})".format(i_stopped))
            found_ckpt = True
        else:
            print('No checkpoint file found!')
            i_stopped = 0
    else:
        saver.restore(sess, ckpt_dir)
        i_stopped = int(ckpt_dir.split('/')[-1].split('-')[-1]) + 1
        print("Found checkpoint for epoch ({0})".format(i_stopped))
        found_ckpt = True

    return i_stopped, found_ckpt


def save_checkpoint(ckpt_dir, sess, saver, i):
    checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt')
    saver.save(sess, checkpoint_path, global_step=i)


def getDataFileNames(fileDir):
    # load the data directory correctly
    if os.path.isdir(fileDir):
        dataset_filenames = [os.path.join(fileDir, fname) for fname in os.listdir(fileDir)]
    else:
        dataset_filenames = [fileDir]

    return dataset_filenames


def getParamStr(param_key, param):
    param_str = ''
    for i in range(len(param)):
        param_str += param_key[i] + ': ' + str(param[i]) + ', '
    return param_str

def createLogFile():
    # create a log file and initialize it with the contents of constants.py
    makeDir('logs')
    now = datetime.datetime.now()
    fileName = 'logs/%s.txt' %(now.strftime("%B_%d_%H_%M_%S"))
    
    shutil.copyfile('constants.py', fileName)
    
    return fileName

def logToFile(logName, contents):
    print(contents)
    with open(logName, 'a+') as f:
        f.write(contents)

def map_output(outData):
    pass
    

if __name__ == "__main__":
    pass