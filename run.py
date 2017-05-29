import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import os
import random
import progressbar

from utils_runtime import *
from utils_misc import *

from models import Unet,ZhangNet
from argparse import ArgumentParser

from constants import *

# GPU settings
GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.9


def run_model(modelStr, runMode, ckptDir, dataDir, sampleDir, overrideCkpt, numEpochs):
    print('Running model...')

    # choose the correct dataset
    if dataDir=='':
        if runMode=='train':
            dataDir = TRAIN_DATA
        elif runMode=='test':
            dataDir = TEST_DATA
        elif runMode=='val':
            dataDir = VALIDATION_DATA

    if not os.path.exists(dataDir):
        print('Please specify a valid data directory, "%s" is not a valid directory. Exiting...' % dataDir)
        return
    else:
        print('Using dataset %s' % dataDir)

    print("Using checkpoint directory: {0}".format(ckptDir))

    is_training = (runMode=='train')
    batch_size = 1 if runMode=='sample' else BATCH_SIZE
    numEpochs = numEpochs if is_training else 1
    overrideCkpt = overrideCkpt if is_training else False

    printSeparator('Initializing %s/reading constants.py' % modelStr)
    input_sz = [IMG_DIM, IMG_DIM, 1]
    if modelStr=='unet':
        output_sz = [IMG_DIM, IMG_DIM, 3]
        curModel = Unet(input_sz, output_sz)
    if modelStr=='zhangnet':
        output_sz = [(IMG_DIM/4)**2, 512]
        curModel = ZhangNet(input_sz, output_sz)
        
    printSeparator('Building ' + modelStr)
    curModel.create_model()
    curModel.metrics()

    print("Running {0} model for {1} epochs.".format(modelStr, numEpochs))

    print("Reading in {0}-set filenames.".format(runMode))

    global_step = tf.Variable(0, trainable=False, name='global_step') #tf.contrib.framework.get_or_create_global_step()
    saver = tf.train.Saver(max_to_keep=numEpochs)
    step = 0
    counter = 0
    
    if runMode=='sample':
        logDir,logName = None,None
    else:
        logDir,logName = createLog(runMode)

    printSeparator('Starting TF session')
    with tf.Session(config=GPU_CONFIG) as sess:
        print("Inititialized TF Session!")

        # load checkpoint if necessary
        i_stopped, found_ckpt = get_checkpoint(overrideCkpt, ckptDir, sess, saver)

        if runMode!='sample':
            file_writer = tf.summary.FileWriter(logDir, graph=sess.graph, max_queue=10, flush_secs=30)

        if (not found_ckpt):
            if is_training:
                init_op = tf.global_variables_initializer() # tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
                init_op.run()
            else:
                # Exit if no checkpoint to test]
                print('Valid checkpoint not found under %s, exiting...' % ckptDir)
                return
           
        if not is_training:
            numEpochs = i_stopped + 1

        # run the network
        if is_training:
            dataset_filenames = getDataFileNames(dataDir, excludeFnames=['.filepart', 'test'])
        else:
            dataset_filenames = getDataFileNames(dataDir, excludeFnames=['.filepart'])
        for epochCounter in range(i_stopped, numEpochs):
            batch_loss = []
            printSeparator("Running epoch %d" % epochCounter)
            random.shuffle(dataset_filenames)

            for j, data_file in enumerate(dataset_filenames):
                mini_loss = []
                for iter_val in range(DATA_LOAD_PARTITION):
                    # Get data
                    print('Reading data in %s, iter_val: %d...' % (data_file, iter_val))
                    try:
                        input_batches,output_batches,imgNames = h52numpy(data_file, batch_sz=batch_size, iter_val=iter_val, 
                                                                         mod_output=(modelStr=='zhangnet'))
                    except:
                        logToFile(logName, "File reading failed...")
                        continue
                    print('Done reading, running the network (%d of %d)' % (j+1, len(dataset_filenames)))

                    bar = progressbar.ProgressBar(maxval=int(len(input_batches)/batch_size))
                    bar.start()
                    count = 0
                    for dataIndx in range(0, len(imgNames), batch_size):
                        in_batch = input_batches[dataIndx:dataIndx+batch_size]
                        out_batch = output_batches[dataIndx:dataIndx+batch_size]
                        imgName = imgNames[dataIndx:dataIndx+batch_size]
                        
                        # look at the images in the dataset (for debug usage)
                        #for kk in range(batch_size):
                        #    numpy2jpg('tmp'+str(kk+dataIndx)+'.jpg', in_batch[kk,:,:,0], overlay=None, meanVal=LINE_MEAN, verbose=False)
                        #    numpy2jpg('KAK'+str(kk+dataIndx)+'.jpg', out_batch[kk,:,:], overlay=None, meanVal=1, verbose=False)
                        #if dataIndx>batch_size*2:
                        #    exit(0)

                        if runMode=='sample':
                            curModel.sample(sess, in_batch, out_batch, imgName=[os.path.join(sampleDir, imgName[0])])
                            if NUM_SAMPLES==count:
                                exit(0)
                        else:
                            summary_loss, loss = curModel.run(sess, in_batch, out_batch, is_training, imgName=os.path.join(sampleDir, imgName[0]))

                            file_writer.add_summary(summary_loss, step)
                            batch_loss.append(loss)
                            mini_loss.append(loss)

                        # Processed another batch
                        step += 1
                        count += 1
                        bar.update(count)
                    bar.finish()
                    
                    input_batches = None
                    output_batches = None
                    
                logToFile(logName, "Epoch %d Dataset #%d loss: %f" %(epochCounter, j, np.mean(mini_loss)))

                # run the sample images through the net to record the results to the Tensorflow (also locally stored)
                img_summary = curModel.sample(sess, out2board=True, imgName=logDir+'/imgs')
                file_writer.add_summary(img_summary, counter)
                
                counter += 1
                if is_training and (counter%SAVE_CKPT_COUNTER==0):
                    save_checkpoint(ckptDir, sess, saver, i_stopped+int(counter/SAVE_CKPT_COUNTER))

            test_loss = np.mean(batch_loss)
            logToFile(logName, "Epoch %d loss: %f" %(epochCounter, test_loss))

            if is_training:
                # Checkpoint model - every epoch
                #save_checkpoint(ckptDir, sess, saver, epochCounter)
                pass
            elif runMode!='sample':
                if runMode == 'val':
                    # Update the file for choosing best hyperparameters
                    curFile = open(curModel.config.val_filename, 'a')
                    curFile.write("Validation set loss: {0}".format(test_loss))
                    curFile.write('\n')
                    curFile.close()


def run_one_epoch(sess, model, dataset_filenames, modelStr, is_training):
    batch_loss = []

    for j, data_file in enumerate(dataset_filenames):
        # Get data
        input_batches,output_batches,imgNames = h52numpy(data_file, batch_sz=BATCH_SIZE)

        for in_batch,out_batch,imgName in zip(input_batches, output_batches, imgNames):
            if modelStr=='trans':
                out_batch = map_output(out_batch)

            _, loss = model.run(sess, in_batch, out_batch, is_training)

            batch_loss.append(loss)

    return np.mean(batch_loss)


def run_hyperparam(modelStr, numEpochs):
    # Hyperparameter search
    import itertools

    #---------------------------------------------
    param_key = ['lr']
    lr = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    #---------------------------------------------

    #----set variables to emulate run_model()-----
    runMode = 'train'
    ckptDir = 'hyperparam_tmpckpt'
    if numEpochs==NUM_EPOCHS:
        numEpochs = NUM_EPOCHS_HYPERPARAM
    #---------------------------------------------

    ps = [lr]
    params = list(itertools.product(*ps))

    logDir,logName = createLog('_hyperparam')
    
    count = 0
    best_val = 0
    best_param = None
    for param in params:
        # build a new computational graph
        tf.reset_default_graph()

        input_sz = [IMG_DIM, IMG_DIM, 1]
        output_sz = [IMG_DIM, IMG_DIM, 3]

        curModel = Unet(input_sz, output_sz, verbose=False)
        curModel.create_model()
        curModel.metrics()

        count += 1
        printSeparator('Running #%d of %d runs...' %(count, len(params)))
        print(getParamStr(param_key, param))
        
        with tf.Session(config=GPU_CONFIG) as sess:
            # train the network
            dataset_filenames = getDataFileNames(TRAIN_DATA, excludeFnames=['.filepart', 'test'])
            for i in range(numEpochs):
                random.shuffle(dataset_filenames)
                train_loss = run_one_epoch(sess, curModel, dataset_filenames, modelStr, is_training=True)
                print('#%d training loss: %f' % (i, train_loss))

            # run the trained network on the validation set
            dataset_filenames = getDataFileNames(VALIDATION_DATA)
            val_loss = run_one_epoch(sess, curModel, dataset_filenames, modelStr, is_training=False)
            
            logToFile(logName, 'train loss: %f, val loss: %f\n' %(train_loss, val_loss))
            if best_val<val_loss:
                best_val = val_loss
                best_param = param

    logToFile(logName, 'Best validation accuracy: %f' % best_val)
    logToFile(logName, getParamStr(param_key, best_param))


if __name__ == "__main__":
    # sample usage: python run.py -m normal -r train -e 1 -data small_dataset/tmpdata/line -ckpt small_dataset/tmpckpt

    #-------------------parse arg---------------------
    desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)
    parser = ArgumentParser(description=desc)

    print("Parsing Command Line Arguments...")
    requiredModel = parser.add_argument_group('Required Model arguments')
    requiredModel.add_argument('-m', choices = ["zhangnet", "unet"], type = str,
                        dest = 'model', required = True, help = 'Type of model to run')
    requiredTrain = parser.add_argument_group('Required Train/Test arguments')
    requiredTrain.add_argument('-r', choices = ["train", "test", "sample", "val", "hyper"], type = str,
                        dest = 'run_mode', required = True, help = 'Run mode (ie. test, train, sample)')

    parser.add_argument('-o', dest='override', action="store_true", help='Override the checkpoints')
    parser.add_argument('-data', dest='data_dir', default='', type=str, help='Set the data directory')
    parser.add_argument('-e', dest='num_epochs', default=NUM_EPOCHS, type=int, 
                        help='Set the number of Epochs')
    parser.add_argument('-ckpt', dest='ckpt_dir', default=CKPT_DIR_DEFAULT, 
                        type=str, help='Set the checkpoint directory')
    parser.add_argument('-sample', dest='sample_dir', default=SAMPLE_OUT_DIR, 
                        type=str, help='Set the sample output directory')

    args = parser.parse_args()
    #-------------------end parse arg---------------------

    makeDir(args.sample_dir)
    if args.run_mode=='hyper':
        run_hyperparam(args.model, args.num_epochs)
    else:
        run_model(args.model, args.run_mode, args.ckpt_dir, args.data_dir, args.sample_dir, args.override, args.num_epochs)

    # if args.train != "sample":
    #   if tf.gfile.Exists(SUMMARY_DIR):
    #       tf.gfile.DeleteRecursively(SUMMARY_DIR)
    #   tf.gfile.MakeDirs(SUMMARY_DIR)