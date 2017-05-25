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


def plot_confusion(confusion_matrix, vocabulary, epoch, characters_remove=[], annotate=False):
    # Get vocabulary components
    vocabulary_keys = music_map.keys()
    vocabulary_values = music_map.values()
    # print(vocabulary_keys
    vocabulary_values, vocabulary_keys =  tuple([list(tup) for tup in zip(*sorted(zip(vocabulary_values, vocabulary_keys)))])
    # print(vocabulary_keys

    removed_indicies = []
    for c in characters_remove:
        i = vocabulary_keys.index(c)
        vocabulary_keys.remove(c)
        index = vocabulary_values.pop(i)
        removed_indicies.append(index)

    # Delete unnecessary rows
    conf_temp = np.delete(confusion_matrix, removed_indicies, axis=0)
    # Delete unnecessary cols
    new_confusion = np.delete(conf_temp, removed_indicies, axis=1)


    vocabulary_values = range(len(vocabulary_keys))
    vocabulary_size = len(vocabulary_keys)

    fig, ax = plt.subplots(figsize=(10, 10))
    res = ax.imshow(new_confusion.astype(int), interpolation='nearest', cmap=plt.cm.jet)
    cb = fig.colorbar(res)

    if annotate:
        for x in range(vocabulary_size):
            for y in range(vocabulary_size):
                ax.annotate(str(new_confusion[x, y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=4)

    plt.xticks(vocabulary_values, vocabulary_keys, fontsize=6)
    plt.yticks(vocabulary_values, vocabulary_keys, fontsize=6)
    fig.savefig('confusion_matrix_epoch{0}.png'.format(epoch))


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
    
    logName = createLogFile()

    printSeparator('Starting TF session')
    with tf.Session(config=GPU_CONFIG) as sess:
        print("Inititialized TF Session!")

        # load checkpoint if necessary
        i_stopped, found_ckpt = get_checkpoint(overrideCkpt, ckptDir, sess, saver)

        file_writer = tf.summary.FileWriter(ckptDir, graph=sess.graph, max_queue=10, flush_secs=30)

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
        dataset_filenames = getDataFileNames(dataDir)
        for i in range(i_stopped, numEpochs):
            batch_loss = []
            printSeparator("Running epoch %d" % i)
            random.shuffle(dataset_filenames)

            for j, data_file in enumerate(dataset_filenames):
                # Get data
                print('Reading data in %s...' % data_file)
                input_batches,output_batches,imgNames = h52numpy(data_file, batch_sz=batch_size, 
                                                                 mod_output=(modelStr=='zhangnet'))
                print('Done reading, running the network (%d of %d)' % (j, len(dataset_filenames)))

                bar = progressbar.ProgressBar(maxval=len(input_batches))
                bar.start()
                count = 0
                for in_batch,out_batch,imgName in zip(input_batches, output_batches, imgNames):
                    if runMode=='sample':
                        out_img = curModel.sample(sess, in_batch, imgName=os.path.join(sampleDir, imgName[0]))
                        exit(0)
                    else:
                        summary, loss = curModel.run(sess, in_batch, out_batch, is_training)

                        file_writer.add_summary(summary, step)
                        batch_loss.append(loss)

                    # Processed another batch
                    step += 1
                    count += 1
                    bar.update(count)
                bar.finish()

            test_loss = np.mean(batch_loss)
            logToFile(logName, "Epoch %d loss: %f" %(i, test_loss))

            if is_training:
                # Checkpoint model - every epoch
                save_checkpoint(ckptDir, sess, saver, i)
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

    logName = createLogFile()
    
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
            dataset_filenames = getDataFileNames(TRAIN_DATA)
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