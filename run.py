import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import os
import sys
import pickle
import random
import re
import copy

import utils_misc

from models import Unet
from argparse import ArgumentParser

DIR_MODIFIER = '/data'

TRAIN_DATA = DIR_MODIFIER + ''
TEST_DATA = DIR_MODIFIER + ''
DEVELOPMENT_DATA = DIR_MODIFIER + ''

SUMMARY_DIR = DIR_MODIFIER + '/dev_summary2'

BATCH_SIZE = 32 # should be dynamically passed into Config
DEFAULT_EPOCHS = 10
GPU_CONFIG = tf.ConfigProto()
GPU_CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.3


def plot_confusion(confusion_matrix, vocabulary, epoch, characters_remove=[], annotate=False):
	# Get vocabulary components
	vocabulary_keys = music_map.keys()
	vocabulary_values = music_map.values()
	# print vocabulary_keys
	vocabulary_values, vocabulary_keys =  tuple([list(tup) for tup in zip(*sorted(zip(vocabulary_values, vocabulary_keys)))])
	# print vocabulary_keys

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
		for x in xrange(vocabulary_size):
			for y in xrange(vocabulary_size):
				ax.annotate(str(new_confusion[x, y]), xy=(y, x),
							horizontalalignment='center',
							verticalalignment='center',
							fontsize=4)

	plt.xticks(vocabulary_values, vocabulary_keys, fontsize=6)
	plt.yticks(vocabulary_values, vocabulary_keys, fontsize=6)
	fig.savefig('confusion_matrix_epoch{0}.png'.format(epoch))


def run_model(modelStr, runMode, ckptDir, dataDir, overrideCkpt, numEpochs):
	print 'Running model...'
	print 'Using dataset %s' %dataDir
	print "Using checkpoint directory: {0}".format(ckptDir)

	batch_size = 1 if testMode=='sample' else BATCH_SIZE
	numEpochs = numEpochs if testMode=='train' else 1

	curModel = Unet(input_size, label_size, batch_size)
	curModel.create_model(is_train = (testMode=='train'))
	curModel.metrics()

	print "Running {0} model for {1} epochs.".format(modelStr, numEpochs)

	print "Reading in {0}-set filenames.".format(testMode)

	global_step = tf.Variable(0, trainable=False, name='global_step') #tf.contrib.framework.get_or_create_global_step()
	saver = tf.train.Saver(max_to_keep=numEpochs)
	step = 0

	with tf.Session(config=GPU_CONFIG) as session:
		print "Inititialized TF Session!"

		file_writer = tf.summary.FileWriter(ckptDir, graph=session.graph, max_queue=10, flush_secs=30)
		batch_accuracies = []

		if runMode == "train":
			init_op = tf.global_variables_initializer() # tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
			init_op.run()
		else:
			# Exit if no checkpoint to test
			if not found_ckpt:
				return
			numEpochs = i_stopped + 1

		# load checkpoint if necessary
		i_stopped, found_ckpt = utils_runtime.get_checkpoint(args, session, saver)

		dateset_filenames = os.listdir(datasetDir)
		# Train, dev, test model
		for i in xrange(i_stopped, numEpochs):
			print "Running epoch ({0})...".format(i)
			random.shuffle(dateset_filenames)
			for j, data_file in enumerate(dateset_filenames):
				# Get train data
				input_batches,output_batches = utils_misc.h52numpy(data_file, batch_sz=batch_size)

				for in_batch,out_batch in zip(input_batches, output_batches):
					if modelStr=='trans':
						out_batch = utils_runtime.map_output(out_batch)

					summary, conf, accuracy = curModel.run(args, session, in_batch, out_batch)

					file_writer.add_summary(summary, step)

					# Record batch accuracies for test code
					if runMode == "test" or runMode == 'dev':
						batch_accuracies.append(accuracy)

					# Processed another batch
					step += 1

			if runMode == "train":
				# Checkpoint model - every epoch
				utils_runtime.save_checkpoint(args, session, saver, i)
			else: # dev or test (NOT sample)
				test_accuracy = np.mean(batch_accuracies)
				print "Model {0} accuracy: {1}".format(runMode, test_accuracy)

				if runMode == 'dev':
					# Update the file for choosing best hyperparameters
					curFile = open(curModel.config.dev_filename, 'a')
					curFile.write("Dev set accuracy: {0}".format(test_accuracy))
					curFile.write('\n')
					curFile.close()


def hyperparamTuning_NN():
	# Hyperparameter search

	import itertools
	param_key = ['param1', 'param2', 'param3']

	param1 = []
	param2 = []
	param3 = []

	ps = [param1, param2, param3]
	params = list(itertools.product(*ps))

	count = 0
	best_val = 0
	best_param = None
	for param in params:
		# build a new computational graph
		tf.reset_default_graph()
		X = tf.placeholder(tf.float32, [None, 32, 32, 3])
		y = tf.placeholder(tf.int64, [None])
		is_training = tf.placeholder(tf.bool)
		
		y_out = my_model(X, y, is_training, param)

		# define our loss
		with tf.variable_scope('vars') as scope:
			scope.reuse_variables()
			regularizer = tf.nn.l2_loss(tf.get_variable('Wconv1')) + tf.nn.l2_loss(tf.get_variable('Wconv2')) \
							 + tf.nn.l2_loss(tf.get_variable('W1')) + tf.nn.l2_loss(tf.get_variable('W2'))

		total_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,10), logits=y_out)
		mean_loss = tf.reduce_mean(total_loss + param[-1]*regularizer)

		# define our optimizer
		optimizer = tf.train.RMSPropOptimizer(param[-2])
		train_step = optimizer.minimize(mean_loss)

		count += 1
		print('Running #%d of %d runs...' %(count, len(params)))
		param_str = ''
		for i in range(len(param)):
			param_str += param_key[i] + ':' + str(param[i]) + ', '
		print(param_str[:-2])
		
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		_,train_acc = run_model(sess,y_out,mean_loss,X_train,y_train,1,64,1000,train_step,plot_losses=False,quiet=True)
		_,val_acc = run_model(sess,y_out,mean_loss,X_val,y_val,1,64,quiet=True)
		
		print('train acc: %f, val acc: %f' %(train_acc, val_acc))
		if best_val<val_acc:
			best_val = val_acc
			best_param = param
			
		print('')

	print('Best validation accuracy: %f' % best_val)
	print(best_param)


if __name__ == "__main__":
	#-------------------parse arg---------------------
	desc = u'{0} [Args] [Options]\nDetailed options -h or --help'.format(__file__)
	parser = ArgumentParser(description=desc)

	print("Parsing Command Line Arguments...")
	requiredModel = parser.add_argument_group('Required Model arguments')
	requiredModel.add_argument('-m', choices = ["normal", "mod"], type = str,
						dest = 'model', required = True, help = 'Type of model to run')
	requiredTrain = parser.add_argument_group('Required Train/Test arguments')
	requiredTrain.add_argument('-r', choices = ["train", "test", "sample", "dev"], type = str,
						dest = 'run_mode', required = True, help = 'Run mode (ie. test, train, sample)')

	requiredTrain.add_argument('-c', type = str, dest = 'set_config',
							   help = 'Set hyperparameters', default='')

	parser.add_argument('-o', dest='override', action="store_true", help='Override the checkpoints')
	parser.add_argument('-data', dest='data_dir', default='', type=str, help='Set the data directory')
	parser.add_argument('-e', dest='num_epochs', default=DEFAULT_EPOCHS, type=int, 
						help='Set the number of Epochs')
	parser.add_argument('-ckpt', dest='ckpt_dir', default=DIR_MODIFIER + '/temp_ckpt/', 
						type=str, help='Set the checkpoint directory')

	args = parser.parse_args()
	#-------------------end parse arg---------------------

	run_model(args.model, args.run_mode, args.ckpt_dir, args.data_dir, args.override, args.num_epochs)

	if args.train != "sample":
		if tf.gfile.Exists(SUMMARY_DIR):
			tf.gfile.DeleteRecursively(SUMMARY_DIR)
		tf.gfile.MakeDirs(SUMMARY_DIR)