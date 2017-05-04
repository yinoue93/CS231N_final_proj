import pickle
import os
import random
import numpy as np
import reader

import tensorflow as tf
from utils_preprocess import *


def get_checkpoint(args, session, saver):
	# Checkpoint
	found_ckpt = False

	if args.override:
		if tf.gfile.Exists(args.ckpt_dir):
			tf.gfile.DeleteRecursively(args.ckpt_dir)
		tf.gfile.MakeDirs(args.ckpt_dir)

	# check if arags.ckpt_dir is a directory of checkpoints, or the checkpoint itself
	if len(re.findall('model.ckpt-[0-9]+', args.ckpt_dir)) == 0:
		ckpt = tf.train.get_checkpoint_state(args.ckpt_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(session, ckpt.model_checkpoint_path)
			i_stopped = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
			print "Found checkpoint for epoch ({0})".format(i_stopped)
			found_ckpt = True
		else:
			print('No checkpoint file found!')
			i_stopped = 0
	else:
		saver.restore(session, args.ckpt_dir)
		i_stopped = int(args.ckpt_dir.split('/')[-1].split('-')[-1])
		print "Found checkpoint for epoch ({0})".format(i_stopped)
		found_ckpt = True


	return i_stopped, found_ckpt


def save_checkpoint(args, session, saver, i):
	checkpoint_path = os.path.join(args.ckpt_dir, 'model.ckpt')
	saver.save(session, checkpoint_path, global_step=i)


def map_output(outData):
	pass
	

if __name__ == "__main__":
	pass