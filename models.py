


class Config(object):
	def __init__(self, hyperparam_path):
		pass


class Unet(object):

	def __init__(self, input_size, batch_size, vocab_size, hyperparam_path):
		self.config = Config(hyperparam_path)
		self.input_size = input_size
		self.config.batch_size = batch_size
		self.config.vocab_size = vocab_size
		self.input_placeholder = tf.placeholder(tf.int32, shape=[None, self.input_size], name="Inputs")
		self.label_placeholder = tf.placeholder(tf.int32, shape=[None], name="Labels")
		self.embeddings = tf.Variable(tf.random_uniform([self.config.vocab_size,
								self.config.embed_size], -1.0, 1.0))

		print("Completed Initializing the CBOW Model.....")

	def create_model(self):
		weight = tf.get_variable("Wout", shape=[self.config.embed_size, self.config.vocab_size],
					initializer=tf.contrib.layers.xavier_initializer())
		bias = tf.Variable(tf.zeros([self.config.vocab_size]))

		word_vec =  tf.nn.embedding_lookup(self.embeddings, self.input_placeholder)
		average_embedding = tf.reduce_sum(word_vec, reduction_indices=1)

		self.logits_op = tf.add(tf.matmul(average_embedding, weight), bias)
		self.probabilities_op = tf.nn.softmax(self.logits_op)

		self.add_loss_op()
		print("Built the Unet Model.....")

	def add_loss_op(self):
		self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_op, labels=self.label_placeholder))
		tf.summary.scalar('Loss', self.loss_op)
		self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss_op)

	def metrics(self):
		last_axis = len(self.probabilities_op.get_shape().as_list())
		self.prediction_op = tf.to_int32(tf.argmax(self.probabilities_op, axis=last_axis-1))
		difference = self.label_placeholder - self.prediction_op
		zero = tf.constant(0, dtype=tf.int32)
		boolean_difference = tf.cast(tf.equal(difference, zero), tf.float64)
		self.accuracy_op = tf.reduce_mean(boolean_difference)
		tf.summary.scalar('Accuracy', self.accuracy_op)

		self.summary_op = tf.summary.merge_all()

		self.confusion_matrix = tf.confusion_matrix(tf.reshape(self.label_placeholder, [-1]), tf.reshape(self.prediction_op, [-1]), num_classes=self.config.vocab_size, dtype=tf.int32)

	def run(self, args, session, in_batch, out_batch):
		feed_dict = {
			self.input_placeholder: input_batch,
			self.label_placeholder: label_batch
		}

		if args.train == "train":
			_, summary, loss, probabilities, prediction, accuracy, confusion_matrix = session.run([self.train_op, self.summary_op, self.loss_op, self.probabilities_op, self.prediction_op, self.accuracy_op, self.confusion_matrix], feed_dict=feed_dict)
		else: # Sample case not necessary b/c function will only be called during normal runs
			summary, loss, probabilities, prediction, accuracy, confusion_matrix = session.run([self.summary_op, self.loss_op, self.probabilities_op, self.prediction_op, self.accuracy_op, self.confusion_matrix], feed_dict=feed_dict)

		print "Average accuracy per batch {0}".format(accuracy)
		print "Batch Loss: {0}".format(loss)

		return summary, confusion_matrix, accuracy


	def sample(self, session, feed_values):
		feed_dict = self._feed_dict(feed_values)

		logits = session.run([self.logits_op], feed_dict=feed_dict)[0]
		return logits, np.zeros((1, 1)) # dummy value