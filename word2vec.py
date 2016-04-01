from __future__ import division
import tensorflow as tf
from tensorflow.models.embedding import gen_word2vec as word2vec
import word_config
import os
import threading
import sys
import numpy as np


class Word2Vec(object):
  def __init__(self, sess):
    self._word2id = {}
    self._id2word = []
    self._sess = sess

    self.build_graph()
    tf.initialize_all_variables().run()

    self._saver = tf.train.Saver()

  def build_graph(self):
    #get the training data
    (words, counts, words_per_epoch, current_epoch, total_words_processed,
     examples, labels) = word2vec.skipgram(filename=word_config.train_data_path,
                                           batch_size=word_config.batch_size,
                                           window_size=word_config.window_size,
                                           min_count=word_config.min_count,
                                           subsample=word_config.subsample)
    vocab_words, vocab_counts, words_per_epoch = self._sess.run([words, counts, words_per_epoch])
    vocab_size = len(vocab_words)
    print("Data file: ", word_config.train_data_path)
    print("Vocab size: ", vocab_size - 1, " + UNK")
    print("Words per epoch: ", words_per_epoch)

    self._id2word = vocab_words
    for id, word in enumerate(self._id2word):
      self._word2id[word] = id

    w_embed_in = tf.Variable(tf.random_uniform([vocab_size, word_config.embed_size],
                                                -0.5 / word_config.embed_size, 0.5 / word_config.embed_size),
                             name="w_embed_in")
    w_embed_out = tf.Variable(tf.zeros([vocab_size, word_config.embed_size]), name="w_embed_out")

    self.param_summary(w_embed_in)
    self.param_summary(w_embed_out)

    # learning_rate = tf.Variable(word_config.learning_rate, trainable=False, name="learning_rate")

    global_step = tf.Variable(0, trainable=False, name="global_step")

    total_words = words_per_epoch * word_config.max_steps

    learning_rate = word_config.learning_rate * tf.maximum(0.0001, tf.cast(1 - total_words_processed / total_words, tf.float32))

    inc = global_step.assign_add(1)
    with tf.control_dependencies([inc]):
      train = word2vec.neg_train(w_embed_in, w_embed_out, examples, labels, learning_rate, vocab_counts.tolist(),
                                 word_config.nr_neg_samples)

    self._vocab_words = vocab_words
    self._vocab_counts = vocab_counts
    self._vocab_size = vocab_size
    self._w_embed_in = w_embed_in
    self._w_embed_out = w_embed_out
    self._train = train
    self._examples = examples
    self._labels = labels
    self._global_step = global_step
    self._current_epoch = current_epoch
    self._total_words_processed = total_words_processed
    self._learning_rate = learning_rate
    print("end of build graph")


  def param_summary(self, x):
    tensor_name = x.op.name
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


  def save_vocab(self):
    with open(os.path.join(word_config.output_dir, "vocab.txt"), "w") as f:
      for i in xrange(self._vocab_size):
        f.write("%s %d\n" % (self._vocab_words[i], self._vocab_counts[i]))

  def train_thread(self):
    initial_epoch = self._sess.run(self._current_epoch)
    while True:
      _, epoch, step, learning_rate = self._sess.run([self._train, self._current_epoch, self._global_step, self._learning_rate])
      print("Epoch %4d Step %8d: lr = %5.3f \n" % (epoch, step, learning_rate))
      sys.stdout.flush()
      if epoch != initial_epoch:
        break

  def train(self):
    initial_epoch, words_processed = self._sess.run([self._current_epoch, self._total_words_processed])

    # threads = []
    # for i in xrange(word_config.nr_threades):
    #   thread = threading.Thread(target=self.train_thread())
    #   thread.start()
    #   threads.append(thread)
    self.train_thread()

    # epoch, step, learning_rate = self._session.run([self._current_epoch,
    #                                                                  self._global_step,
    #                                                                  self._learning_rate])
    # print("Epoch %4d Step %8d: lr = %5.3f words/sec = %8.0f\r" % (epoch, step, learning_rate))


def main(_):

  with tf.Graph().as_default(), tf.Session() as sess:
    model = Word2Vec(sess)

    for step in xrange(word_config.max_steps):
      model.train()

if __name__ == "__main__":
  tf.app.run()