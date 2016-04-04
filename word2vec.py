from __future__ import division
import tensorflow as tf
from tensorflow.models.embedding import gen_word2vec as word2vec
import word_config
import os
import threading
import sys
import numpy as np
import time


class Word2Vec(object):
  def __init__(self):
    self._word2id = {}
    self._id2word = []
    self._sess = tf.Session()

    self.build_graph()
    self.save_vocab()
    self.build_eval_graph()
    self.load_analogies()

    self.init_or_resume()

  def init_or_resume(self):
    self._saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(word_config.checkpoint_path)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
      print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      self._saver.restore(self._sess, ckpt.model_checkpoint_path)
      self._global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print("Created model with fresh parameters.")
      self._sess.run(tf.initialize_all_variables())


  def save(self):
    checkpoint_path = os.path.join(word_config.checkpoint_path, 'model.ckpt')
    self._saver.save(self._sess, checkpoint_path, global_step=self._global_step)


  def build_graph(self):
    #get the training data
    (words, counts, words_per_epoch, current_epoch, total_words_processed,
     examples, labels) = word2vec.skipgram(filename=word_config.train_data_path,
                                           batch_size=word_config.batch_size,
                                           window_size=word_config.window_size,
                                           min_count=word_config.min_count,
                                           subsample=word_config.subsample)
    # vocab_words, vocab_counts, words_per_epoch = self._sess.run([words, counts, words_per_epoch])
    if tf.gfile.Exists(os.path.join(word_config.output_dir, 'vocab.txt')):
      vocab_words, vocab_counts = self.load_vocab()
    else:
      vocab_words, vocab_counts = self._sess.run([words, counts])

    vocab_size = len(vocab_words)
    print("Data file: ", word_config.train_data_path)
    print("Vocab size: ", vocab_size - 1, " + UNK")
    # print("Words per epoch: ", words_per_epoch)

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


  def build_eval_graph(self):
    analogy_a = tf.placeholder(dtype=tf.int32)
    analogy_b = tf.placeholder(dtype=tf.int32)
    analogy_c = tf.placeholder(dtype=tf.int32)

    norm_w_embed = tf.nn.l2_normalize(self._w_embed_in, 1)

    a_embed = tf.gather(norm_w_embed, analogy_a)
    b_embed = tf.gather(norm_w_embed, analogy_b)
    c_embed = tf.gather(norm_w_embed, analogy_c)

    target = c_embed + (b_embed - a_embed)

    cosine_analogy_dist = tf.matmul(target, norm_w_embed, transpose_b=True)

    _, analogy_indices = tf.nn.top_k(cosine_analogy_dist, word_config.top_k_analogy)

    near_word = tf.placeholder(dtype=tf.int32)
    near_embed = tf.gather(norm_w_embed, near_word)
    cosine_near_dist = tf.matmul(near_embed, norm_w_embed, transpose_b=True)
    near_val, near_ind = tf.nn.top_k(cosine_near_dist, min(1000, self._vocab_size))

    self._analogy_a = analogy_a
    self._analogy_b = analogy_b
    self._analogy_c = analogy_c
    self._analogy_indices = analogy_indices
    self._near_word = near_word
    self._near_val = near_val
    self._near_ind = near_ind


  def param_summary(self, x):
    tensor_name = x.op.name
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


  def save_vocab(self):
    with open(os.path.join(word_config.output_dir, "vocab.txt"), "w") as f:
      for i in xrange(self._vocab_size):
        f.write("%s %d\n" % (self._vocab_words[i], self._vocab_counts[i]))

  def load_vocab(self):
    self._vocab_words = []
    self._vocab_counts = []
    with open(os.path.join(word_config.output_dir, "vocab.txt"), "r") as f:
      for line in f:
        word, count = line.split(" ")
        self._vocab_words.append(word)
        self._vocab_counts.append(int(count))


  def train_thread(self):
    initial_epoch = self._sess.run(self._current_epoch)
    while True:
      _, epoch = self._sess.run([self._train, self._current_epoch])
      if epoch != initial_epoch:
        break


  def train(self):
    initial_epoch, words_processed = self._sess.run([self._current_epoch, self._total_words_processed])

    threads = []
    for i in xrange(word_config.nr_threades):
      thread = threading.Thread(target=self.train_thread)
      thread.start()
      threads.append(thread)
    # self.train_thread()

    while True:
      time.sleep(100)
      epoch, step, learning_rate = self._sess.run([self._current_epoch, self._global_step, self._learning_rate])
      print("Epoch %4d Step %8d: lr = %5.3f \n" % (epoch, step, learning_rate))
      sys.stdout.flush()
      if epoch != initial_epoch:
        break

    for t in threads:
      t.join()


  def load_analogies(self):
    analogies = []
    skiped_analogies = 0
    with open(word_config.eval_data_path, 'rw') as f:
      for line in f:
        if line.startswith(b":"):
          continue
        analogy_words = line.strip().lower().split(b" ")
        analogy_ids = [self._word2id.get(w) for w in analogy_words]

        if None in analogy_ids or len(analogy_ids) != 4:
          skiped_analogies += 1
        else:
          analogies.append(np.array(analogy_ids))

    print ("Analogies file path %s" % word_config.eval_data_path)
    print ("Skiped analogies with errors: %d" % skiped_analogies)
    print ("Loaded nr of analogies %d" % len(analogies))

    self._analogies = np.array(analogies, dtype=np.int32)


  def predict(self, analogies):
    dict = {
      self._analogy_a: analogies[:, 0],
      self._analogy_b: analogies[:, 1],
      self._analogy_c: analogies[:, 2]
    }
    indices = self._sess.run(self._analogy_indices, feed_dict=dict)

    return indices


  def eval(self):
    correct = 0

    total = self._analogies.shape[0]
    i = 0
    while i < total:
      j = i + 2500
      if j > total:
        j = total

      analogies = self._analogies[i:j, :]
      indices = self.predict(analogies)
      i = j

      for index, analogy in enumerate(analogies):
        for k in xrange(word_config.top_k_analogy):
          if indices[index, k] == analogy[3]:
            correct += 1
            break
          elif (indices[index, k] == analogy[:3]).any():
            continue
          else:
            break

      print("Eval accuracy at precision@1 - correct / total = %4d/%d => %4.1f%% " % (correct, total, correct * 100 / total) )


def main(_):
  model = Word2Vec()

  for step in xrange(word_config.max_steps):
    model.train()
    model.save()
    model.eval()
    print("Step %4d \n" % step)

    # Perform a final save.
    # model.save()
  print ( "sess closed ")
if __name__ == "__main__":
  tf.app.run()