import tensorflow as tf
from tensorflow.models.embedding import gen_word2vec as word2vec
import word_config

class Word2Vec(object):
  def __init__(self):
    self._word2id = {}
    self._id2word = {}


  def build_graph(self):
    #get the training data
    (words, counts, words_per_epoch, current_epoch, total_words_processed,
     examples, labels) = word2vec.skipgram(filename=word_config.train_data_path,
                                           batch_size=word_config.batch_size,
                                           window_size=word_config.window_size,
                                           min_count=word_config.min_count,
                                           subsample=word_config.subsample)
    vocab_words, vocab_counts, words_per_epoch = self._session.run([words, counts, words_per_epoch])
    vocab_size = len(vocab_words)
    print("Data file: ", word_config.train_data_path)
    print("Vocab size: ", vocab_size - 1, " + UNK")
    print("Words per epoch: ", words_per_epoch)


def main():
  model = Word2Vec()

if __name__ == "__main__":
  tf.app.run()