import tensorflow as tf
from bhtsne.bhtsne import bh_tsne
import word_config
from word2vec import Word2Vec
import os

DEFAULT_THETA = 0.5
EMPTY_SEED = -1
VERBOSE = True
INITIAL_DIMENSIONS = 50

def main(_):
  model = Word2Vec()
  norm_w_embed = tf.nn.l2_normalize(model._w_embed_in, 1) # [vocab_size, embed_size]
  embedings = model._sess.run(norm_w_embed)
  results = bh_tsne(embedings, no_dims=2, perplexity=50, theta=DEFAULT_THETA, randseed=EMPTY_SEED,
            verbose=VERBOSE)

  with open(os.path.join(word_config.output_dir, "tsne.txt"), "w") as f:
    for result in results:
      fmt = ''
      for i in range(1, len(result)):
        fmt = fmt + '{}\t'
      fmt = fmt + '{}\n'

      f.write(fmt.format(*result))


if __name__ == "__main__":
  tf.app.run()