import numpy as np
import matplotlib.pyplot as plt

vocab_path = "./output/vocab.txt"
tsne_path = "./output/tsne.txt"

with open(tsne_path, "r") as f:
  data_list = []
  for line in f:
    x, y = line.split("\t")
    data_list.append([x,y])
  data = np.asarray(data_list)


vocab_words = []
vocab_counts = []
with open(vocab_path, "r") as f:
  for line in f:
    word, count = line.split(" ")
    vocab_words.append(word)
    vocab_counts.append(int(count))

N = data.shape[0]
labels = ['%s' % vocab_words[i] for i in range(N)]

plt.subplots_adjust(bottom = 0.1)

plt.scatter(data[:, 0], data[:, 1], marker = 'o', color='b', alpha=.4, cmap = plt.get_cmap('Spectral'))

for label, x, y in zip(labels, data[:, 0], data[:, 1]):
  plt.annotate(
      label,
      xy = (x, y), xytext = (-20, 20),
      textcoords = 'offset points', ha = 'right', va = 'bottom',
      bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
      arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plt.show()