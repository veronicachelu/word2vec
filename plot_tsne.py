import numpy as np
import matplotlib.pyplot as plt

vocab_path = "./output/vocab.txt"
tsne_path = "./output/tsne.txt"
scale_factor = 100000

with open(tsne_path, "r") as f:
  data_list = []
  for line in f:
    x, y = line.split("\t")
    data_list.append([float(x) * scale_factor, float(y) * scale_factor])
  data = np.asarray(data_list, dtype=np.float32)


vocab_words = []
vocab_counts = []
with open(vocab_path, "r") as f:
  for line in f:
    word, count = line.split(" ")
    vocab_words.append(word)
    vocab_counts.append(int(count))

N = data.shape[0]
labels = ['%s' % vocab_words[i] for i in range(N)]

# plt.subplots_adjust(bottom = 0.1)

plt.scatter(data[:, 0], data[:, 1], color='b', alpha=0, s=1)

for label, x, y in zip(labels, data[:, 0], data[:, 1]):
  plt.annotate(
      label,
      xy = (x, y), fontsize=1)

# plt.show()
plt.savefig('test.pdf', dpi=1200, format='pdf')