train_data_path = "./train/text8"
eval_data_path = "./eval/questions-words.txt"
output_dir = "./output"
checkpoint_path = "./checkpoint"
embed_size = 200
max_steps = 100
learning_rate = 0.025
#number of negative examples
nr_neg_samples = 25
batch_size = 500
window_size = 5
# minimum number of word occurrences needed to be included in the voc
min_count = 5
#subsample threshold. Words that appear with higher frequency will be randomly downsampled
subsample = 1e-3
nr_threades = 12
top_k_analogy = 4
checkpoint_steps = 10
