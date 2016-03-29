train_data_path = "./train"
eval_dir = ".eval"
output_dir = "./output"
embed_size = 200
max_steps = 100
learning_rate = 0.025
#number of negative examples
nr_neg = 25
batch_size = 500
window_size = 5
# minimum number of word occurrences needed to be included in the voc
min_count = 5
#subsample threshold. Words that appear with higher frequency will be randomly downsampled
subsample = 1e-3