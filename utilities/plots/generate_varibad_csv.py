import argparse
import numpy as np
import pandas as pd
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--input-file", type=str, required=True,
                    help="Input folder of MAML result (the folder should contain only data of the given sequence)")
parser.add_argument("--output-file", type=str, required=True,
                    help="Where to store mean rewards")
parser.add_argument("--output-file-cumulative", type=str, required=True,
                    help="Where to store cumulative rewards file")

args, rest_args = parser.parse_known_args()
input_file = args.input_file
output_file_mean = args.output_file
output_file_cumsum = args.output_file_cumulative

objects = []
with (open(input_file, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
objects = objects[0]

# Generate mean and standard deviation
seq_len = objects[0].shape[0]
num_policies = len(objects)
res = np.zeros((num_policies, seq_len))
for i, o in enumerate(objects):
    res[i] = o.numpy()

data = np.zeros((3, seq_len))
data[0] = np.mean(res, 0)
data[1] = np.std(res, 0)
data[2] = np.arange(seq_len)

df = pd.DataFrame(data.transpose())
df.rename(columns={0: 'mean_varibad', 1: 'std_varibad', 2: 'task'}, inplace=True)
df.to_csv(output_file_mean, index=False)

# Generate cumulative rewards
data = np.zeros((3, seq_len))
cum_res = np.cumsum(res, 1)
data[0] = np.mean(cum_res, 0)
data[1] = np.std(cum_res, 0)
data[2] = np.arange(seq_len)

df = pd.DataFrame(data.transpose())
df.rename(columns={0: 'mean_varibad', 1: 'std_varibad', 2: 'task'}, inplace=True)
df.to_csv(output_file_cumsum, index=False)
