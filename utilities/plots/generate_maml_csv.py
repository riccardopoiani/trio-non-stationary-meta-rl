import numpy as np
import pandas as pd
import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input-folder", type=str, required=True,
                    help="Input folder of MAML result (the folder should contain only data of the given sequence)")
parser.add_argument("--output-folder", type=str, required=True,
                    help="Where to store results in CSV format")
parser.add_argument("--seq-len", type=int, required=True,
                    help="Number of tasks in the sequence")
parser.add_argument("--num-steps", type=int, required=True,
                    help="Number of MAML adaptation steps")
parser.add_argument("--seq-name", type=str, required=True,
                    help="Name of the sequence")

args, rest_args = parser.parse_known_args()

seq_len = args.seq_len
num_steps = args.num_steps
seq_name = args.seq_name
input_folder = args.input_folder
output_folder = args.output_folder


# Read all the files
def read_file_list(file_list):
    o = []
    for f in file_list:
        objects = []
        with (open(f, "rb")) as openfile:
            while True:
                try:
                    objects.append(pickle.load(openfile))
                except EOFError:
                    break
        o.append(objects)
    return o


files_in_f = os.listdir(input_folder)

# Reading all the data and filling the array
all_data = np.zeros(shape=(num_steps + 1, len(files_in_f), seq_len))
for step in range(num_steps + 1):
    for policy_idx, file in enumerate(files_in_f):
        curr_res = np.load(input_folder + file)  # read results of the folder
        assert seq_len == curr_res['valid_returns'].shape[0], curr_res['valid_returns'].shape[0]

        # Select the data according to the correct adaptation step
        if step == num_steps:
            curr_res = curr_res['valid_returns']
        else:
            curr_res = curr_res['train_returns'][seq_len * step: seq_len * step + seq_len, :]

        # Now that we have the data, we should update the mean and the standard deviation of the current policy
        all_data[step, policy_idx, :] = np.mean(curr_res, 1)

# Calculate mean and standard deviation
for step in range(num_steps + 1):
    mean_data = np.zeros(shape=(2, seq_len))
    std_data = np.zeros(shape=(2, seq_len))
    mean_data[-1] = np.arange(seq_len)
    std_data[-1] = np.arange(seq_len)

    mean_data[0] = np.mean(all_data[step, :, :], 0)
    std_data[0] = np.std(all_data[step, :, :], 0)

    # Now that we have all the data for the current step, we can create the CSV and dump it on file
    mean_df = pd.DataFrame(mean_data.transpose())
    std_df = pd.DataFrame(std_data.transpose())

    # Rename columns
    mean_df.rename(columns={1: "task"}, inplace=True)
    std_df.rename(columns={1: "task"}, inplace=True)
    mean_df.rename(columns={0: "mean_reward_maml_step_{}".format(step)}, inplace=True)
    std_df.rename(columns={0: "std_reward_maml_step_{}".format(step)}, inplace=True)

    # Merge DF and dump on files
    total_df = mean_df.merge(std_df, left_on="task", right_on="task")
    total_df.to_csv("{}{}_rewards_step_{}.csv".format(output_folder, seq_name, step), index=False)

    # Print the mean of the rewards for the current step in order to select the adaptation step with the better results
    print("Step {} -> Mean = {}".format(step, np.mean(all_data[step, :, :])))

# GENERATE CUMULATIVE REWARD CSV
cum_sum = np.cumsum(all_data, 2)
for step in range(num_steps + 1):
    mean_data = np.zeros(shape=(2, seq_len))
    std_data = np.zeros(shape=(2, seq_len))
    mean_data[-1] = np.arange(seq_len)
    std_data[-1] = np.arange(seq_len)
    mean_data[0] = np.mean(cum_sum[step, :, :], 0)
    std_data[0] = np.std(cum_sum[step, :, :], 0)

    mean_df = pd.DataFrame(mean_data.transpose())
    std_df = pd.DataFrame(std_data.transpose())

    # Rename columns
    mean_df.rename(columns={1: "task"}, inplace=True)
    std_df.rename(columns={1: "task"}, inplace=True)
    mean_df.rename(columns={0: "mean_reward_maml_step_{}".format(step)}, inplace=True)
    std_df.rename(columns={0: "std_reward_maml_step_{}".format(step)}, inplace=True)

    # Merge DF and dump on files
    total_df = mean_df.merge(std_df, left_on="task", right_on="task")
    total_df.to_csv("{}{}_cumrewards_step_{}.csv".format(output_folder, seq_name, step), index=False)

    # Print the mean of the rewards for the current step in order to select the adaptation step with the better results
    print("Step {} -> CumMean = {}".format(step, np.mean(cum_sum[step, :, :])))
