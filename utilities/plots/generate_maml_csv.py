import numpy as np
import pandas as pd
import os
import pickle
import argparse

INPUT_FOLDER = "../../result/50run/maml/cheetah/sequence3/"
OUTPUT_FOLDER = "../../result/50run/maml/cheetah/sequence3/"
SEQ_NAME = "quad150"
SEQ_LEN = 150
NUM_STEPS = 5

"""
parser = argparse.ArgumentParser()
parser.add_argument("--input-folder", type=str, required=True)
parser.add_argument("--output-folder", type=str, required=True)
parser.add_argument("--seq-len", type=int, required=True)
parser.add_argument("--num-steps", type=int, required=True)
parser.add_argument("--seq-name", type=str, required=True)

args, rest_args = parser.parse_known_args()
"""


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


files_in_f = os.listdir(INPUT_FOLDER)

# Reading all the data and filling the array
all_data = np.zeros(shape=(NUM_STEPS + 1, len(files_in_f), SEQ_LEN))
for step in range(NUM_STEPS + 1):
    for policy_idx, file in enumerate(files_in_f):
        curr_res = np.load(INPUT_FOLDER + file)  # read results of the folder
        assert SEQ_LEN == curr_res['valid_returns'].shape[0], curr_res['valid_returns'].shape[0]

        # Select the data according to the correct adaptation step
        if step == NUM_STEPS:
            curr_res = curr_res['valid_returns']
        else:
            curr_res = curr_res['train_returns'][SEQ_LEN * step: SEQ_LEN * step + SEQ_LEN, :]

        # Now that we have the data, we should update the mean and the standard deviation of the current policy
        all_data[step, policy_idx, :] = np.mean(curr_res, 1)

# Calculate mean and standard deviation
for step in range(NUM_STEPS + 1):
    mean_data = np.zeros(shape=(2, SEQ_LEN))
    std_data = np.zeros(shape=(2, SEQ_LEN))
    mean_data[-1] = np.arange(SEQ_LEN)
    std_data[-1] = np.arange(SEQ_LEN)

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
    total_df.to_csv("{}{}_rewards_step_{}.csv".format(OUTPUT_FOLDER, SEQ_NAME, step), index=False)

    # Print the mean of the rewards for the current step in order to select the adaptation step with the better results
    print("Step {} -> Mean = {}".format(step, np.mean(all_data[step, :, :])))

# GENERATE CUMULATIVE REWARD CSV
cum_sum = np.cumsum(all_data, 2)
for step in range(NUM_STEPS + 1):
    mean_data = np.zeros(shape=(2, SEQ_LEN))
    std_data = np.zeros(shape=(2, SEQ_LEN))
    mean_data[-1] = np.arange(SEQ_LEN)
    std_data[-1] = np.arange(SEQ_LEN)
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
    total_df.to_csv("{}{}_cumrewards_step_{}.csv".format(OUTPUT_FOLDER, SEQ_NAME, step), index=False)

    # Print the mean of the rewards for the current step in order to select the adaptation step with the better results
    print("Step {} -> CumMean = {}".format(step, np.mean(cum_sum[step, :, :])))