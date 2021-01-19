import pandas as pd
import numpy as np
import pickle

INPUT_FILE = "../../result/50run/ours/newchetah/data_results.pkl"
OUTPUT_FOLDER = "../../result/50run/ours/newchetah/"

label_list = ['Ours', 'TS', 'RL2']
has_track_list = [True, True, False]
num_seq = 2
seq_len_list = [2, 150]
sequence_name_list = ["sintan2", "quad150"]

objects = []
with (open(INPUT_FILE, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
objects = objects[0]
r_list = objects

tot_evaluation = 0
for track in has_track_list:
    tot_evaluation += 1  # reward algo
    if track:
        tot_evaluation += 2  # other modes

for seq in range(num_seq):
    seq_data_mean = np.zeros((tot_evaluation + 1, seq_len_list[seq]))
    seq_data_std = np.zeros((tot_evaluation + 1, seq_len_list[seq]))
    seq_data_mean[-1] = np.arange(seq_len_list[seq])
    seq_data_std[-1] = np.arange(seq_len_list[seq])

    # View rewards
    algo_idx = 0
    for r, label, has_track in zip(r_list, label_list, has_track_list):
        r = np.array(r)
        if not has_track:
            t = np.array([r[i][seq] for i in range(r.shape[0])])
            t = np.cumsum(t, 1)
            seq_data_mean[algo_idx] = np.mean(t, 0)
            seq_data_std[algo_idx] = np.std(t, 0)
            algo_idx += 1
        else:
            # True Sigma
            t = np.array([r[p, 0, seq] for p in range(r.shape[0])])
            t = np.cumsum(t, 1)
            seq_data_mean[algo_idx] = np.mean(t, 0)
            seq_data_std[algo_idx] = np.std(t, 0)
            algo_idx += 1

            # False sigma
            # t = np.array([r[p, 1, seq] for p in range(r.shape[0])])
            # seq_data_mean[algo_idx] = np.mean(t, 0)
            # seq_data_std[algo_idx] = np.std(t, 0)
            # algo_idx += 1

            # True prior
            t = np.array([r[p, 2, seq] for p in range(r.shape[0])])
            t = np.cumsum(t, 1)
            seq_data_mean[algo_idx] = np.mean(t, 0)
            seq_data_std[algo_idx] = np.std(t, 0)
            algo_idx += 1

            # No tracking
            t = np.array([r[p, 3, seq] for p in range(r.shape[0])])
            t = np.cumsum(t, 1)
            seq_data_mean[algo_idx] = np.mean(t, 0)
            seq_data_std[algo_idx] = np.std(t, 0)
            algo_idx += 1

    mean_df = pd.DataFrame(seq_data_mean.transpose())
    std_df = pd.DataFrame(seq_data_std.transpose())

    mean_df.rename(columns={tot_evaluation: "task"}, inplace=True)
    std_df.rename(columns={tot_evaluation: "task"}, inplace=True)

    algo_idx = 0
    for has_track, label in zip(has_track_list, label_list):
        if not has_track:
            mean_df.rename(columns={algo_idx: "mean_reward_{}".format(label)}, inplace=True)
            std_df.rename(columns={algo_idx: "std_reward_{}".format(label)}, inplace=True)
        else:
            mean_df.rename(columns={algo_idx: "mean_reward_true_sigma_{}".format(label)}, inplace=True)
            std_df.rename(columns={algo_idx: "std_reward_true_sigma_{}".format(label)}, inplace=True)
            algo_idx += 1

            # mean_df.rename(columns={algo_idx: "mean_reward_false_sigma_{}".format(label)}, inplace=True)
            # std_df.rename(columns={algo_idx: "std_reward_false_sigma_{}".format(label)}, inplace=True)
            # algo_idx += 1

            mean_df.rename(columns={algo_idx: "mean_reward_true_prior_{}".format(label)}, inplace=True)
            std_df.rename(columns={algo_idx: "std_reward_true_prior_{}".format(label)}, inplace=True)
            algo_idx += 1

            mean_df.rename(columns={algo_idx: "mean_reward_no_tracking_{}".format(label)}, inplace=True)
            std_df.rename(columns={algo_idx: "std_reward_no_tracking_{}".format(label)}, inplace=True)
            algo_idx += 1

    total_df = mean_df.merge(std_df, left_on="task", right_on="task")
    total_df.to_csv("{}{}_cumrewards.csv".format(OUTPUT_FOLDER, sequence_name_list[seq]), index=False)

