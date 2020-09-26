import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


def view_results(r_list, label_list, has_track_list, num_seq, prior_seqs, init_priors,
                 save_fig, folder, dump_data, rescale_latent=None):
    if dump_data:
        with open("{}results.pkl".format(folder), "wb") as output:
            pickle.dump(r_list, output)

    for seq in range(num_seq):
        # View rewards
        for r, label, has_track in zip(r_list, label_list, has_track_list):
            if not has_track:
                plt.plot(np.mean(r[:, seq, :], 0), label=label)
                print("Mean {} seq {}".format(label, np.mean(r[:, seq, :])))
            else:
                t = np.array([r[p, 0, seq] for p in range(r.shape[0])])
                plt.plot(np.mean(t, 0), label=label + " True sigma")
                print("Mean {} seq {}".format(label, np.mean(t)))

                t = np.array([r[p, 1, seq] for p in range(r.shape[0])])
                plt.plot(np.mean(t, 0), label=label + " False sigma")
                print("Mean {} seq {}".format(label, np.mean(t)))

                t = np.array([r[p, 2, seq] for p in range(r.shape[0])])
                plt.plot(np.mean(t, 0), label=label + " True prior")
                print("Mean {} seq {}".format(label, np.mean(t)))

        plt.title("Seq idx {}".format(seq))
        plt.legend()
        if save_fig:
            plt.savefig("{}seq_{}_reward".format(folder, seq))
        plt.show()

        # View tracking
        for r, label, has_track in zip(r_list, label_list, has_track_list):
            if has_track:
                seq_len = len(prior_seqs[seq])
                x = np.arange(seq_len)

                t = np.array([r[p, 3, seq][:, 0, 0].tolist() for p in range(r.shape[0])])
                if rescale_latent is not None:
                    t = ((rescale_latent[1] - rescale_latent[0]) / (1 - (-1))) * (t - 1) + rescale_latent[1]
                plt.plot(x, np.mean(t, 0), label=label + " Posterior true")

                t = np.array([r[p, 5, seq][:, 0, 0].tolist() for p in range(r.shape[0])])
                if rescale_latent is not None:
                    t = ((rescale_latent[1] - rescale_latent[0]) / (1 - (-1))) * (t - 1) + rescale_latent[1]
                plt.plot(x, np.mean(t, 0), label=label + " Posterior false")

                t = np.array([r[p, 4, seq] for p in range(r.shape[0])])
                t = np.mean(t, 0)
                t2 = np.zeros(t.shape[0])
                t2[1:] = t[:-1]
                t2[0] = init_priors[seq][0][0].item()
                if rescale_latent is not None:
                    t2 = ((rescale_latent[1] - rescale_latent[0]) / (1 - (-1))) * (t2 - 1) + rescale_latent[1]
                plt.plot(x, t2, label=label + " Prediction true")

                t = np.array([r[p, 6, seq] for p in range(r.shape[0])])
                t = np.mean(t, 0)
                t2 = np.zeros(t.shape[0])
                t2[1:] = t[:-1]
                t2[0] = init_priors[seq][0][0].item()
                if rescale_latent is not None:
                    t2 = ((rescale_latent[1] - rescale_latent[0]) / (1 - (-1))) * (t2 - 1) + rescale_latent[1]
                plt.plot(x, t2, label=label + " Prediction false")

                num_t = len(prior_seqs[seq])
                true_task = np.array([prior_seqs[seq][i][0].item() for i in range(num_t)])
                if rescale_latent is not None:
                    true_task = ((rescale_latent[1] - rescale_latent[0]) / (1 - (-1))) * (true_task - 1) + \
                                rescale_latent[1]
                plt.plot(true_task, label="True task")

        plt.title("Seq idx {}".format(seq))
        plt.legend()
        if save_fig:
            plt.savefig("{}seq_{}_tracking".format(folder, seq))
        plt.show()


def create_csv(r_list, label_list, has_track_list, num_seq, prior_seqs, seq_len_list, sequence_name_list,
               folder_path_with_date):
    tot_evaluation = 0
    for track in has_track_list:
        tot_evaluation += 1
        if track:
            tot_evaluation += 2

    for seq in range(num_seq):
        seq_data_mean = np.zeros(tot_evaluation + 1, seq_len_list[seq])
        seq_data_std = np.zeros(tot_evaluation + 1, seq_len_list[seq])
        seq_data_mean[-1] = np.arange(seq_len_list[seq])
        seq_data_std[-1] = np.arange(seq_len_list[seq])

        # View rewards
        algo_idx = 0
        for r, label, has_track in zip(r_list, label_list, has_track_list):
            if not has_track:
                seq_data_mean[algo_idx] = np.mean(r[:, seq, :], 0)
                seq_data_std[algo_idx] = np.std(r[:, seq, :], 0)
                algo_idx += 1
            else:
                # True Sigma
                t = np.array([r[p, 0, seq] for p in range(r.shape[0])])
                seq_data_mean[algo_idx] = np.mean(t, 0)
                seq_data_std[algo_idx] = np.std(t, 0)
                algo_idx += 1

                # False sigma
                t = np.array([r[p, 1, seq] for p in range(r.shape[0])])
                seq_data_mean[algo_idx] = np.mean(t, 0)
                seq_data_std[algo_idx] = np.std(t, 0)
                algo_idx += 1

                # True prior
                t = np.array([r[p, 2, seq] for p in range(r.shape[0])])
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

                mean_df.rename(columns={algo_idx: "mean_reward_false_sigma_{}".format(label)}, inplace=True)
                std_df.rename(columns={algo_idx: "std_reward_false_sigma_{}".format(label)}, inplace=True)
                algo_idx += 1

                mean_df.rename(columns={algo_idx: "mean_reward_true_prior_{}".format(label)}, inplace=True)
                std_df.rename(columns={algo_idx: "std_reward_true_prior_{}".format(label)}, inplace=True)
                algo_idx += 1

        total_df = mean_df.merge(std_df, left_on="task", right_on="task")
        total_df.to_csv("{}_{}.csv".format(folder_path_with_date, sequence_name_list[seq]), index=False)

        # View tracking
        tot_evaluation = 0
        for track in has_track_list:
            if track:
                tot_evaluation += 4

        mean_data = np.zeros(tot_evaluation + 1 + 1, seq_len_list[seq])
        mean_data[-1] = np.arange(seq_len_list[seq])

        std_data = np.zeros(tot_evaluation + 1 + 1, seq_len_list[seq])
        std_data[-1] = np.arange(seq_len_list[seq])

        idx = 0
        for r, label, has_track in zip(r_list, label_list, has_track_list):
            if has_track:
                # Posterior True
                t = np.array([r[p, 3, seq][:, 0, 0].tolist() for p in range(r.shape[0])])
                mean_data[idx] = np.mean(t, 0)
                std_data[idx] = np.std(t, 0)
                idx += 1

                # Posterior False
                t = np.array([r[p, 5, seq][:, 0, 0].tolist() for p in range(r.shape[0])])
                mean_data[idx] = np.mean(t, 0)
                std_data[idx] = np.std(t, 0)
                idx += 1

                # Prediction True
                t = np.array([r[p, 4, seq] for p in range(r.shape[0])])
                mean_data[idx] = np.mean(t, 0)
                std_data[idx] = np.std(t, 0)
                idx += 1

                # Prediction False
                t = np.array([r[p, 6, seq] for p in range(r.shape[0])])
                mean_data[idx] = np.mean(t, 0)
                std_data[idx] = np.std(t, 0)
                idx += 1

        num_t = len(prior_seqs[seq])
        true_task_mean = [prior_seqs[seq][i][0].item() for i in range(num_t)]
        true_task_std = [prior_seqs[seq][i][1].item() ** (1 / 2) for i in range(num_t)]
        mean_data[idx] = true_task_mean
        std_data[idx] = true_task_std

        mean_df = pd.DataFrame(seq_data_mean.transpose())
        std_df = pd.DataFrame(seq_data_std.transpose())

        mean_df.rename(columns={tot_evaluation: "task"}, inplace=True)
        std_df.rename(columns={tot_evaluation: "task"}, inplace=True)

        algo_idx = 0
        for has_track, label in zip(has_track_list, label_list):
            if has_track:
                mean_df.rename(columns={algo_idx: "posterior_true_mean_{}".format(label)}, inplace=True)
                std_df.rename(columns={algo_idx: "posterior_true_std_{}".format(label)}, inplace=True)
                algo_idx += 1

                mean_df.rename(columns={algo_idx: "posterior_false_mean_{}".format(label)}, inplace=True)
                std_df.rename(columns={algo_idx: "posterior_false_std_{}".format(label)}, inplace=True)
                algo_idx += 1

                mean_df.rename(columns={algo_idx: "prediction_true_mean_{}".format(label)}, inplace=True)
                std_df.rename(columns={algo_idx: "prediction_true_std_{}".format(label)}, inplace=True)
                algo_idx += 1

                mean_df.rename(columns={algo_idx: "prediction_false_mean_{}".format(label)}, inplace=True)
                std_df.rename(columns={algo_idx: "prediction_false_std_{}".format(label)}, inplace=True)
                algo_idx += 1

        mean_df.rename(columns={algo_idx: "true_task_mean"}, inplace=True)
        std_df.rename(columns={algo_idx: "true_task_std"}, inplace=True)

        total_df = mean_df.merge(std_df, left_on="task", right_on="task")
        total_df.to_csv("{}_{}.csv".format(folder_path_with_date, sequence_name_list[seq]), index=False)
