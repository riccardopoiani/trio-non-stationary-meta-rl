import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def view_results_multiple_dim(r_list, label_list, has_track_list, num_seq, prior_seqs, init_priors,
                              save_fig, folder, rescale_latent=None, view_tracking=False,
                              num_dim=None, show_no_tracking=False):
    for seq in range(num_seq):
        # View rewards
        for r, label, has_track in zip(r_list, label_list, has_track_list):
            r = np.array(r)
            if not has_track:
                r = np.array([r[i][seq] for i in range(r.shape[0])])
                plt.plot(np.mean(r, 0), label=label)
                print("Seq {} Algo {} = {} +- {}".format(seq, label, np.mean(r), np.std(r)))
            else:
                t = np.array([r[p, 0, seq] for p in range(r.shape[0])])
                plt.plot(np.mean(t, 0), label=label + " True sigma")
                print("Seq {} Algo {} = {} +- {}".format(seq, label, np.mean(t), np.std(t)))

                # t = np.array([r[p, 1, seq] for p in range(r.shape[0])])
                # plt.plot(np.mean(t, 0), label=label + " False sigma")

                t = np.array([r[p, 2, seq] for p in range(r.shape[0])])
                plt.plot(np.mean(t, 0), label=label + " True prior")
                print("Seq {} Algo True prior {} = {} +- {}".format(seq, label, np.mean(t), np.std(t)))

                if show_no_tracking:
                    t = np.array([r[p, 3, seq] for p in range(r.shape[0])])
                    plt.plot(np.mean(t, 0), label=label + " No tracking")
                    print("Seq {} Algo No tracking {} = {} +- {}".format(seq, label, np.mean(t), np.std(t)))

        plt.title("Seq idx {}".format(seq))
        plt.legend(title='legend', bbox_to_anchor=(1.05, 1), loc='upper left')
        if save_fig:
            plt.savefig("{}seq_{}_reward".format(folder, seq))
        plt.show()
        print("Here")
        if view_tracking:
            # View tracking
            for d in range(num_dim):
                for r, label, has_track in zip(r_list, label_list, has_track_list):
                    r = np.array(r)
                    if has_track:
                        seq_len = len(prior_seqs[seq])
                        x = np.arange(seq_len)
                        t = np.array([r[p, 4, seq][:, :, d].mean(1).tolist() for p in range(r.shape[0])])
                        if rescale_latent is not None:
                            t = ((rescale_latent[1] - rescale_latent[0]) / (1 - (-1))) * (t - 1) + rescale_latent[1]
                        plt.plot(x, np.mean(t, 0), label=label + " Posterior true")

                        # t = np.array([r[p, 6, seq][:, :, d].mean(1).tolist() for p in range(r.shape[0])])
                        # if rescale_latent is not None:
                        #    t = ((rescale_latent[1] - rescale_latent[0]) / (1 - (-1))) * (t - 1) + rescale_latent[1]
                        # plt.plot(x, np.mean(t, 0), label=label + " Posterior false")

                        t = np.array([r[p, 5, seq] for p in range(r.shape[0])])[:, :, d]
                        t = np.mean(t, 0)
                        t2 = np.zeros(t.shape[0])
                        t2[1:] = t[:-1]
                        t2[0] = init_priors[seq][0][0][d].item()
                        if rescale_latent is not None:
                            t2 = ((rescale_latent[1] - rescale_latent[0]) / (1 - (-1))) * (t2 - 1) + rescale_latent[1]
                        plt.plot(x, t2, label=label + " Prediction true")

                        # t = np.array([r[p, 7, seq] for p in range(r.shape[0])])
                        # t = np.mean(t, 0)
                        # t2 = np.zeros(t.shape[0])
                        # t2[1:] = t[:-1]
                        # t2[0] = init_priors[seq][0][0].item()
                        # if rescale_latent is not None:
                        #    t2 = ((rescale_latent[1] - rescale_latent[0]) / (1 - (-1))) * (t2 - 1) + rescale_latent[1]
                        # plt.plot(x, t2, label=label + " Posterior false")

                num_t = len(prior_seqs[seq])
                true_task = np.array([prior_seqs[seq][i][0][d].item() for i in range(num_t)])
                if rescale_latent is not None:
                    true_task = ((rescale_latent[1] - rescale_latent[0]) / (1 - (-1))) * (true_task - 1) + \
                                rescale_latent[1]
                plt.plot(true_task, label="True task")

                plt.title("Seq idx {}".format(seq))
                plt.legend(title='legend', bbox_to_anchor=(1.05, 1), loc='upper left')
                if save_fig:
                    plt.savefig("{}seq_{}_tracking".format(folder, seq))
                plt.show()


def create_csv_tracking(r_list, label_list, has_track_list, num_seq, prior_seqs, seq_len_list, sequence_name_list,
                        folder_path_with_date, init_priors, rescale_latent, num_dim):
    for seq in range(num_seq):
        # View tracking
        tot_evaluation = 0
        for track in has_track_list:
            if track:
                tot_evaluation += 3

        for d in range(num_dim):
            mean_data = np.zeros((tot_evaluation + 1 + 1, seq_len_list[seq]))
            mean_data[-1] = np.arange(seq_len_list[seq])

            std_data = np.zeros((tot_evaluation + 1 + 1, seq_len_list[seq]))
            std_data[-1] = np.arange(seq_len_list[seq])

            idx = 0
            for r, label, has_track in zip(r_list, label_list, has_track_list):
                r = np.array(r)
                if has_track:
                    # Posterior True
                    seq_len = len(prior_seqs[seq])
                    x = np.arange(seq_len)
                    t = np.array([r[p, 4, seq][:, :, d].mean(1).tolist() for p in range(r.shape[0])])
                    if rescale_latent is not None:
                        t = ((rescale_latent[1] - rescale_latent[0]) / (1 - (-1))) * (t - 1) + rescale_latent[1]
                    mean_data[idx] = np.mean(t, 0)
                    std_data[idx] = np.std(t, 0)
                    idx += 1

                    # Prediction True
                    t = np.array([r[p, 5, seq] for p in range(r.shape[0])])[:, :, d]
                    t = np.mean(t, 0)
                    t2 = np.zeros(t.shape[0])
                    t2[1:] = t[:-1]
                    t2[0] = init_priors[seq][0][0][d].item()
                    if rescale_latent is not None:
                        t2 = ((rescale_latent[1] - rescale_latent[0]) / (1 - (-1))) * (t2 - 1) + rescale_latent[1]
                    mean_data[idx] = t2

                    t = np.array([r[p, 5, seq] for p in range(r.shape[0])])[:, :, d]
                    t = np.std(t, 0)
                    t2 = np.zeros(t.shape[0])
                    t2[1:] = t[:-1]
                    t2[0] = init_priors[seq][0][1][d].item()
                    std_data[idx] = t2
                    idx += 1

            num_t = len(prior_seqs[seq])
            true_task_mean = np.array([prior_seqs[seq][i][0].item() for i in range(num_t)])
            if rescale_latent is not None:
                true_task_mean = ((rescale_latent[1] - rescale_latent[0]) / (1 - (-1))) * (true_task_mean - 1) + \
                                 rescale_latent[1]
            true_task_std = np.array([prior_seqs[seq][i][1].item() ** (1 / 2) for i in range(num_t)])
            mean_data[idx] = true_task_mean
            std_data[idx] = true_task_std

            mean_df = pd.DataFrame(mean_data.transpose())
            std_df = pd.DataFrame(std_data.transpose())
            mean_df.rename(columns={tot_evaluation + 1: "task"}, inplace=True)
            std_df.rename(columns={tot_evaluation + 1: "task"}, inplace=True)

            algo_idx = 0
            for has_track, label in zip(has_track_list, label_list):
                if has_track:
                    mean_df.rename(columns={algo_idx: "posterior_true_mean_{}".format(label)}, inplace=True)
                    std_df.rename(columns={algo_idx: "posterior_true_std_{}".format(label)}, inplace=True)
                    algo_idx += 1

                    # mean_df.rename(columns={algo_idx: "posterior_false_mean_{}".format(label)}, inplace=True)
                    # std_df.rename(columns={algo_idx: "posterior_false_std_{}".format(label)}, inplace=True)
                    # algo_idx += 1

                    mean_df.rename(columns={algo_idx: "prediction_true_mean_{}".format(label)}, inplace=True)
                    std_df.rename(columns={algo_idx: "prediction_true_std_{}".format(label)}, inplace=True)
                    algo_idx += 1

                    # mean_df.rename(columns={algo_idx: "prediction_false_mean_{}".format(label)}, inplace=True)
                    # std_df.rename(columns={algo_idx: "prediction_false_std_{}".format(label)}, inplace=True)
                    # algo_idx += 1
            mean_df.rename(columns={algo_idx: "true_task_mean"}, inplace=True)
            std_df.rename(columns={algo_idx: "true_task_std"}, inplace=True)
            total_df = mean_df.merge(std_df, left_on="task", right_on="task")
            total_df.to_csv("{}{}_tracking_dim_{}.csv".format(folder_path_with_date, sequence_name_list[seq], d),
                            index=False)

def create_csv_rewards(r_list, label_list, has_track_list, num_seq, prior_seqs, seq_len_list, sequence_name_list,
                       folder_path_with_date):
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
                seq_data_mean[algo_idx] = np.mean(t, 0)
                seq_data_std[algo_idx] = np.std(t, 0)
                algo_idx += 1
            else:
                # True Sigma
                t = np.array([r[p, 0, seq] for p in range(r.shape[0])])
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
                seq_data_mean[algo_idx] = np.mean(t, 0)
                seq_data_std[algo_idx] = np.std(t, 0)
                algo_idx += 1

                # No tracking
                t = np.array([r[p, 3, seq] for p in range(r.shape[0])])
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
        total_df.to_csv("{}{}_rewards.csv".format(folder_path_with_date, sequence_name_list[seq]), index=False)
