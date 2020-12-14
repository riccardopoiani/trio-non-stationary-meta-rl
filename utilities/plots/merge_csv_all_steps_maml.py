import pandas as pd

f_list = ["result/golfmamltest/sequence0/sin_rewards_step_0.csv",
          "result/golfmamltest/sequence0/sin_rewards_step_1.csv",
          "result/golfmamltest/sequence0/sin_rewards_step_2.csv",
          "result/golfmamltest/sequence0/sin_rewards_step_3.csv",
          "result/golfmamltest/sequence0/sin_rewards_step_4.csv",
          "result/golfmamltest/sequence0/sin_rewards_step_5.csv"
          ]

output_folder = "result/golfmamltest/sequence0/"

df_list = []
for f in f_list:
    df_list.append(pd.read_csv(f))

tot_df = df_list[0]
for i in range(1, len(df_list)):
    curr_df = df_list[i]
    tot_df = tot_df.merge(curr_df, left_on="task", right_on="task")

tot_df.to_csv("{}rewards_all_steps.csv".format(output_folder), index=False)
