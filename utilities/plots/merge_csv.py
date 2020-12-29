import pandas as pd

f_maml = "../../result/final/final_metatest_ours/minigolf/Dec18_03-21-11/sawtooth_rewards.csv"
f_ours = "../../result/final/maml/final_golf/sequence1/sawtooth_rewards_step_5.csv"
f_varibad = "../../result/final/varibad/golf/varibad_seq1.csv"

output_folder = "../../result/final/grouped/golf/sequence1/"

df1 = pd.read_csv(f_maml)
df2 = pd.read_csv(f_ours)
df3 = pd.read_csv(f_varibad)

tot_df = df2.merge(df1, left_on="task", right_on="task")
tot_df = tot_df.merge(df3, left_on="task", right_on="task")

tot_df.to_csv("{}rewards_golf_seq1.csv".format(output_folder), index=False)

