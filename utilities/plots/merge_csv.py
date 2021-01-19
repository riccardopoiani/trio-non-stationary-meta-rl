import pandas as pd

f_ours = "../../result/50run/ours/newchetah/quad150_cumrewards.csv"
f_maml = "../../result/50run/maml/cheetah/sequence3/quad150_cumrewards_step_5.csv"
f_varibad = "../../result/50run/varibad/cheetah/varibad_cumsum_seq3.csv"

output_folder = "../../result/50run/grouped/cheetah/sequence2/"

df1 = pd.read_csv(f_maml)
df2 = pd.read_csv(f_ours)
df3 = pd.read_csv(f_varibad)

tot_df = df2.merge(df1, left_on="task", right_on="task")
tot_df = tot_df.merge(df3, left_on="task", right_on="task")

tot_df.to_csv("{}cumrewards_cheetah_seq3.csv".format(output_folder), index=False)

