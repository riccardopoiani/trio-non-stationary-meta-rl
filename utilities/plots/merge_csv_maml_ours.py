import pandas as pd

f_maml = "../../result/maml_test/golfmamltest/sequence1/sawtooth_rewards_step_2.csv"
f_ours = "../../result/metatest/minigolf/Nov20_21-08-57/sawtooth_rewards.csv"

output_folder = "../../result/maml_test/golfmamltest/sequence1/"

df1 = pd.read_csv(f_maml)
df2 = pd.read_csv(f_ours)

tot_df = df2.merge(df1, left_on="task", right_on="task")

tot_df.to_csv("{}rewards_golf.csv".format(output_folder), index=False)

