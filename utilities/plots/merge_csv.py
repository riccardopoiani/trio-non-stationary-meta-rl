import argparse
import pandas as pd

"""
This script merges different CSV files.
It requires a csv file from our repo, one from varibad and one from MAML.
Files should be related to the same sequence/experiment. 
"""

parser = argparse.ArgumentParser()
parser.add_argument("--ours-file", type=str, required=True,
                    help="Where to store results in CSV format")
parser.add_argument("--maml-file", type=str, required=True,
                    help="Where to store results in CSV format")
parser.add_argument("--varibad-file", type=str, required=True,
                    help="Where to store results in CSV format")
parser.add_argument("--output-file", type=str, required=True,
                    help="Where to store results in CSV format")

args, rest_args = parser.parse_known_args()
f_ours = args.ours_file
f_maml = args.maml_file
f_varibad = args.varibad_file
f_output = args.output_file

df1 = pd.read_csv(f_maml)
df2 = pd.read_csv(f_ours)
df3 = pd.read_csv(f_varibad)

tot_df = df2.merge(df1, left_on="task", right_on="task")
tot_df = tot_df.merge(df3, left_on="task", right_on="task")

tot_df.to_csv("{}".format(f_output), index=False)
