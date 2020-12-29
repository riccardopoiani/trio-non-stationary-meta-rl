import numpy as np
import pandas as pd
import pickle

INPUT_FILE = "../../result/final/varibad/golf/results_varibad_seq2.pkl"
OUTPUT_FILE = "../../result/final/varibad/golf/varibad_seq2.csv"

objects = []
with (open(INPUT_FILE, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
objects = objects[0]


seq_len = objects[0].shape[0]
num_policies = len(objects)
res = np.zeros((num_policies, seq_len))
for i, o in enumerate(objects):
    res[i] = o.numpy()

data = np.zeros((3, seq_len))
data[0] = np.mean(res, 0)
data[1] = np.std(res, 0)
data[2] = np.arange(seq_len)

df = pd.DataFrame(data.transpose())
df.rename(columns={0: 'mean_varibad', 1: 'std_varibad', 2: 'task'}, inplace=True)
df.to_csv(OUTPUT_FILE, index=False)
