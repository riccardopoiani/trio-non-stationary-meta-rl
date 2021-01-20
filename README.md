# Code for the paper "Meta-Reinforcement Learning by Tracking Task Non-Stationarity"

### Requirements
The code is developed mainly with PyTorch. <br>
The main requirements can be found in `requirements.txt`.
For the MuJoCo experiments you need to install MuJoCo.
You also need to install requirements specified in [ikostrikov/pytorch-a2c-ppo-acktr-gail/](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/).

### Overview
The main training loops for the methods are found in `learner`.  <br>
Folder `ppo` contains implementation of PPO and policy models 
and it is mainly based on the work presented by [ikostrikov/pytorch-a2c-ppo-acktr-gail/](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/). 
Check their repository and follow their installation process. <br>
Folder `inference` contains inference networks models and utils.  <br>
Folder `task` contains task generators, you may want to look examples
here in the case in which you need to run the code on other environments. <br> 
Folder `configs` contains details on all the hyper parameters used 
to train each of our proposed algorithms. Each training script will be 
executed under these configurations. It is possible to modify parameters 
by changing the flags. <br>

### Running experiments
In order to train Thomson sampling, RL2, and Bayes optimal policy you can type: <br>
`python train.py --env-type ant_goal --algo rl2` <br>
This will train RL2 on the ant environment using the hyper-parameters specified in the corresponding `config` file. <br>
`--algo` may be `rl2`, `ts`, `bayes`. <br>
`--env-type` may be one of the following options: `cheetah_vel`, `ant_goal`, `golf`, `golf_signals`. In the case
of `golf_signals`, you can specify the number of additional latent variables as `--golf-num-signals`. <br> <br>

Policy and inference networks will be automatically stored in `result/env_name/algo_name/current_timestamp/`.<br> <br>

Once you have trained a policy for a given environment, you can launch the meta-testing script on its meta-test
sequences. For instance, you can type `python ant_goal_meta_test.py --task-len 1 --num-test-processes 50 --n-jobs 1`. 
`--task-len 1` should be set to `1` for MuJoCo experiments, and `4` for MiniGolf. `--bayes-folder`, `--ts-folder` and 
`--rl2-folder` is used to specify the folders in which training policies and inference networks are stored.
`--output-folder` is used to specify the folders in which results will be stored. <br>
If you trained more policy for each environment, each policy of each algorithm will be used for the tested 
environment, results will be averaged and stored on a CSV file together with the standard deviations. Moreover,
raw data will be dumped on a 'pickle' file. <br>
Please, note that meta-test scripts assumes that you trained at least a policy for each of the 3 algorithms.
The scripts requires that the input folder for each of the algorithm (i.e. bayes, ts, and rl2) are folders
containing at least a folder obtained via the training scripts. 
Also, if you stored results in different folders, you need to specify the correct folders where to find trained
policies and inference networks. 
<br> <br>

For what concerns results MAML and VariBAD, we used the code available at
[tristandeleu/pytorch-maml-rl](https://github.com/tristandeleu/pytorch-maml-rl) and 
[lmzintgraf/varibad](https://github.com/lmzintgraf/varibad) respectively, with small modifications in order to meta-test
specific sequences of tasks. We provide modified code for both cases in two zip folder within the repository.
You need to follow their installation process to get things working. <br>
For VariBAD, once the models has been trained as in the original repository, you can launch 
`python meta_test.py` to obtain test results. 
For instance, `python meta_test.py --env-type cheetah_vel_varibad --input-main-folder cheetah_varibad/ --has-done 0 --num-test-processes 50 --task-len 1 --noise-seq-var 0.01 --output cheetah_varibad/`
launches the script to obtain HalfCheetahVel results. <br>
For MAML, once the models has been trained as in the original repository, you can launch 
`python test_sequences.py` to obtain test results.
For instance, `python test_sequence.py --num-steps 5 --num-workers 8 --output cheetah_test --input-main-folder cheetah_train/ --seq-std 0.01 --fast-batch-size 50`
launches the script for HalfCheetahVel environment. `input-main-folder` is a folder that contains all the folders 
generated by the training scripts launched for that environment.
<br> <br> 
 
All the code we provided has not been tested on GPUs. 
Random seeds has been used to train and test the algorithms. 

### Visualizing results
Once you have obtained the results for each of the method `utilities/plots/` contains utilities to generate and merge
CSV files from raw data. You can visualize CSV results with the tool you prefer. 









