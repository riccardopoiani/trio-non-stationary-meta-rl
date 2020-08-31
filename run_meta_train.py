import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import Parallel, delayed

from learner.BasicLearner import BasicLearner
from learner.LearnPerturbationMAML import PerturbatedMAML, LearnPerturbationMAML
from inference.meta_network import MetaNetworkWithPertubation, NoisyMetaNetwork
from task.TaskGenerator import SinTaskGenerator
from utilities.folder_management import handle_folder_creation

BASIC_OUTPUT_FOLDER = "report/sin_regressor/"

# Network parameters
INPUT_SIZE = 1
OUTPUT_SIZE = 1
HIDDEN_SIZE = (10,)

TRAINING_EPOCHS = 10
TEST_PERCENTAGE = 0.2

NET_LR = 0.0001

# Meta parameters
FIRST_ORDER = False
RES_STOCHASTIC_ITERATION = 30
RES_LR = 0.001

# Dataset parameters
MIN_X = -5
MAX_X = 5

BATCH_SIZE = 32
LATER_BATCH_SIZE = 10

NUM_BATCH_FIRST_TASK = 1000
NUM_BATCH_LATER_TASK = 5

MAX_BATCHES_FIRST_TASK = 1000
MAX_BATCHES_LATER_TASK = 30

SIMULATE_MAX_BATCHES = 1000
SIMULATE_BATCH_SIZE = 32
SIMULATE_N_BATCHES = 250
SIMULATE_TEST_SPLIT = 0.2

# Task parameters
AMPLITUDE_LIST = [1, 1.4, 1.8, 2.2, 2.8, 3.2, 3.6, 4.0, 4.4, 4.0, 3.6, 3.2, 3.6, 3.2, 3.2, 2.8]
PHASE_LIST = [0, 0.3, 0.6, 0.8, 1.0, 1.3, 1.6, 1.3, 1.6, 1.3, 1.0, 1.5, 1.8, 1.4, 1.5, 1.3]
n_tasks = len(AMPLITUDE_LIST)


def get_arguments():
    """
    Defining the arguments available for the script
    :return: argument parser
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-l", "--learner_name", default="StandardRegressor", type=str)

    # Experiment run details
    parser.add_argument("-n_runs", "--n_runs", default=1, help="Number of runs of the experiments", type=int)
    parser.add_argument("-n_jobs", "--n_jobs", default=1, help="Number of jobs to be run", type=int)
    parser.add_argument("-device", "--device", default="cpu", type=str)

    # Store results
    parser.add_argument("-s", "--save_result", help="Whether to store results or not", type=lambda x: int(x) != 0,
                        default=0)
    parser.add_argument("-o", "--output_folder", default=BASIC_OUTPUT_FOLDER, help="Basic folder where"
                                                                                   "to store the output",
                        type=str)
    parser.add_argument("-v", "--verbose", type=lambda x: int(x) != 0, default=1)

    return parser.parse_args()


def get_learner(args, task_generator: SinTaskGenerator):
    name = args.learner_name
    if name == "meta":
        print("Instantating meta learner...")
        init_mean = torch.zeros(HIDDEN_SIZE[-1], device=args.device)
        init_std = torch.ones(HIDDEN_SIZE[-1], device=args.device)
        net: torch.nn.Module = MetaNetworkWithPertubation(n_in=INPUT_SIZE, n_out=OUTPUT_SIZE,
                                                          hidden_sizes=HIDDEN_SIZE,
                                                          init_mean=init_mean,
                                                          init_std=init_std)
        optimizer = torch.optim.Adam(net.parameters(), lr=NET_LR)
        loss = torch.nn.MSELoss()
        learner: PerturbatedMAML = PerturbatedMAML(network=net,
                                                   n_in=INPUT_SIZE,
                                                   n_out=OUTPUT_SIZE,
                                                   optimizer=optimizer,
                                                   loss_function=loss,
                                                   x_space=task_generator.x_space,
                                                   device=args.device,
                                                   first_order=FIRST_ORDER)
        simulating = True
        residual = False
    elif name == "basic":
        print("Instantiating basic learner...")
        init_mean = torch.zeros(HIDDEN_SIZE[-1], device=args.device)
        init_std = torch.zeros(HIDDEN_SIZE[-1], device=args.device)
        net: torch.nn.Module = MetaNetworkWithPertubation(n_in=INPUT_SIZE, n_out=OUTPUT_SIZE,
                                                          hidden_sizes=HIDDEN_SIZE,
                                                          init_mean=init_mean,
                                                          init_std=init_std)
        optimizer = torch.optim.Adam(net.parameters(), lr=NET_LR)
        loss = torch.nn.MSELoss()
        learner: BasicLearner = BasicLearner(network=net, optimizer=optimizer, loss_function=loss,
                                             device=args.device,)
        simulating = False
        residual = False
    elif name == "meta_res":
        print("Meta learning with residual learning...")
        init_mean_w = torch.zeros(HIDDEN_SIZE[-1], device=args.device)
        init_std_w = torch.ones(HIDDEN_SIZE[-1], device=args.device)
        init_mean_b = torch.zeros(OUTPUT_SIZE, device=args.device)
        init_std_b = torch.ones(OUTPUT_SIZE, device=args.device)
        net: torch.nn.Module = NoisyMetaNetwork(n_in=INPUT_SIZE, n_out=OUTPUT_SIZE, hidden_sizes=HIDDEN_SIZE,
                                                init_mean_w=init_mean_w, init_std_w=init_std_w,
                                                init_mean_b=init_mean_b, init_std_b=init_std_b)
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=NET_LR)
        learner: LearnPerturbationMAML = LearnPerturbationMAML(network=net,
                                                               n_in=INPUT_SIZE,
                                                               n_out=OUTPUT_SIZE,
                                                               optimizer=optimizer,
                                                               loss_function=loss,
                                                               x_space=task_generator.x_space,
                                                               device=args.device,
                                                               first_order=FIRST_ORDER)
        simulating = True
        residual = True
    else:
        raise NotImplemented()

    return learner, simulating, residual


def main(args, id):
    assert n_tasks == len(AMPLITUDE_LIST), "Len does not match: {} {}".format(n_tasks, len(AMPLITUDE_LIST))
    assert n_tasks == len(PHASE_LIST), "Len does not match: {} {}".format(n_tasks, len(PHASE_LIST))

    task_generator: SinTaskGenerator = SinTaskGenerator(x_min=MIN_X, x_max=MAX_X)
    learner, simulating, residual = get_learner(args=args, task_generator=task_generator)
    error_list = []

    for task in range(n_tasks):
        print("Process {} task {}".format(id, task))
        num_batches = NUM_BATCH_FIRST_TASK if task == 0 else NUM_BATCH_LATER_TASK
        bs = BATCH_SIZE if task == 0 else LATER_BATCH_SIZE
        max_batches = MAX_BATCHES_FIRST_TASK if task == 0 else MAX_BATCHES_LATER_TASK

        data_loader = task_generator.get_data_loader(amplitude=AMPLITUDE_LIST[task],
                                                     phase=PHASE_LIST[task],
                                                     num_batches=num_batches,
                                                     batch_size=bs,
                                                     test_perc=TEST_PERCENTAGE)
        data_loader_test = task_generator.get_data_loader_eval(amplitude=AMPLITUDE_LIST[task],
                                                               phase=PHASE_LIST[task])
        if task != 0 and simulating:
            # Prepare meta-training
            print("Process {} Task simulation".format(id))
            learner.simulate_meta_training(num_batches=SIMULATE_N_BATCHES,
                                           batch_size=SIMULATE_BATCH_SIZE,
                                           test_split=SIMULATE_TEST_SPLIT,
                                           max_batches=SIMULATE_MAX_BATCHES,
                                           verbose=args.verbose)

        # Adaptation: with all the training dataset
        print("Process {} task adapation".format(id))
        learner.train(data_loader=data_loader, max_batches=max_batches, verbose=args.verbose)

        if residual:
            print("Process {} residual learning".format(id))
            learner.learn_perturbation_model(curr_data_loader=data_loader,
                                             stochastic_iteration=RES_STOCHASTIC_ITERATION,
                                             lr=RES_LR,
                                             verbose=args.verbose)

        # Evaluation of the predictions: with all the test set
        print("Process {} task evaluation".format(id))
        eval = learner.evaluate(data_loader=data_loader_test, verbose=args.verbose)
        print("EVAL {}".format(eval))
        error_list.append(eval)

    return error_list


def run(id, seed, args):
    """
    Run a task to carry out the experiment
    :param id: id of the task
    :param seed: random seed that is used in this execution
    :param args: arguments given to the experiment
    :return: collected rewards
    """
    np.random.seed(seed)
    torch.manual_seed(seed=seed)
    print("Starting run {}".format(id))
    error = main(args=args, id=id)
    print("Done run {}".format(id))
    return error


args = get_arguments()
seeds = [np.random.randint(1000000) for _ in range(args.n_runs)]
if args.n_jobs == 1:
    results = [run(id=id, seed=seed, args=args) for id, seed in zip(range(args.n_runs), seeds)]
else:
    results = Parallel(n_jobs=args.n_jobs, backend='loky')(
        delayed(run)(id=id, seed=seed, args=args) for id, seed in zip(range(args.n_runs), seeds))

errors = [results[0] for res in results]

x_points = np.arange(n_tasks)
mean_error = np.zeros(shape=n_tasks)
for task in range(n_tasks):
    for exp in range(args.n_runs):
        mean_error[task] += errors[exp][task]
    mean_error[task] /= args.n_runs
plt.plot(x_points, mean_error)
plt.show()

print(mean_error)

if args.save_result:
    # Set up writing folder and file
    fd, folder_path_with_date = handle_folder_creation(result_path=args.output_folder)

    # Writing results and experiment details
    print("Storing results on file...", end="")
    with open("{}errors_{}.pkl".format(folder_path_with_date, args.learner_name), "wb") as output:
        pickle.dump(errors, output)

