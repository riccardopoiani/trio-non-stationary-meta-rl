import numpy as np
import gym

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Product, ConstantKernel as C


class GaussianProcessThompsonSampling:

    def __init__(self, arms, alpha, n_restart_opt, init_std_dev, normalized=True):
        self.x_space = arms.copy()
        if normalized:
            arms = (arms - np.min(arms)) / (np.max(arms) - np.min(arms))

        self.arms = arms
        self.init_std_dev = init_std_dev
        self.sigmas = np.ones(len(self.arms)) * self.init_std_dev
        self.means = np.zeros(len(self.arms))

        self.kernel = C(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-5, 1e5))
        self.alpha = alpha
        self.n_restarts_optimizer = n_restart_opt

        self.gp: GaussianProcessRegressor = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha ** 2,
                                                                     normalize_y=True,
                                                                     n_restarts_optimizer=self.n_restarts_optimizer)

    def fit_model(self, collected_rewards, pulled_arm_history):
        if len(collected_rewards) == 0 or len(pulled_arm_history) == 0:
            self.reset_parameters()
        else:
            x = np.atleast_2d(np.array(self.arms)[pulled_arm_history]).T

            self.gp.fit(x, collected_rewards)
            self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
            self.sigmas = np.maximum(self.sigmas, 1e-2)  # avoid negative numbers

    def reset_parameters(self):
        self.sigmas = np.ones(len(self.arms)) * self.init_std_dev
        self.means = np.zeros(len(self.arms))

    def sample_distribution(self):
        return np.random.normal(self.means, self.sigmas)

    def pull_arm(self):
        return np.argmax(self.sample_distribution())

    def solve_task(self, env):
        env.reset()
        done = False

        collected_rewards = []
        arm_history = []

        while not done:
            arm_idx = self.pull_arm()
            _, reward, done, infos = env.step(self.x_space[arm_idx])

            collected_rewards.append(reward)
            arm_history.append(arm_idx)

            self.fit_model(collected_rewards=np.array(collected_rewards), pulled_arm_history=np.array(arm_history))

        return np.sum(collected_rewards)

    def test_sequence(self, sequence, verbose=True):
        r = []

        for i, task in enumerate(sequence):
            if verbose and i % 10 == 0:
                print("Task {} in {}".format(i, len(sequence)))

            self.reset_parameters()
            r.append(self.solve_task(task))

        return r

    def meta_test(self, task_generator, prior_sequences, env_name, verbose=True):
        log = []

        for i, sequence in enumerate(prior_sequences):
            if verbose:
                print("Sequence {} in {}")

            self.reset_parameters()

            r_sequence = []

            for p in sequence:
                kwargs = task_generator.sample_task_from_prior(p)
                env = gym.make(env_name, **kwargs)
                r_sequence.append(self.solve_task(env))

            log.append(r_sequence)

        return log






