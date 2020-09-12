from abc import ABC, abstractmethod


class TaskGenerator(ABC):

    def __init__(self):
        super(TaskGenerator, self).__init__()

    @abstractmethod
    def create_task_family(self, n_tasks, n_batches=1, test_perc=0, batch_size=160):
        pass

    @abstractmethod
    def sample_task_from_prior(self, prior):
        pass

    @abstractmethod
    def sample_pair_tasks(self, num_processes):
        pass



