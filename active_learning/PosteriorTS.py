class PosteriorTSAgent:

    def __init__(self, vi, vi_optim):
        # Inference network
        self.vi = vi
        self.vi_optim = vi_optim

    def train(self, n_train_iter, eval_interval):
        test_list = []

        for i in range(n_train_iter):
            self.train_iter()

            if i % eval_interval == 0:
                print("Iteration {} / {}".format(i, n_train_iter))
                e = self.evaluate()
                test_list.append(e)

        return test_list

    def train_iter(self):
        pass

    def evaluate(self):
        return 0