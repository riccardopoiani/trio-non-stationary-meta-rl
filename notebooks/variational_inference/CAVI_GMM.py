import numpy as np


class Cavi_GMM(object):
    """
    Resources:
    - https://arxiv.org/pdf/1601.00670.pdf
    - https://zhiyzuo.github.io/VI/
    """

    def __init__(self, x, n_mixtures, sigma=1):
        """
        Init the variational inference CAVI procedure.

        The variational parameters in this case are two:
        - mean and standard deviation for gaussian mixtures (m_k, s_k^2)
        - assignment probabilities: for each point a distribution over the mixture assignment

        Assumptions:
        - standard deviation of the data is is one (i.e. data comes from gaussian mean with mu_k and variance 1)

        :param x: data
        :param n_mixtures: number of gaussian mixtures the data come from
        :param sigma: hyper-parameter representing the variance of the normal prior over the mixture
        components. Indeed, we have mu_k that is a Gaussian with a certain mean and a certain standard deviation.
        The mean vector is sampled from a gaussian with 0 mean and standard deviation sigma, that is
        this parameter. We are assuming this common prior as a gaussian with mean 0 and standard deviation sigma.
        """
        super(Cavi_GMM).__init__()
        self.x = x
        self.n_mixtures = n_mixtures
        self.sigma2 = sigma ** 2
        self.n_data = x.shape[0]

        # Initialize data structures
        self.m = np.random.randint(int(self.x.min()), high=int(self.x.max()), size=self.n_mixtures).astype(float)
        self.m += self.x.max() * np.random.random(self.n_mixtures)
        self.s2 = np.ones(self.n_mixtures) * np.random.random(self.n_mixtures)
        self.psi = np.random.dirichlet([np.random.random() * np.random.randint(1, 10)] * self.n_mixtures, self.n_data)

    def fit(self, max_iter=100, tol=1e-10):
        elbo_values = [self.get_elbo()]

        for iter in range(max_iter):
            self._cavi()
            elbo_values.append(self.get_elbo())

            if np.abs(elbo_values[-2] - elbo_values[-1]) <= tol:
                print("Elbo converged at iteration {} with value of {}".format(iter, elbo_values[-1]))
                break

        return elbo_values

    def get_elbo(self):
        t1 = np.log(self.s2) - self.m / self.sigma2
        t1 = t1.sum()
        t2 = -0.5 * np.add.outer(self.x ** 2, self.s2 + self.m ** 2)
        t2 += np.outer(self.x, self.m)
        t2 -= np.log(self.psi)
        t2 *= self.psi
        t2 = t2.sum()
        return t1 + t2

    def _cavi(self):
        self._update_assignment()
        self._update_mixtures()

    def _update_assignment(self):
        t1 = np.outer(self.x, self.m)
        t2 = -(0.5 * self.m ** 2 + 0.5 * self.s2)
        exponent = t1 + t2[np.newaxis, :]
        self.psi = np.exp(exponent)
        self.psi = self.psi / self.psi.sum(1)[:, np.newaxis]

    def _update_mixtures(self):
        """
        Update the parameters of the mixture models.
        This can be derived taking the derivative of the ELBO formulation w.r.t. them
        """
        self.m = np.dot(self.x, self.psi) / ((1 / self.sigma2) + np.sum(self.psi, axis=0))
        self.s2 = 1 / ((1 / self.sigma2) + np.sum(self.psi, axis=0))
