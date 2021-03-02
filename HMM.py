import numpy as np
from scipy.stats import norm


def build_pdf(mean, std):
    def norm_pdf(x):
        return norm.pdf(x, mean, std)
    return norm_pdf


def build_rvs(mean, std):
    def norm_rvs():
        return norm.rvs(mean, std)
    return norm_rvs


class HMM:
    def __init__(self, n_states,
                 transition_matrix: np.ndarray = [],
                 pdf_mean_and_stds=[]):  # list of tuples
        self.N = n_states

        if transition_matrix.any():
            self.N = len(transition_matrix[0])
            self.A = transition_matrix
        else:
            self.A = np.diag(np.repeat(1., n_states))

        self.pdfs = []
        self.rvss = []
        if pdf_mean_and_stds and len(pdf_mean_and_stds) == n_states:
            for mean, std in pdf_mean_and_stds:
                self.pdfs.append(build_pdf(mean, std))
                self.rvss.append(build_rvs(mean, std))
        else:
            means = np.linspace(-1, 1, num=n_states)
            stds = [0.1 for i in range(n_states)]
            for mean, std in zip(means, stds):
                self.pdfs.append(build_pdf(mean, std))
                self.rvss.append(build_rvs(mean, std))

        self.pi = np.repeat(1., self.N) / self.N

    from forward import forward, backward

    def simulate_state_seq(self, T):
        current_state = np.random.choice(range(self.N), 1, p=self.pi)[0]
        states = [current_state]
        for i in range(T-1):
            current_state = np.random.choice(range(self.N), 1,
                                             p=self.A[current_state])[0]
            states.append(current_state)
        return states

    def simulate_observation_seq(self, T):
        states = self.simulate_state_seq(T)
        observations = [self.rvss[state]() for state in states]
        return observations
