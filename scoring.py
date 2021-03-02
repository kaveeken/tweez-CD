def forward(self, Oseq):
    Q = [i for i in range(self.N)]
    current_alphas = [self.pi[i] * self.pdfs[i](Oseq[0]) for i in Q]  # alpha_0
    for t in range(1, len(Oseq)):
        next_alphas = \
            [sum([alpha * self.A[i, j] for i, alpha
                  in enumerate(current_alphas)])
             * self.pdfs[j](Oseq[t]) for j in Q]
        current_alphas = next_alphas  # may be unnessecary
    return sum(current_alphas)


def backward(self, Oseq):
    Q = [i for i in range(self.N)]
    current_betas = [1. for i in Q]
    for t in reversed(range(0, len(Oseq))):
        next_betas = \
            [sum([self.A[i, j] * self.pdfs[j](Oseq[t]) * beta
                  for j, beta in enumerate(current_betas)])
             for i in Q]
        current_betas = next_betas
    return sum(current_betas)
