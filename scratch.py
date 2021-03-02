import numpy as np
from matplotlib import pyplot as plt

from HMM import HMM, build_pdf


pdf1 = build_pdf(0, 2)
print(pdf1(10))
# plt.plot(np.linspace(-10, 10), [pdf1(x) for x in np.linspace(-10, 10)])
# plt.show()

transition = np.asarray([[0.6, 0.3, 0.1], [0.4, 0.3, 0.3], [0.1, 0.4, 0.5]])
meanstds = [(2., 0.1), (1., 0.1), (1., 0.1)]
hmm = HMM(3, transition_matrix=transition, pdf_mean_and_stds=meanstds)
print(hmm.A[0])
print(hmm.pi)
print(np.random.choice(range(hmm.N), 1, p=hmm.pi))
print(hmm.simulate_state_seq(50))
obs = hmm.simulate_observation_seq(100)
plt.plot(obs, 'ko')
plt.plot(obs, 'k--')
# plt.show()

print(hmm.forward(obs))

transition2 = np.asarray([[0.8, 0.18, 0.02], [0.2, 0.4, 0.4], [0.02, 0.4, 0.58]])
hmm2 = HMM(3, transition_matrix=transition2, pdf_mean_and_stds=meanstds)
# obs = hmm2.simulate_observation_seq(100)
print(hmm2.forward(obs))

print(hmm.backward(obs))
print(hmm2.backward(obs))

print(hmm.backward(obs)
      / hmm2.backward(obs))
print(hmm.forward(obs)
      / hmm2.forward(obs))

print(hmm.backward(obs)
      / hmm2.backward(obs)
      / (hmm.forward(obs)
      / hmm2.forward(obs)))  # always around 0.998, 1.078 or 0.947
