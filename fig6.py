import math
import numpy as np
import matplotlib.pyplot as plt

def P(n, k, e):
  p = 0.0
  for i in range(k, n+1):
    p += math.comb(n, i)*e**i*(1-e)**(n-i)
  return p

nk_list = [[5, 4],
           [4, 3],
           [3, 1]]

cp = np.arange(0, 0.61, 0.001)

dep=[]

for nk in nk_list:
  d = []
  for eps in cp:
    d.append(1/(1-P(nk[0], nk[1], eps)))
  dep.append(d)


fig, ax = plt.subplots()


plt.plot(cp, dep[0], ls='-')
plt.plot(cp, dep[1], ls='--')
plt.plot(cp, dep[2], ls='-.', c='#ecb01f')

plt.legend(['For query(1)', 'For query(2)', 'For query(3)'])
plt.xlabel('Crossover Probability $\epsilon$')
plt.xlim(0, 0.6)
# plt.ylim(10**-6, 1)
# plt.yscale('log')
plt.grid(which='major')
plt.grid(which='minor', ls=':')
plt.show()