import random
import numpy as np
import matplotlib.pyplot as plt

# X_id = {'X1':0, 'X2':1, 'X3':2, 'X4':3, 'X5':4}
p = {'Ground Truth': {'Original': {'12':0.7, '15':0.3, '23':0.7, '24':0.3, '25':0.0, '34':0.7, '35':0.3}},
     'Alice':        {'Original': {'12':0.5, '15':0.5, '23':0.7, '24':0.3, '25':0.0, '34':0.7, '35':0.3}},
     'Bob':          {'Original': {'12':0.7, '15':0.3, '23':0.5, '24':0.5, '25':0.5, '34':0.5, '35':0.5},
                      'A3':       {'12':0.7, '15':0.3, '23':0.5, '24':0.5, '25':0.5, '34':0.5, '35':0.5},
                      'Beyond':   {'12':0.7, '15':0.3, '23':0.5, '24':0.5, '25':0.5, '34':0.5, '35':0.5},
                      'Within':   {'12':0.7, '15':0.3, '23':0.5, '24':0.5, '25':0.5, '34':0.5, '35':0.5},}, 
    }




def Pr(name, rule='Original'):
  X = np.ones(6)
  X[2] = 1 - (1-p[name][rule]['12']*X[1])
  X[3] = 1 - (1-p[name][rule]['23']*X[2])
  X[4] = 1 - (1-p[name][rule]['24']*X[2])*(1-p[name][rule]['34']*X[3])
  X[5] = 1 - (1-p[name][rule]['15']*X[1])*(1-p[name][rule]['25']*X[2])*(1-p[name][rule]['35']*X[3])
  return X

def H_f(p):
  if p == 0 or p == 1:
    return 0
  else:
    return -(p*np.log2(p)+(1-p)*np.log2(1-p))

def U_KB(X_list):
  H = 0
  for X in X_list[1:]:
    H += H_f(X)
  H = H/len(X_list[1:])
  return H


X_gt = Pr('Ground Truth')

def AverageError(X, X_gt=X_gt):
  err_abs = abs(X[1:]-X_gt[1:])
  err_avg = np.mean(err_abs)
  return err_avg

print(AverageError(Pr('Ground Truth')))



def MinimumEdgeEntropy():
  ji_list = ['25', '23', '24', '34', '35']
  X = Pr('Bob', 'A3')
  ae = [AverageError(X)]
  kbe = [U_KB(X)]
  for ji in ji_list:
    p['Bob']['A3'][ji] = p['Alice']['Original'][ji]
    X = Pr('Bob', 'A3')
    ae.append(AverageError(X))
    kbe.append(U_KB(X))
  return ae, kbe

def BeyoundTask():
  ji_list = ['25', '23', '24', '34', '35']
  
  X = Pr('Bob', 'Beyond')
  ae = [abs(X[5]-X_gt[5])]
  kbe = [U_KB(X)]
  for ji in ji_list:
    p['Bob']['Beyond'][ji] = p['Alice']['Original'][ji]
    X = Pr('Bob', 'Beyond')
    ae.append(abs(X[5]-X_gt[5]))
    kbe.append(U_KB(X))
  return ae, kbe

def WithinTask():
  ji_list = ['25', '24', '34', '35']
  X = Pr('Bob', 'Within')
  ae = [abs(X[5]-X_gt[5])]
  kbe = [U_KB(X)]
  for ji in ji_list:
    p['Bob']['Within'][ji] = p['Alice']['Original'][ji]
    print(p['Bob']['Within'])
    X = Pr('Bob', 'Within')
    ae.append(abs(X[5]-X_gt[5]))
    kbe.append(U_KB(X))
  return ae, kbe

ae_kbe = []


ae_kbe.append(MinimumEdgeEntropy())
ae_kbe.append(BeyoundTask())
ae_kbe.append(WithinTask())

for i in range(3):
  plt.plot(ae_kbe[i][0])

plt.xlim(0, 7)
# plt.ylim(0.70, 0.78)

plt.show()

for i in range(3):
  plt.plot(ae_kbe[i][1])

plt.xlim(0, 7)
# plt.ylim(0.70, 0.78)

plt.show()

# p_gt = X_gt[1:]
# H = 0
# for pgt in p_gt:
#   if pgt:
#     H += -pgt*np.log2(pgt)

# H = -3/14*np.log2(3/14)-3/14*np.log2(3/14)-8/14*np.log2(8/14)
H = -(1/7*np.log2(1/7)+(1-1/7)*np.log2(1-1/7))-(1/7*np.log2(1/7)+(1-1/7)*np.log2(1-1/7))-(5/7*np.log2(5/7)+(1-5/7)*np.log2(1-5/7))
H = H/3
H = 0
for X in Pr('Bob', 'Original'):
  H +=H_f(X)
H = H/5


print(U_KB(Pr('Ground Truth', 'Original')))
# print(U_KB())
# print((-0.3*np.log2(0.3)-0.7*np.log2(0.7))*3)



