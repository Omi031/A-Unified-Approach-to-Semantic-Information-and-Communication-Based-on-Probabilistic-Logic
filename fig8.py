import random
import numpy as np
import matplotlib.pyplot as plt

# X_id = {'X1':0, 'X2':1, 'X3':2, 'X4':3, 'X5':4}
p = {'Ground Truth': {'Original': {'12':0.7, '15':0.3, '23':0.7, '24':0.3, '25':0.0, '34':0.7, '35':0.3}},
     'Alice':        {'Original': {'12':0.5, '15':0.5, '23':0.7, '24':0.3, '25':0.0, '34':0.7, '35':0.3}},
     'Bob':          {'Original': {'12':0.7, '15':0.3, '23':0.5, '24':0.5, '25':0.5, '34':0.5, '35':0.5},
                      'A1':       {'12':0.7, '15':0.3, '23':0.5, '24':0.5, '25':0.5, '34':0.5, '35':0.5},
                      'A2':       {'12':0.7, '15':0.3, '23':0.5, '24':0.5, '25':0.5, '34':0.5, '35':0.5},
                      'A3':       {'12':0.7, '15':0.3, '23':0.5, '24':0.5, '25':0.5, '34':0.5, '35':0.5},
                      'A4':       {'12':0.7, '15':0.3, '23':0.5, '24':0.5, '25':0.5, '34':0.5, '35':0.5},
                      'A5':       {'12':0.7, '15':0.3, '23':0.5, '24':0.5, '25':0.5, '34':0.5, '35':0.5},}, 
    }



def Pr(name, rule='Original'):
  X = np.ones(6)
  X[2] = 1 - (1-p[name][rule]['12']*X[1])
  X[3] = 1 - (1-p[name][rule]['23']*X[2])
  X[4] = 1 - (1-p[name][rule]['24']*X[2])*(1-p[name][rule]['34']*X[3])
  X[5] = 1 - (1-p[name][rule]['15']*X[1])*(1-p[name][rule]['25']*X[2])*(1-p[name][rule]['35']*X[3])
  return X

X_gt = Pr('Ground Truth')

def AverageError(X, X_gt=X_gt):
  err_abs = abs(X[1:]-X_gt[1:])
  err_avg = np.mean(err_abs)
  return err_avg

def Replacement():
  keys = list(p['Bob']['Original'].keys())
  r = [AverageError(Pr('Bob', 'A1'))]
  for key in keys:
    if random.randint(0, 1):
      p['Bob']['A1'][key] = p['Alice']['Original'][key]
    r.append(AverageError(Pr('Bob', 'A1')))
  return r

def MaximumEdgeProbability():
  mep = [AverageError(Pr('Bob', 'A2'))]
  ji_list = ['23', '34', '12', '15', '24']
  for ji in ji_list:
    if p['Bob']['A2'][ji] < p['Alice']['Original'][ji]:
      p['Bob']['A2'][ji] = p['Alice']['Original'][ji]
    mep.append(AverageError(Pr('Bob', 'A2')))
  return mep

def MinimumEdgeEntropy():
  mee = [AverageError(Pr('Bob', 'A3'))]
  p['Bob']['A3']['25'] = p['Alice']['Original']['25']
  mee.append(AverageError(Pr('Bob', 'A3')))
  p['Bob']['A3']['23'] = p['Alice']['Original']['23']
  mee.append(AverageError(Pr('Bob', 'A3')))
  p['Bob']['A3']['24'] = p['Alice']['Original']['24']
  mee.append(AverageError(Pr('Bob', 'A3')))
  p['Bob']['A3']['34'] = p['Alice']['Original']['34']
  mee.append(AverageError(Pr('Bob', 'A3')))
  p['Bob']['A3']['35'] = p['Alice']['Original']['35']
  mee.append(AverageError(Pr('Bob', 'A3')))
  return mee

def MinimumKnowledgeBaseEntropy():
  pass

plt.plot(Replacement())
plt.plot(MaximumEdgeProbability())
plt.plot(MinimumEdgeEntropy())
# plt.show()

# p_gt = X_gt[1:]
# H = 0
# for pgt in p_gt:
#   if pgt:
#     H += -pgt*np.log2(pgt)

# H = -3/14*np.log2(3/14)-3/14*np.log2(3/14)-8/14*np.log2(8/14)
H = -1/7*np.log2(1/7)-1/7*np.log2(1/7)
H = -1/8*np.log2(1/8)-1/8*np.log2(1/8)

print(H)
# print((-0.3*np.log2(0.3)-0.7*np.log2(0.7))*3)



