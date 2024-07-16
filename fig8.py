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
                      'A5':       {'12':0.7, '15':0.3, '23':0.5, '24':0.5, '25':0.5, '34':0.5, '35':0.5},
                      'K:-p':     {'12':0.0, '15':0.0, '23':0.0, '24':0.0, '25':0.0, '34':0.0, '35':0.0}}, 
    }


a = ['23', '34', '12', '15', '24']
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

# def U_KB(name='Bob', rule='Original'):
#   p[name]['K:-p']['12'] = 1 - (1 - p[name][rule]['12'])
#   p[name]['K:-p']['15'] = 1 - (1 - p[name][rule]['15'])
#   p[name]['K:-p']['23'] = 1 - (1 - p[name][rule]['23'] * p[name]['K:-p']['12'])
#   p[name]['K:-p']['24'] = 1 - (1 - p[name][rule]['24'] * p[name]['K:-p']['12'])
#   p[name]['K:-p']['25'] = 1 - (1 - p[name][rule]['25'] * p[name]['K:-p']['12'])
#   p[name]['K:-p']['34'] = 1 - (1 - p[name][rule]['34'] * p[name]['K:-p']['23'])
#   p[name]['K:-p']['35'] = 1 - (1 - p[name][rule]['35'] * p[name]['K:-p']['23'])
#   p_q = np.array(list(p[name]['K:-p'].values()))
#   H_f = -(p_q*np.log2(p_q)+(1-p_q)*np.log2(1-p_q))
#   return np.mean(H_f)

# def U_KB(name='Bob', rule='Original'):
#   H = 0
#   X = Pr(name, rule)
#   p[name]['K:-p']['12'] = X[1]
#   p[name]['K:-p']['15'] = X[1]
#   p[name]['K:-p']['23'] = X[2]
#   p[name]['K:-p']['24'] = X[2]
#   p[name]['K:-p']['25'] = X[2]
#   p[name]['K:-p']['34'] = X[3]
#   p[name]['K:-p']['35'] = X[3]
#   p_q_list = list(p[name]['K:-p'].values())
#   for p_q in p_q_list:
#     H += H_f(p_q)
#   H = H/7
#   return H


X_gt = Pr('Ground Truth')

def AverageError(X, X_gt=X_gt):
  err_abs = abs(X[1:]-X_gt[1:])
  err_avg = np.mean(err_abs)
  return err_avg

def Replacement():
  keys = list(p['Bob']['Original'].keys())
  X = Pr('Bob', 'A1')
  r = [AverageError(X)]
  kbe = [U_KB(X)]
  for key in keys:
    if random.randint(0, 1):
      p['Bob']['A1'][key] = p['Alice']['Original'][key]
    X = Pr('Bob', 'A1')
    r.append(AverageError(X))
    kbe.append(U_KB(X))
  return r, kbe

def MaximumEdgeProbability():
  ji_list = ['23', '34', '12', '15', '24']
  # ji_list = ['12', '15', '23', '24', '25', '34', '35']
  X = Pr('Bob', 'A2')
  mep = [AverageError(X)]
  kbe = [U_KB(X)]
  for ji in ji_list:
    if p['Bob']['A2'][ji] < p['Alice']['Original'][ji]:
      p['Bob']['A2'][ji] = p['Alice']['Original'][ji]
    X = Pr('Bob', 'A2')
    mep.append(AverageError(X))
    kbe.append(U_KB(X))
  return mep, kbe

def MinimumEdgeEntropy():
  ji_list = ['25', '23', '24', '34', '35']
  X = Pr('Bob', 'A3')
  mee = [AverageError(X)]
  kbe = [U_KB(X)]
  for ji in ji_list:
    p['Bob']['A3'][ji] = p['Alice']['Original'][ji]
    X = Pr('Bob', 'A3')
    mee.append(AverageError(X))
    kbe.append(U_KB(X))
  return mee, kbe

def MinimumKnowledgeBaseEntropy():
  ji_list = ['12', '15', '23', '24', '25', '34', '35']
  X = Pr('Bob', 'A4')
  mnbe = [AverageError(X)]
  kbe = [U_KB(X)]
  for i in range(1):
    k = 1000
    for ji in ji_list:
      p_temp = p['Bob']['A4']
      p['Bob']['A4'][ji] = p['Alice']['Original'][ji]
      k_temp = U_KB(Pr('Bob', 'A4'))
      if k_temp < k:
        k = k_temp
        idx = ji
      p['Bob']['A4'] = p_temp
    p['Bob']['A4'][idx] = p['Alice']['Original'][idx]
    X = Pr('Bob', 'A4')
    mnbe.append(AverageError(X))
    kbe.append(U_KB(X))
  return mnbe, kbe

def MaximumAverageAnswerProbability():
  ji_list = ['12', '15', '23', '24', '25', '34', '35']
  X = Pr('Bob', 'A5')
  ae = [AverageError(X)]
  kbe = [U_KB(X)]
  for i in range(3):
    aep = 1000
    for ji in ji_list:
      p_temp = p['Bob']['A5']
      p['Bob']['A5'][ji] = p['Alice']['Original'][ji]
      aep_temp = AverageError(Pr('Bob', 'A5'))
      if aep_temp < aep:
        aep = aep_temp
        idx = ji
      p['Bob']['A5'] = p_temp
    p['Bob']['A4'][idx] = p['Alice']['Original'][idx]
    X = Pr('Bob', 'A4')
    ae.append(AverageError(X))
    kbe.append(U_KB(X))
  return ae, kbe

ae_kbe = []

ae_kbe.append(Replacement())
ae_kbe.append(MaximumEdgeProbability())
ae_kbe.append(MinimumEdgeEntropy())
ae_kbe.append(MinimumKnowledgeBaseEntropy())
ae_kbe.append(MaximumAverageAnswerProbability())

for i in range(5):
  plt.plot(ae_kbe[i][0])

plt.xlim(0, 7)
# plt.ylim(0.70, 0.78)

plt.show()

for i in range(5):
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



