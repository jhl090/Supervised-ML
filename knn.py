from scipy.spatial import distance
from random import *
import numpy as np
import time as t 
import operator as op

### Global structs for feature vectors and labels, mapped by index
train_vs = []
train_ls = []
testd_vs = []
testd_ls = []
valid_vs = []
valid_ls = []

### Global structs for projections
projmatix = []
projtrain = []
projtests = []
projvalid = []

### Helper function to parse data .txt files
def parseData(): 
  training_file = open("pa1train.txt", "r")
  testdata_file = open("pa1test.txt", "r")
  validata_file = open("pa1validate.txt", "r")
  projectn_file = open("projection.txt", "r")

  # Parse TRAINING data
  for line in training_file:
    str = line
    fv = []
    
    v = [int(s) for s in str.split() if s.isdigit()]
    
    for i in range(0, 784):
      fv.append(v[i])
    
    train_vs.append(fv)
    train_ls.append(v[784])

  # Finished parsing TRAINING data, now parse TESTING data
  for line in testdata_file:
    str = line
    tfv = []

    v = [int(s) for s in str.split() if s.isdigit()]
    
    for i in range(0, 784):
      tfv.append(v[i])

    testd_vs.append(tfv)
    testd_ls.append(v[784])

  # Finished parsing TESTING data, now parse VALIDATION data
  for line in validata_file:
    str = line
    vfv = []

    v = [int(s) for s in str.split() if s.isdigit()]
    
    for i in range(0, 784):
      vfv.append(v[i])

    valid_vs.append(vfv)
    valid_ls.append(v[784])

  # Finished parsing VALIDATION data, now parse PROJECTION data
  for line in projectn_file:
    str = line
    pv = []

    v = [float(s) for s in str.split()]

    for i in range(0, 20):
      pv.append(v[i])
    
    projmatix.append(pv)

  # Finished parsing PROJECTION data, clean up
  training_file.close()
  testdata_file.close()
  validata_file.close()
  projectn_file.close()
  print ("FINISHED PARSING\n")

### Helper function to get k nearest neighbors
def kNN(tp, k):
  dsts = []
  for i in range(len(train_vs)):
    d = distance.euclidean(tp, train_vs[i])
    dsts.append((train_ls[i], d))

  # Sort dsts by shortest distance to largest distance
  dsts.sort(key=op.itemgetter(1))
  nn = []
  for sk in range(k):
    nn.append(dsts[sk][0])
  return nn

### Helper function to get k nearest neighbors for projections
def kNN_pj(tp, k):
  dsts = []
  for i in range(len(projtrain)):
    d = distance.euclidean(tp, projtrain[i])
    dsts.append((train_ls[i], d))

  # Sort dsts by shortest distance to largest distance
  dsts.sort(key=op.itemgetter(1))
  nn = []
  for sk in range(k):
    nn.append(dsts[sk][0])
  return nn

### Helper function to pick one of the kNN
def sett(cands):
  occurs = {}
  print("cands: ", cands)
  for c in range(len(cands)):
    occ = cands[c]
    if occ in occurs:
      occurs[occ] += 1
    else:
      occurs[occ] = 1
  soccurs = sorted(occurs.items(), key=op.itemgetter(1), reverse=True)
  
  f = soccurs[0][1]
  u = 1
  for e in range(len(soccurs)):
    if f != soccurs[e][1]: 
      u = 0
      break

  if u == 1 and len(soccurs) > 1:
    r = randint(0, len(soccurs)-1)
    print("r: ", r)
    print(soccurs)
    return soccurs[r][0]

  return soccurs[0][0]





#*******************#    MAIN CHUNK    #*******************#
parseData()

# Some bookkeeping for kNN
k = 9 #1 5 9 15
e = 0
start = t.time()

# Iterate thru train_vs, testd_vs, valid_vs
projtrain = np.matmul(train_vs, projmatix)
projtests = np.matmul(testd_vs, projmatix)
projvalid = np.matmul(valid_vs, projmatix)

for i in range(len(projtests)):
  print ("Point: ", i)
  print ("\n")

  tru = testd_ls[i] 
  #nbs = kNN(testd_vs[i], k)
  nbs = kNN_pj(projtests[i], k)
  prd = sett(nbs)
  print("e: ", e)
  
  if prd != tru:
    print("tru: ", tru)
    print("prd: ", prd)
    print ("ERROR!!!")
    e += 1
    print ("e: ", e)

end = t.time()
print("errors: ", e)
print("Time elapsed: ", end - start)
