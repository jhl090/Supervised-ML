from operator import add
import copy as cp
import math as mt
import numpy as np
import operator as op
import random as rd

#*******************#   Global data   #*******************#
### Global structs for training and test datasets (ds)
train_ds = []
testd_ds = []
sub_train_ds = []
sub_testd_ds = []

#*******************#   Helpers for parsing   #*******************#
### Parse .txt files
def parseData(d_file, fn):
  if not d_file or fn < 0:
    print("Error: 'dfile' is nil or 'fn' is negative")
    return -1
  
  for line in d_file:
    str = line
    fv = []
    
    v = [int(s) for s in str.split() if s.isdigit()]
    
    for i in range(820):
      fv.append(v[i])
    
    if fn == 0:
      train_ds.append(fv)

    elif fn == 1:
      testd_ds.append(fv)

### Filter training set and test set for data with label lab1 or lab2
def filterData(lab1, lab2):
  if lab1 <= 0 or lab2 <= 0:
    print("Error: 'lab1' or 'lab2' is non-positive\n")
    return -1

  # Save vectors w/labels 1 and 2, replace 2 w/-1
  for i in range(len(train_ds)):
    if train_ds[i][-1] == lab1:
      sub_train_ds.append(train_ds[i])

    elif train_ds[i][-1] == lab2:
      neg_cpy = train_ds[i]
      neg_cpy[-1] = -1
      sub_train_ds.append(neg_cpy)


  for i in range(len(testd_ds)):
    if testd_ds[i][-1] == lab1:
      sub_testd_ds.append(testd_ds[i])

    elif testd_ds[i][-1] == lab2:
     neg_cpy = testd_ds[i]
     neg_cpy[-1] = -1
     sub_testd_ds.append(neg_cpy)

### Fill global structs
def manageData():
  training_file = open("pa3train.txt", "r")
  testdata_file = open("pa3test.txt", "r")

  for i in range(2):
    if i == 0: 
      parseData(training_file, i)

    elif i == 1: 
      parseData(testdata_file, i)

  training_file.close()
  testdata_file.close()
  #filterData(1, 2)

  print ("FINISHED PARSING\n")

#*******************#   Helpers for perceptron   #*******************#
### In dset, convert all vectors whose label isn't lab to -1
def convLabels(lab, dset):
  if lab < 1: 
    print("Error: 'lab' invalid\n")
    return -1

  print("lab: ", lab)

  cdset = cp.deepcopy(dset)
  sub_cdset = []
  ctr = 0
  lab_ct = 0

  for i in range(len(cdset)):

    if cdset[i][-1] != lab:
      cdset[i][-1] = -1
      ctr += 1
    elif cdset[i][-1] == lab:
      lab_ct += 1

  return cdset

### Get feature vectors in dset
def getVectors(dset):
  if not dset: 
    print("Error: 'dset' is nil\n")
    return -1

  fvs = []
  for i in range(len(dset)):
    fvs.append(dset[i][0:-1])
  return fvs

### Run voted perceptron and return hyperplane classifier
def votedPtron(dset, pnum):
  if pnum < 1:
    print("Error: 'rounds' must be positive\n")
    return -1

  i = 1
  dset_cpy = dset
  while i < pnum:
    i += 1
    dset_cpy = dset_cpy + dset
  ds = dset_cpy

  m = 0
  carr = list(np.zeros(len(ds), dtype = int))
  carr[m] = 1
  
  w_0 = list(np.zeros(len(ds[0])-1, dtype = int))
  w_vects = list(np.zeros(len(ds), dtype = int))
  w_vects[m] = w_0

  for t in range(len(ds)):
    w_m = w_vects[m]   

    x_t = ds[t][0:-1]
    y_t = ds[t][-1]

    wm_xt = np.dot(w_m, x_t)
    dist = np.dot(y_t, wm_xt) 

    if dist <= 0:
      w_vects[m+1] = (list(map(add, w_m, np.dot(y_t, x_t))))
      m += 1
      carr[m] = 1
    else:
      carr[m] += 1
 
  cfier = []
  for i in range(m):
    cfier.append((w_vects[i], carr[i]))
  return cfier, m
    
### Run vanilla perceptron and return hyperplane classifier
def simplePtron(dset, pnum):
  if pnum < 1:
    print("Error: 'rounds' must be positive\n")
    return -1
  
  i = 1
  dset_cpy = dset
  while i < pnum:
    i += 1
    dset_cpy = dset_cpy + dset
  ds = dset_cpy

  w_0 = np.zeros(len(ds[0])-1, dtype = int)
  w_vects = [w_0]

  for t in range(len(ds)):
    w_t = w_vects[t]
      
    x_t = ds[t][0:-1] 
    y_t = ds[t][-1] 
    wxt = np.dot(w_t, x_t)
    dist = np.dot(y_t, wxt) 
    
    if dist <= 0:
      w_vects.append(list(map(add, w_t, np.dot(y_t, x_t))))
    else:
      w_vects.append(w_t)

  return w_vects[-1]

### Get error of ova multiclass classifier
def evalOva (cfiers, dset):

  cfn_mtx = np.zeros((7,6))
  mches = [0, 0, 0, 0, 0, 0]
  lablj = [0, 0, 0, 0, 0, 0]

  for i in range(len(dset)):
    print("i: ", i)
    tru = dset[i][-1]
    lablj[tru-1] += 1

    prds = []
    for k in range(len(cfiers)):
      dp = np.dot(cfiers[k], dset[i][0:-1])
      
      prd = 0
      if dp == 0:
        if (rd.randint(0, 1) == 0):
          prd = k
        else:
          prd = -1
      elif dp > 0:
        prd = k
      else: 
        prd = -1
      prds.append(prd)

    uniq = list(set(prds))
    fprd = -1
    if len(uniq) == 2:
      if uniq[0] == -1:
        fprd = uniq[1]+1
      elif uniq[1] == -1:
        fprd = uniq[0]+1

    if fprd == tru and tru != -1: 
      mches[tru-1] += 1
      cfn_mtx[fprd-1][tru-1] += 1
    elif fprd == -1:
      cfn_mtx[fprd+7][tru-1] += 1
    else:
      cfn_mtx[fprd-1][tru-1] += 1

  print("mches: ", mches)
  print("cfn_mtx1: ")
  print(cfn_mtx)
  print("\n")

  for i in range(len(cfn_mtx)):
    for j in range(len(cfn_mtx[i])):
      cfn_mtx[i][j] = cfn_mtx[i][j] / lablj[j]

  print("cfn_mtx2: ")
  print(cfn_mtx)
  
### Get error of hyperplane classifier from average perceptron
def avError(m, cfier, dset):
  print("Getting average perceptron error...\n")

  ci_wi = list(np.zeros(len(cfier[0][0]), dtype = int))
  for i in range(m):
    ci_wi = list(map(add, ci_wi, np.dot(cfier[i][0], cfier[i][1])))

  err = 0
  for i in range(len(dset)):
    tru = dset[i][-1]
    prd = 0
    dp = np.dot(ci_wi, dset[i][0:-1])

    if dp == 0:
      if (rd.randint(0, 1) == 0):
        prd = 1
      else:
        prd = -1
    elif dp > 0:
        prd = 1
    else:
        prd = -1

    if prd != tru:
      err += 1

  return err

### Get error of hyperplane classifier from voted perceptron
def vpError(m, cfier, dset):
  print("Getting voted perceptron error...\n")

  err = 0
  for i in range(len(dset)):
    tru = dset[i][-1]
    prd = summ = 0

    for j in range(m):
      dp = np.dot(cfier[j][0], dset[i][0:-1])

      if dp == 0:
        prd = rd.randint(0, 1)
        if (prd == 0):
          prd = 1
        else:
          prd = -1
      elif dp > 0:
        prd = 1
      else: 
        prd = -1
      summ += cfier[j][1] * prd

    if summ == 0:
      if (rd.randint(0, 1) == 0):
          prd = 1
      else:
          prd = -1
    elif summ > 0:
      prd = 1
    else: 
      prd = -1

    if prd != tru:
      err += 1

  return err

### Get error of hyperplane classifier from simple perceptron
def spError(thp, dset):
  print("Getting simple perceptron error...\n")

  err = 0
  for i in range(len(dset)):
    tru = dset[i][-1]
    prd = 0
    dp = np.dot(thp, dset[i][0:-1])

    # Flip a coin to predict label if req'd, else evaluate via hyperplane
    if dp == 0:
      if (rd.randint(0, 1) == 0):
        prd = 1
      else:
        prd = -1
    elif dp > 0:
      prd = 1
    else: 
      prd = -1

    if prd != tru:
      err += 1

  return err

### Get 3 coordinates in w_avg w/highest and lowest values
def getExtrema(m, cfier):
  ci_wi = list(np.zeros(len(cfier[0][0]), dtype = int))
  for i in range(m):
    ci_wi = list(map(add, ci_wi, np.dot(cfier[i][0], cfier[i][1])))

  verbose = []
  for i in range(len(ci_wi)):
    verbose.append((i, ci_wi[i]))
  
  svb = sorted(verbose, key=op.itemgetter(1))

  min_v = svb[0:3]
  max_v = svb[-3:len(svb)]

  print("min_v: ", min_v)
  print("max_v: ", max_v)

### Get k binary classifiers
def ovaReduce():
  k_cfiers = []

  for i in range(1, 7):
    conv_ova = convLabels(i, train_ds)

    hypl = simplePtron(conv_ova, 1)
    k_cfiers.append(hypl)
  
  evalOva(k_cfiers, testd_ds)

#*******************#   MAIN CHUNK   #*******************#
manageData()

### Test regular perceptron
hypl = simplePtron(sub_train_ds, 1)
sp_err = spError(hypl, sub_train_ds)

### Test voted perceptron
(cfier, m) = votedPtron(sub_train_ds, 1)
vp_err = vpError(m, cfier, sub_train_ds) 

### Test average perceptron
ap_err = avError(m, cfier, sub_train_ds)

print("sp_err: ", sp_err)
print("vp_err: ", vp_err)
print("ap_err: ", ap_err)

### Find strongest words that indicate (+)/(-) class
getExtrema(m, cfier)

### Test o.v.a. multiclass classifier
ovaReduce()
