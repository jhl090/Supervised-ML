import math as mt

#*******************# Global data #*******************#
### Global structs for training and test datasets
train_ds = []
testd_ds = []
di_words = []

#*******************# Helpers for parsing #*******************#
### Parse .txt files
def parseData(d_file, fn):  
  for line in d_file:
    line = line.rstrip("\n")
    lstr = str(line)

    v = []
    if fn == 0 or fn == 1:
      for i in range(len(lstr)):
        if lstr[i] != " " and lstr[i] != "-":
          v.append(int(lstr[i]))
        if lstr[i] == '-':
          v.append(-1)
          break
    
    if fn == 0:
      train_ds.append(v)
    elif fn == 1: 
      testd_ds.append(v)
    else:
      di_words.append(lstr)

### Fill global structs
def manageData():
  training_file = open("pa5train.txt", "r")
  testdata_file = open("pa5test.txt", "r")
  dictiony_file = open("pa5dictionary.txt", "r")

  for i in range(3):
    if i == 0: 
      parseData(training_file, i)

    elif i == 1: 
      parseData(testdata_file, i)

    else:
      parseData(dictiony_file, i)

  training_file.close()
  testdata_file.close()
  dictiony_file.close()
  print ("FINISHED PARSING\n")

#*******************# Helpers for boosting #*******************#
### Predict 1 if word i occurs in email x, -1 otherwise
def h_pos(idx, x):
  if x[idx] == 1:
    return 1
  return -1

### Predict 1 if word i doesn't occur in email x, -1 otherwise
def h_neg(idx, x):
  if x[idx] == 0:
    return 1
  return -1

### Return weak learner with highest accuracy w/respect to Dt
def minErr(errs_p, errs_n):
  pmin_v = 100
  pmin_i = -1
  for i in range(len(errs_p)):
    if errs_p[i] < pmin_v:
      pmin_v = errs_p[i]
      pmin_i = i

  nmin_v = 100
  nmin_i = -1
  for i in range(len(errs_n)):
    if errs_n[i] < nmin_v:
      nmin_v = errs_n[i]
      nmin_i = i

  # Choose classifier w/smaller error
  if pmin_v < nmin_v:
    return pmin_v, pmin_i, 1
  elif nmin_v < pmin_v:  
    return nmin_v, nmin_i, -1
  # Equal errors, so pick classifier by coin flip
  else: 
    if (rd.randint(0, 1) == 0):
      return pmin_v, pmin_i, 1
    return nmin_v, nmin_i, -1

### Get error of weak learner
def getErr(dset, dsbn, t):
  errs_p = []
  errs_n = []

  for i in range(len(di_words)):
    tot_perr = 0
    tot_nerr = 0

    # Get errors of positive classifiers
    for j in range(len(dset)):
      labl = dset[j][-1]
      hpos = h_pos(i, dset[j])
      if labl != hpos:
        tot_perr += dsbn[j]
    errs_p.append(tot_perr)

    # Get errors of negative classifiers
    for j in range(len(dset)):
      labl = dset[j][-1]
      hneg = h_neg(i, dset[j])
      if labl != hneg:
        tot_nerr += dsbn[j]
    errs_n.append(tot_nerr)

  return minErr(errs_p, errs_n)

### Update dset distribution
def updateDistr(dset, p1, p2):
  d = []
  for i in range(len(dset)):
    if p1 == p2:
      d.append(p1)
    else:
      d.append(dset[i]/p1)
  return d

### Calculate h_pos or h_neg based on flag
def hpn(idx, x, f):
  if f == 1:
    return h_pos(idx, x)
  return h_neg(idx, x)

### Compute weak learners after t rounds
def boost(dset, t):
  als = []
  fls = []
  wts = [] 

  W = [0] * len(dset)
  D = []
  D.append(updateDistr(dset, 1/len(dset), 1/len(dset)))

  for i in range(t):
    ε_t, w_t, f = getErr(dset, D[i], i)
    
    print("ε_t, w_t, f: ", ε_t, w_t, f)
    print( "di_words[w_t]: ", di_words[w_t])
    print("\n")
    
    n_αt = (mt.log((1 - ε_t)/ε_t))/(-2)
    als.append(n_αt * -1)
    fls.append(f)
    wts.append(w_t)
    for j in range(len(dset)):
      labl = dset[j][-1]
      h_t = hpn(w_t, dset[j], f)                 #1 if correct, -1 if incorrect
      W[j] = D[i][j] * mt.exp(n_αt * labl * h_t)
    Z = sum(W)
    D.append(updateDistr(W, Z, -1))

  return als, fls, wts

### Test accuracy of returned classifiers
def testCfiers(dset, als, fls, wts):
  prd = err = 0
  for i in range(len(dset)):
    tru = dset[i][-1]

    h_x = 0
    for j in range(len(als)):
      h_t = hpn(wts[j], dset[i], fls[j])
      h_x += (als[j] * h_t)

    # Prediction Logic
    if h_x == 0: 
      if (rd.randint(0, 1) == 0):
        prd = 1
      else:
        prd = -1
    elif h_x > 0:
      prd = 1
    else:
      prd = -1

    # Accuracy aggregation
    if prd != tru:
      err += 1
  print("err: ", (err/len(dset)))


#*******************# MAIN CHUNK #*******************#
manageData()
als, fls, wts = boost(train_ds, 10)
testCfiers(train_ds, als, fls, wts)
