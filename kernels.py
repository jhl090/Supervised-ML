import operator as op
import random as rd
import time as t 

#*******************# Global data #*******************#
### Global structs for training and test datasets
train_ds = []
train_ls = []
testd_ds = []
testd_ls = []

#*******************# Helpers for parsing #*******************#
### Parse .txt files
def parseData(d_file, fn):
  if not d_file or fn < 0:
    print("Error: 'dfile' is nil or 'fn' is negative")
    return -1
  
  for line in d_file:
    lstr = line
    labl = lstr[-3:-1]
    
    if fn == 0:
      train_ds.append(lstr[0:-4])
      train_ls.append(int(labl))
    elif fn == 1: 
      testd_ds.append(lstr[0:-4])
      testd_ls.append(int(labl))

### Fill global structs
def manageData():
  training_file = open("pa4train.txt", "r")
  testdata_file = open("pa4test.txt", "r")

  for i in range(2):
    if i == 0: 
      parseData(training_file, i)

    elif i == 1: 
      parseData(testdata_file, i)

  training_file.close()
  testdata_file.close()
  print ("FINISHED PARSING\n")

#*******************# Helpers for kernelazing perceptron #*******************#
### Return # of common substr.s btwn strings s and t
def mtrKernel(s, t_set, p_len):
  slist = []

  # Get all substr.s from s, of length p
  for i in range(len(s) - p_len + 1):
    slist.append(s[i:i+p_len])

  s_set = set(slist)
  comm_s = s_set & t_set
  return len(comm_s)

### Return # of common substr.s btwn strings s and t, as a * b
def strKernel(s, t_freqs, p_len):
  slist = []
  tlist = []

  # Get all substr.s from s, of length p
  for i in range(len(s) - p_len + 1):
    slist.append(s[i:i+p_len])

  s_set = set(slist)
  slset = list(s_set)
  s_freqs = {}

  for i in range(len(slset)):
    i_cnt = slist.count(slset[i])
    s_freqs[slset[i]] = i_cnt

  comm = 0
  for key, val in s_freqs.items():
    if key in t_freqs:
      comm += (val * t_freqs[key])
  return comm

### Calculate < w_t, Ф(x) >
def sumKernels(x_t, y_t, w_set, p_num):
  if p_num < 0:
    print("Error: bad args\n")
    return -1

  # Count substrings of x_t
  tlist = []
  tfreq = {}
  for i in range(len(x_t) - p_num + 1):
    tlist.append(x_t[i:i+p_num])
  
  tset = set(tlist)
  tlset = list(tset)
  for i in range(len(tlset)):
    i_cnt = tlist.count(tlset[i])
    tfreq[tlset[i]] = i_cnt

  tot = 0
  for i in range(len(w_set)):
    y_i = train_ls[w_set[i]]
    krn = strKernel(train_ds[w_set[i]], tfreq, p_num)
    ###krn = mtrKernel(train_ds[w_set[i]], tset, p_num)
    tot += (y_t * train_ls[w_set[i]] * krn)

  return tot

### Find 2 coordinates in w_T with the highest positive values
def getExtrema(w_T, p_num):

 w_freqs = {}
 for i in range(len(w_T)):
   print("i: ", i)
   c_str = train_ds[w_T[i]]
   c_lbl = train_ls[w_T[i]]
   for j in range(len(c_str) - p_num + 1):
     s_str = c_str[j:j+p_num]
     if s_str in w_freqs:
       w_freqs[s_str] += c_lbl
     else:
       w_freqs[s_str] = c_lbl

 s_wfreqs = sorted(w_freqs.items(), key=op.itemgetter(1))
 print("s_wfreqs:")
 print(s_wfreqs)

### Run kernalized perceptron and return hyperplane classifier
def kernelPtron(p_num):
  w_t = [0]

  for t in range(1, len(train_ds)):
    print("t: ", t)
    y_l = train_ls[t]
    dp = sumKernels(train_ds[t], y_l, w_t, p_num)

    if dp <= 0:
      w_t.append(t)

  return w_t

### Get error of linear classifier from kernalized perceptron
def kernErr(w_T, p_num):
  prd = err = 0
  for i in range(len(train_ds)):
  #for i in range(len(testd_ds)):
    print("i: ", i)
    tru = train_ls[i]
    dp = sumKernels(train_ds[i], 1, w_T, p_num)
    #tru = testd_ls[i]
    #dp = sumKernels(testd_ds[i], 1, w_T, p_num)
    
    # Prediction logic
    if dp == 0: 
      if (rd.randint(0, 1) == 0):
        prd = 1
      else:
        prd = -1
    elif dp > 0:
      prd = 1
    else:
      prd = -1

    # Accuracy aggregation
    if prd != tru:
      err += 1
    print("err: ", err)

#*******************# MAIN CHUNK #*******************#
manageData()

start = t.time()
p = 5
w = kernelPtron(p)
#print("w:", w)
#print(w)
#getExtrema(w, p)
kernErr(w, p)
end = t.time()
print("Time elapsed: ", end - start)
