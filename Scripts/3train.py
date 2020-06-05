

#Data reading

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

import itertools
import random
import sys


import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


#---------------------------------------------------------------------------------------------------------------
#You may have to change these

dispinterval = int(1e6) 	#print loss and output summary embeddings after every N iterations
niter = int(200e6) 		#Total number of iterations



embedding_size = 100  		# Dimension of the embedding vector.
learningrate = 10 		#for Adagrad. Setting it too high improves the loss, but risks overflow


nthreads = 11 			#Number of parallel threads


max_nwords = 100		#Sentences fed into tf.data are padded to ensure constant length tensors
#---------------------------------------------------------------------------------------------------------------
#Get data



if False: #testing
  paragraphs = [paragraphs[i] for i in range(0, int(1e5))]
  save_obj(paragraphs, "smallparagraphs")

#paragraphs = load_obj("smallparagraphs") #testing

paragraphs = load_obj("Paragraphdata-training")




vocabulary = load_obj("BOCE.English.400K.vocab") #load pkl file
print('Num words: ', len(vocabulary))
vocabulary_size = len(vocabulary)

dictionary = dict(enumerate(vocabulary))
reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))


#---------------------------------------------------------------------------------------------------------------
#Python functions for sending the text into tensorflow


def by_sentence_vocab(sentences):
  encoded = []
  voca = [x.split(" ") for x in sentences]
  for j in range(len(voca)): 
    vocb = [x for x in voca[j] if x] #remove empty
    vocc = []
    for k in vocb:
      if k in reverse_dictionary:
        vocc.append(reverse_dictionary[k])
    encoded.append(vocc)
  return encoded


def generate_singlet():
  span = len(paragraphs)
  target1 = random.randint(0, span - 1) 
  target2 = random.randint(0, span - 1)
  while np.abs(target1 - target2)<1:
    target2 = random.randint(0, span - 1)
  s1 = paragraphs[target1]
  s2 = paragraphs[target2]
  s1 = [x for x in s1 if x]
  s2 = [x for x in s2 if x]
  v1 = by_sentence_vocab(s1)
  v2 = by_sentence_vocab(s2)
  v1 = [x for x in v1 if len(x)>2]
  v2 = [x for x in v2 if len(x)>2]
  if len(v1)>3:
    v1index = sorted(random.sample(range(len(v1)), 3))
    v1 = [ v1[i] for i in v1index]
  if len(v2)>3:
    v2index = sorted(random.sample(range(len(v2)), 3))
    v2 = [ v2[i] for i in v2index]
  return [v1, v2] #should look like [ v1=[[1,2,3],[4,5,6],[7,8,9]], ... ]


def generate_singlet_pad():
  span = len(paragraphs)
  target1 = random.randint(0, span - 1) 
  target2 = random.randint(0, span - 1)
  while np.abs(target1 - target2)<1:
    target2 = random.randint(0, span - 1)
  s1 = paragraphs[target1]
  s2 = paragraphs[target2]
  s1 = [x for x in s1 if x]
  s2 = [x for x in s2 if x]
  v1 = by_sentence_vocab(s1)
  v2 = by_sentence_vocab(s2)
  v1 = [x for x in v1 if len(x)>2]
  v2 = [x for x in v2 if len(x)>2]
  if len(v1)>3:
    v1index = sorted(random.sample(range(len(v1)), 3))
    v1 = [ v1[i] for i in v1index]
  if len(v2)>3:
    v2index = sorted(random.sample(range(len(v2)), 3))
    v2 = [ v2[i] for i in v2index]
  #Padding
  N=max_nwords 
  v1 = [a+[int(1e6)] * (N - len(a)) for a in v1]
  v2 = [a+[int(1e6)] * (N - len(a)) for a in v2]
  return [v1, v2]

def generate_batch(n):
  s1 = []
  while True:
    v = generate_singlet_pad()
    if len(v[0])>=3 and len(v[1])>=3 and len(v[0][0])==max_nwords and len(v[0][1])==max_nwords and len(v[0][2])==max_nwords and len(v[1][0])==max_nwords and len(v[1][1])==max_nwords and len(v[1][2])==max_nwords:
      s1.append(v)
    if len(s1)==n:
      return [s1]




#---------------------------------------------------------------------------------------------------------------
#Graph functions



import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions



def get_predprobs(muT, TT, kappaT, nuT, d, t, tdata):  #TODO: Profile  with smallparagraphs and (if successful) send to C++
  # Track both the loop index and summation in a tuple in the form (index, summation)
  # The loop condition, note the loop condition is 'i < 3'
  def condition(index, summation):
    return tf.less(index, t)
  # The loop body, this will return a result tuple in the same form (index, summation)
  def body(index, summation):
    loc = muT[index]
    sigma = (kappaT[index]+1.0)/(kappaT[index]*(nuT[index]-d+1)) * TT[index]
    df = tf.cast(nuT[index]-d+1, tf.float64)
    dv = tf.linalg.cholesky(sigma)
    mvt = tfd.MultivariateStudentTLinearOperator(
      df=df,
      loc=loc,
      scale=tf.linalg.LinearOperatorLowerTriangular(dv))
    mvt.prob(tdata)
    summand = tf.reshape(mvt.prob(tdata), (1,))
    return tf.add(index, 1), tf.concat([summation, summand], axis=0)
  # We do not care about the index value here
  a = tf.while_loop(condition, body, [tf.constant(0), tf.reshape(tf.constant(0.0, dtype=tf.float64), (1,))],
                            [tf.constant(0).get_shape(), tf.TensorShape((None,))])[1][1:]
  a = tf.reshape(a, (t,))
  return a


def constant_hazard(lambdaval, t): #t works as a normal int
  return tf.zeros(shape=[t], dtype=tf.float64) + tf.cast(1/lambdaval, tf.float64)

def expandRvector(R, predprobs, lambdaval, tnow):
  H = constant_hazard(lambdaval, tnow)
  Rmov = R * predprobs * (1-H)
  #
  S = tf.math.reduce_sum(Rmov)
  Rmovf = S / (lambdaval-1)
  #
  if tnow ==1:
    Rprime = tf.stack([tf.reshape(Rmovf, (1,)), Rmov])
  else:
    Rprime = tf.concat([tf.reshape(Rmovf, (1,)), Rmov], axis=0)
  Rprime = Rprime / tf.math.reduce_sum(Rprime)
  return tf.reshape(Rprime, (tnow+1,))



def unPad(s1):
  mask = tf.math.not_equal(s1, int(1e6))
  return tf.boolean_mask(s1, mask)






def encoder(s1):
  embed1 = tf.nn.embedding_lookup(embeddings, unPad(s1))
  return tf.reduce_mean(embed1, axis=0)





def calculateloss(thelist):
  s1 = thelist[0][0]
  s2 = thelist[0][1]
  s3 = thelist[0][2]
  s4 = thelist[1][0]
  s5 = thelist[1][1]
  s6 = thelist[1][2]
  #
  #
  tdata = tf.stack([encoder(s1),encoder(s2),encoder(s3),encoder(s4),encoder(s5),encoder(s6)])
  #Start calculating the R matrix using BOCD
  d = tf.cast(embedding_size, tf.float64)
  #
  #
  #
  #
  mu0 = tf.zeros(shape=[1,d], dtype=tf.float64)
  kappa0 = tf.reshape(tf.constant(1.0, dtype = tf.float64), shape=(1,))
  nu0 = tf.reshape(d, shape=(1,))
  T0 = tf.reshape(tf.linalg.diag(.1* tf.ones(shape=[d], dtype = tf.float64)), shape=(1,d,d))
  #
  #
  #
  #
  #Inference  
  lambdaval = tf.constant(10.0, dtype = tf.float64)
  R1 = tf.reshape(tf.constant(1.0, dtype = tf.float64), shape=(1,))
  #
  #
  muT = mu0
  kappaT = kappa0
  nuT = nu0
  TT = T0
  #---------------------------------------------------------------------------------------------------------------
  #iteration 1
  tnow=1
  predprobs1 = get_predprobs(muT, TT, kappaT, nuT, d, tnow, tdata[tnow-1]) 
  #
  #
  R2 = expandRvector(R1, predprobs1, lambdaval, tnow)
  #
  #
  #
  #
  #Updates
  kappaT0 = tf.stack([kappa0, kappaT+1])
  nuT0 = tf.stack([nu0, nuT+1])
  #
  muT0a = tf.reshape((kappaT[0]*muT[0] + tdata[tnow-1] )/ (kappaT[0] +1) , shape=(1,d))
  TT0a = TT[0] + kappaT[0]/(kappaT[0] +1) * tf.linalg.matmul( tf.reshape(muT[0]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[0]-tdata[tnow-1], (d,1))) )
  TT0a = tf.reshape(TT0a, (1,d,d))
  #
  #muT0a = tf.concat([muT0a, tf.reshape((kappaT[index]*muT[index] + tdata[tnow-1] )/ (kappaT[index] +1), (1,d))], axis=0)
  #v = TT[index] + kappaT[index]/(kappaT[index] +1) * tf.linalg.matmul( tf.reshape(muT[index]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[index]-tdata[tnow-1], (d,1))) )
  #v = tf.reshape(v, (1, d, d))
  #TT0a = tf.concat([TT0a,v], axis=0)
  #
  muT0 = tf.concat([mu0 , muT0a], axis=0)
  TT0 = tf.concat([T0 , TT0a], axis=0)
  muT = muT0
  kappaT = tf.reshape(kappaT0, (2,)) 
  nuT = tf.reshape(nuT0, (2,)) 
  TT = TT0
  #---------------------------------------------------------------------------------------------------------------
  #iteration 2
  tnow = 2
  predprobs2 = get_predprobs(muT, TT, kappaT, nuT, d, tnow, tdata[tnow-1])
  #
  #
  R3 = expandRvector(R2, predprobs2, lambdaval, tnow)
  #
  #
  #
  #Updates
  kappaT0 = tf.concat([kappa0, kappaT+1], axis=0)
  nuT0 = tf.concat([nu0, nuT+1], axis=0)
  #
  muT0a = tf.reshape((kappaT[0]*muT[0] + tdata[tnow-1] )/ (kappaT[0] +1) , shape=(1,d))
  TT0a = TT[0] + kappaT[0]/(kappaT[0] +1) * tf.linalg.matmul( tf.reshape(muT[0]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[0]-tdata[tnow-1], (d,1))) )
  TT0a = tf.reshape(TT0a, (1,d,d))
  #
  index=1
  muT0a = tf.concat([muT0a, tf.reshape((kappaT[index]*muT[index] + tdata[tnow-1] )/ (kappaT[index] +1), (1,d))], axis=0)
  v = TT[index] + kappaT[index]/(kappaT[index] +1) * tf.linalg.matmul( tf.reshape(muT[index]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[index]-tdata[tnow-1], (d,1))) )
  v = tf.reshape(v, (1, d, d))
  TT0a = tf.concat([TT0a,v], axis=0)
  #
  muT0 = tf.concat([mu0 , muT0a], axis=0)
  TT0 = tf.concat([T0 , TT0a], axis=0)
  muT = muT0
  kappaT = kappaT0
  nuT = nuT0
  TT = TT0  
  #---------------------------------------------------------------------------------------------------------------
  #iteration 3
  tnow = 3
  predprobs3 = get_predprobs(muT, TT, kappaT, nuT, d, tnow, tdata[tnow-1])
  #
  #
  R4 = expandRvector(R3, predprobs3, lambdaval, tnow)
  #
  #
  #
  #Updates
  kappaT0 = tf.concat([kappa0, kappaT+1], axis=0)
  nuT0 = tf.concat([nu0, nuT+1], axis=0)
  #
  muT0a = tf.reshape((kappaT[0]*muT[0] + tdata[tnow-1] )/ (kappaT[0] +1) , shape=(1,d))
  TT0a = TT[0] + kappaT[0]/(kappaT[0] +1) * tf.linalg.matmul( tf.reshape(muT[0]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[0]-tdata[tnow-1], (d,1))) )
  TT0a = tf.reshape(TT0a, (1,d,d))
  #
  index=1
  muT0a = tf.concat([muT0a, tf.reshape((kappaT[index]*muT[index] + tdata[tnow-1] )/ (kappaT[index] +1), (1,d))], axis=0)
  v = TT[index] + kappaT[index]/(kappaT[index] +1) * tf.linalg.matmul( tf.reshape(muT[index]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[index]-tdata[tnow-1], (d,1))) )
  v = tf.reshape(v, (1, d, d))
  TT0a = tf.concat([TT0a,v], axis=0)
  index=2
  muT0a = tf.concat([muT0a, tf.reshape((kappaT[index]*muT[index] + tdata[tnow-1] )/ (kappaT[index] +1), (1,d))], axis=0)
  v = TT[index] + kappaT[index]/(kappaT[index] +1) * tf.linalg.matmul( tf.reshape(muT[index]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[index]-tdata[tnow-1], (d,1))) )
  v = tf.reshape(v, (1, d, d))
  TT0a = tf.concat([TT0a,v], axis=0)
  #
  muT0 = tf.concat([mu0 , muT0a], axis=0)
  TT0 = tf.concat([T0 , TT0a], axis=0)
  muT = muT0
  kappaT = kappaT0
  nuT = nuT0
  TT = TT0
  matrix1 = TT[3]
  #---------------------------------------------------------------------------------------------------------------
  #iteration 4
  tnow = 4
  predprobs4 = get_predprobs(muT, TT, kappaT, nuT, d, tnow, tdata[tnow-1])
  #
  #
  R5 = expandRvector(R4, predprobs4, lambdaval, tnow)
  #
  #
  #
  #Updates
  kappaT0 = tf.concat([kappa0, kappaT+1], axis=0)
  nuT0 = tf.concat([nu0, nuT+1], axis=0)
  #
  muT0a = tf.reshape((kappaT[0]*muT[0] + tdata[tnow-1] )/ (kappaT[0] +1) , shape=(1,d))
  TT0a = TT[0] + kappaT[0]/(kappaT[0] +1) * tf.linalg.matmul( tf.reshape(muT[0]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[0]-tdata[tnow-1], (d,1))) )
  TT0a = tf.reshape(TT0a, (1,d,d))
  #
  index=1
  muT0a = tf.concat([muT0a, tf.reshape((kappaT[index]*muT[index] + tdata[tnow-1] )/ (kappaT[index] +1), (1,d))], axis=0)
  v = TT[index] + kappaT[index]/(kappaT[index] +1) * tf.linalg.matmul( tf.reshape(muT[index]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[index]-tdata[tnow-1], (d,1))) )
  v = tf.reshape(v, (1, d, d))
  TT0a = tf.concat([TT0a,v], axis=0)
  index=2
  muT0a = tf.concat([muT0a, tf.reshape((kappaT[index]*muT[index] + tdata[tnow-1] )/ (kappaT[index] +1), (1,d))], axis=0)
  v = TT[index] + kappaT[index]/(kappaT[index] +1) * tf.linalg.matmul( tf.reshape(muT[index]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[index]-tdata[tnow-1], (d,1))) )
  v = tf.reshape(v, (1, d, d))
  TT0a = tf.concat([TT0a,v], axis=0)
  index=3
  muT0a = tf.concat([muT0a, tf.reshape((kappaT[index]*muT[index] + tdata[tnow-1] )/ (kappaT[index] +1), (1,d))], axis=0)
  v = TT[index] + kappaT[index]/(kappaT[index] +1) * tf.linalg.matmul( tf.reshape(muT[index]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[index]-tdata[tnow-1], (d,1))) )
  v = tf.reshape(v, (1, d, d))
  TT0a = tf.concat([TT0a,v], axis=0)
  #
  muT0 = tf.concat([mu0 , muT0a], axis=0)
  TT0 = tf.concat([T0 , TT0a], axis=0)
  muT = muT0
  kappaT = kappaT0
  nuT = nuT0
  TT = TT0
  #---------------------------------------------------------------------------------------------------------------
  #iteration 5
  tnow = 5
  predprobs5 = get_predprobs(muT, TT, kappaT, nuT, d, tnow, tdata[tnow-1])
  #
  #
  R6 = expandRvector(R5, predprobs5, lambdaval, tnow)
  #
  #
  #
  #Updates
  kappaT0 = tf.concat([kappa0, kappaT+1], axis=0)
  nuT0 = tf.concat([nu0, nuT+1], axis=0)
  #
  muT0a = tf.reshape((kappaT[0]*muT[0] + tdata[tnow-1] )/ (kappaT[0] +1) , shape=(1,d))
  TT0a = TT[0] + kappaT[0]/(kappaT[0] +1) * tf.linalg.matmul( tf.reshape(muT[0]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[0]-tdata[tnow-1], (d,1))) )
  TT0a = tf.reshape(TT0a, (1,d,d))
  #
  index=1
  muT0a = tf.concat([muT0a, tf.reshape((kappaT[index]*muT[index] + tdata[tnow-1] )/ (kappaT[index] +1), (1,d))], axis=0)
  v = TT[index] + kappaT[index]/(kappaT[index] +1) * tf.linalg.matmul( tf.reshape(muT[index]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[index]-tdata[tnow-1], (d,1))) )
  v = tf.reshape(v, (1, d, d))
  TT0a = tf.concat([TT0a,v], axis=0)
  index=2
  muT0a = tf.concat([muT0a, tf.reshape((kappaT[index]*muT[index] + tdata[tnow-1] )/ (kappaT[index] +1), (1,d))], axis=0)
  v = TT[index] + kappaT[index]/(kappaT[index] +1) * tf.linalg.matmul( tf.reshape(muT[index]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[index]-tdata[tnow-1], (d,1))) )
  v = tf.reshape(v, (1, d, d))
  TT0a = tf.concat([TT0a,v], axis=0)
  index=3
  muT0a = tf.concat([muT0a, tf.reshape((kappaT[index]*muT[index] + tdata[tnow-1] )/ (kappaT[index] +1), (1,d))], axis=0)
  v = TT[index] + kappaT[index]/(kappaT[index] +1) * tf.linalg.matmul( tf.reshape(muT[index]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[index]-tdata[tnow-1], (d,1))) )
  v = tf.reshape(v, (1, d, d))
  TT0a = tf.concat([TT0a,v], axis=0)
  index=4
  muT0a = tf.concat([muT0a, tf.reshape((kappaT[index]*muT[index] + tdata[tnow-1] )/ (kappaT[index] +1), (1,d))], axis=0)
  v = TT[index] + kappaT[index]/(kappaT[index] +1) * tf.linalg.matmul( tf.reshape(muT[index]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[index]-tdata[tnow-1], (d,1))) )
  v = tf.reshape(v, (1, d, d))
  TT0a = tf.concat([TT0a,v], axis=0)
  #
  muT0 = tf.concat([mu0 , muT0a], axis=0)
  TT0 = tf.concat([T0 , TT0a], axis=0)
  muT = muT0
  kappaT = kappaT0
  nuT = nuT0
  TT = TT0 
  #---------------------------------------------------------------------------------------------------------------
  #iteration 6
  tnow = 6
  predprobs6 = get_predprobs(muT, TT, kappaT, nuT, d, tnow, tdata[tnow-1])
  #
  #
  R7 = expandRvector(R6, predprobs6, lambdaval, tnow)
  #
  #
  #
  #Updates
  kappaT0 = tf.concat([kappa0, kappaT+1], axis=0)
  nuT0 = tf.concat([nu0, nuT+1], axis=0)
  #
  muT0a = tf.reshape((kappaT[0]*muT[0] + tdata[tnow-1] )/ (kappaT[0] +1) , shape=(1,d))
  TT0a = TT[0] + kappaT[0]/(kappaT[0] +1) * tf.linalg.matmul( tf.reshape(muT[0]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[0]-tdata[tnow-1], (d,1))) )
  TT0a = tf.reshape(TT0a, (1,d,d))
  #
  index=1
  muT0a = tf.concat([muT0a, tf.reshape((kappaT[index]*muT[index] + tdata[tnow-1] )/ (kappaT[index] +1), (1,d))], axis=0)
  v = TT[index] + kappaT[index]/(kappaT[index] +1) * tf.linalg.matmul( tf.reshape(muT[index]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[index]-tdata[tnow-1], (d,1))) )
  v = tf.reshape(v, (1, d, d))
  TT0a = tf.concat([TT0a,v], axis=0)
  index=2
  muT0a = tf.concat([muT0a, tf.reshape((kappaT[index]*muT[index] + tdata[tnow-1] )/ (kappaT[index] +1), (1,d))], axis=0)
  v = TT[index] + kappaT[index]/(kappaT[index] +1) * tf.linalg.matmul( tf.reshape(muT[index]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[index]-tdata[tnow-1], (d,1))) )
  v = tf.reshape(v, (1, d, d))
  TT0a = tf.concat([TT0a,v], axis=0)
  index=3
  muT0a = tf.concat([muT0a, tf.reshape((kappaT[index]*muT[index] + tdata[tnow-1] )/ (kappaT[index] +1), (1,d))], axis=0)
  v = TT[index] + kappaT[index]/(kappaT[index] +1) * tf.linalg.matmul( tf.reshape(muT[index]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[index]-tdata[tnow-1], (d,1))) )
  v = tf.reshape(v, (1, d, d))
  TT0a = tf.concat([TT0a,v], axis=0)
  index=4
  muT0a = tf.concat([muT0a, tf.reshape((kappaT[index]*muT[index] + tdata[tnow-1] )/ (kappaT[index] +1), (1,d))], axis=0)
  v = TT[index] + kappaT[index]/(kappaT[index] +1) * tf.linalg.matmul( tf.reshape(muT[index]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[index]-tdata[tnow-1], (d,1))) )
  v = tf.reshape(v, (1, d, d))
  TT0a = tf.concat([TT0a,v], axis=0)
  index=5
  muT0a = tf.concat([muT0a, tf.reshape((kappaT[index]*muT[index] + tdata[tnow-1] )/ (kappaT[index] +1), (1,d))], axis=0)
  v = TT[index] + kappaT[index]/(kappaT[index] +1) * tf.linalg.matmul( tf.reshape(muT[index]-tdata[tnow-1], (d,1)) , tf.transpose(tf.reshape(muT[index]-tdata[tnow-1], (d,1))) )
  v = tf.reshape(v, (1, d, d))
  TT0a = tf.concat([TT0a,v], axis=0)
  #
  muT0 = tf.concat([mu0 , muT0a], axis=0)
  TT0 = tf.concat([T0 , TT0a], axis=0)
  muT = muT0
  kappaT = kappaT0
  nuT = nuT0
  TT = TT0
  matrix2 = TT[3]
  #
  loss = R2[1] + R3[2] + R4[3] + R5[1] + R6[2] + R7[3] 
  return loss



#---------------------------------------------------------------------------------------------------------------

#sess = tf.InteractiveSession()
graph = tf.Graph()

nbatch = 1 #how many pairs of paragraphs to contrast. This doesn't seem to increase the loss.

with graph.as_default():
  dataset = tf.data.Dataset.from_generator(generate_batch, args=[nbatch], output_types=(tf.int32), output_shapes=(nbatch,2,3,max_nwords) )
  dataset = dataset.repeat()
  iterator = dataset.make_one_shot_iterator()
  thebatch = iterator.get_next()
  #
  #
  #lambdaFunc = lambda x: calculateloss(x, embeddings)
  #lossvec = tf.map_fn(lambdaFunc, thebatch, dtype = tf.float64)
  #
  #
  embeddings = tf.Variable(
      tf.random_uniform([vocabulary_size, embedding_size], -20, 20, dtype = tf.float64))
  #sess.run(tf.global_variables_initializer()) #during testing
  loss = calculateloss( thebatch[0] )
  #loss = tf.reduce_mean(lossvec)
  #
  optimizer = tf.train.AdagradOptimizer(learningrate).minimize(-1*loss)




#---------------------------------------------------------------------------------------------------------------


import time


import math

with tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=nthreads)) as sess: #
  sess.run(tf.global_variables_initializer())
  old_loss = []
  embed_old = []
  mysummarized = []
  start_time = time.time()
  for i in range(niter+1):
    _, new_loss  = sess.run([optimizer, loss])
    loss_diff = np.abs(np.mean(old_loss[-150:]) - new_loss)
    old_loss.append(new_loss)
    runningaverage = np.mean(old_loss[-150:])
    print(runningaverage, "\t", new_loss, "\t" , i% dispinterval )
    if i>0 and i % dispinterval ==0:
      with open('rundetails', 'w') as h:
        h.write('Currently at {} iterations.'.format(i))
        h.write('\n\n')
        h.write('Elapsed time: ' + str(time.time() - start_time))
        if i >= dispinterval:
          mysummarized.append(np.mean(old_loss))
          h.write('\n\n')
          h.write("Current trace: " +str(mysummarized))
      #with open('lossdetails'+str(i/dispinterval), 'w') as g:
      #  g.write(str(old_loss))
      save_obj(old_loss, 'lossdetails'+str(i/dispinterval))
      old_loss = []
      embed = sess.run(embeddings)
      save_obj(embed, "final_embeddings"+str(i/dispinterval))


print('Elapsed time: ' + str(time.time() - start_time))





