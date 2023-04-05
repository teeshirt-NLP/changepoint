import tensorflow as tf
import numpy as np


solve = lambda x: tf.math.reciprocal(x)
log_normalize = lambda x: x-tf.math.reduce_logsumexp(x)


def unPad(s1):
  mask = tf.math.less(s1, int(1e6))
  return tf.boolean_mask(s1, mask)

def constant_hazard(lambdaval, t, thedtype=tf.float32): #t works as a normal int
  return tf.zeros(shape=[t], dtype=thedtype) + tf.cast(1/lambdaval, thedtype)


def encoder_statistical(s1, embeddings, variances, thedtype=tf.float32):
  z = unPad(s1)
  embed1 = tf.nn.embedding_lookup(embeddings, z)
  tdata = tf.reduce_mean(embed1, axis=0)
  #
  #
  var1 = tf.math.abs( tf.nn.embedding_lookup(variances, z) )
  vardata = tf.reduce_sum(var1, axis=0)/tf.square( tf.cast(tf.shape(var1)[0], dtype=thedtype) ) 
  return tf.stack([tdata, vardata], axis=1)



def updateRvector(R, predprobs, lambdaval, tnow):
  H = constant_hazard(lambdaval, tnow)
  growthprobs = R + predprobs + tf.math.log(1-H)
  changeprobs = tf.math.reduce_logsumexp( R + predprobs + tf.math.log(H) )
  #
  #
  Rprime = tf.concat([tf.reshape(changeprobs, (1,)), growthprobs], axis=0)
  Rprime = tf.reshape(Rprime, (tnow+1,))
  return log_normalize(Rprime)



def get_predprobs(muT, TT, vardata, tdata, thedtype=tf.float32):
  sigmadiag = vardata + TT
  A = tf.math.log(tf.constant(2*np.pi, dtype=thedtype))
  B = tf.reduce_sum( tf.math.log(sigmadiag) , axis=1) 
  C = tf.reduce_sum( (tdata - muT)**2 / sigmadiag  , axis=1)  
  return log_normalize( -0.5*(A+B+C) )



def run_iteration(package, vardata, tdata, lambdaval, mu0, T0, tnow):
  [Ra, muT, TT] = package
  with tf.name_scope('Predprobs') as scope:
    predprobs1 = get_predprobs(muT, TT, vardata[tnow-1], tdata[tnow-1])
  #
  #
  with tf.name_scope('UpdateR') as scope:
    Rb = updateRvector(Ra, predprobs1, lambdaval, tnow)
  #
  #
  with tf.name_scope('UpdatePar') as scope:
    TT0a = solve( solve(TT) + solve(vardata[tnow-1]) )
    muT0a = tf.multiply(TT0a , tf.multiply(solve(vardata[tnow-1]), tdata[tnow-1]) + tf.multiply(solve(TT), muT ) ) 
    TT = tf.concat([T0 , TT0a], axis=0)
    muT = tf.concat([mu0 , muT0a], axis=0)
  return [Rb, muT, TT]

