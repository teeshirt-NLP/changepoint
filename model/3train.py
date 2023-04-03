



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
from collections import Counter

import random
import sys
 

import csv
import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

import tensorflow as tf



#---------------------------------------------------------------------------------------------------------------
#Training/displaying settings

dispinterval = int(1e6) 	#print loss and output summary embeddings after every N iterations
niter = int(1e20) 		#Total number of iterations
nthreads = 16 			#Number of parallel threads
max_nwords = 100		#Sentences fed into tf.data are padded to ensure constant length tensors
thedtype = tf.float32		

#---------------------------------------------------------------------------------------------------------------
#Hyperparameters

embedding_size =  int(500)		#previously 498.421  --- test for 50k iter just to be sure.
initmean = 48.116			#20
initdatavariance = 573.9727	 	#200
initpriorvariance = 2.86002		#5
learningrate =  258.41458



nbatch = int(25.5558)	


#---------------------------------------------------------------------------------------------------------------
#Get data


paragraphs = load_obj("Paragraphdata-training")
len(paragraphs)


vocabulary = load_obj("BOCE.English.400K.vocab") #load pkl file
print('Num words: ', len(vocabulary))
vocabulary_size= len(vocabulary)

dictionary = dict(enumerate(vocabulary))
reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))




#---------------------------------------------------------------------------------------------------------------
#Python functions for sending the text into tensorflow



def encode_sentences(sentences, reverse_dictionary):
    encoded_sentences = []
    words_lists = [sentence.split(" ") for sentence in sentences]

    for words in words_lists:
        non_empty_words = [word for word in words if word]
        encoded_words = [reverse_dictionary[word] for word in non_empty_words if word in reverse_dictionary]
        encoded_sentences.append(encoded_words)

    return encoded_sentences


def generate_single_triplet(paragraphs, reverse_dictionary):
    span = len(paragraphs)
    target1, target2 = random.sample(range(span), 2)

    s1, s2 = paragraphs[target1], paragraphs[target2]
    v1, v2 = encode_sentences(s1, reverse_dictionary), encode_sentences(s2, reverse_dictionary)

    v1, v2 = [x for x in v1 if len(x) > 2], [x for x in v2 if len(x) > 2]

    if len(v1) > 3:
        v1_index = random.choice(range(len(v1) - 3))
        v1 = v1[v1_index:v1_index + 3]

    if len(v2) > 3:
        v2_index = random.choice(range(len(v2) - 3))
        v2 = v2[v2_index:v2_index + 3]

    return [v1, v2]


def generate_single_triplet_padded(paragraphs, reverse_dictionary, max_nwords=50):
    span = len(paragraphs)
    target1, target2 = random.sample(range(span), 2)

    s1, s2 = paragraphs[target1], paragraphs[target2]
    v1, v2 = encode_sentences(s1, reverse_dictionary), encode_sentences(s2, reverse_dictionary)

    v1, v2 = [x for x in v1 if len(x) > 2], [x for x in v2 if len(x) > 2]

    if len(v1) > 2:
        v1_index = random.choice(range(len(v1) - 2))
        v1 = v1[v1_index:v1_index + 2]

    if len(v2) > 2:
        v2_index = random.choice(range(len(v2) - 2))
        v2 = v2[v2_index:v2_index + 2]

    # Padding
    padding_value = int(1e6)
    v1 = [a + [padding_value] * (max_nwords - len(a)) for a in v1]
    v2 = [a + [padding_value] * (max_nwords - len(a)) for a in v2]

    return [v1, v2]


def generate_batch(paragraphs, reverse_dictionary, n, max_nwords=50):
    batch = []
    while len(batch) < n:
        triplet = generate_single_triplet_padded(paragraphs, reverse_dictionary, max_nwords)
        if all(len(sublist) == 2 and len(sublist[0]) == max_nwords and len(sublist[1]) == max_nwords for sublist in triplet):
            batch.append(triplet)
    return [batch]


def generate_single_triplet_raw(paragraphs):
    span = len(paragraphs)
    target1, target2 = random.sample(range(span), 2)

    s1, s2 = paragraphs[target1], paragraphs[target2]

    if len(s1) > 3:
        s1_index = random.choice(range(len(s1) - 3))
        s1 = s1



#---------------------------------------------------------------------------------------------------------------
#Graph functions



solve = lambda x: tf.math.reciprocal(x)
log_normalize = lambda x: x-tf.math.reduce_logsumexp(x)


def unPad(s1):
  mask = tf.math.less(s1, int(1e6))
  return tf.boolean_mask(s1, mask)

def constant_hazard(lambdaval, t): #t works as a normal int
  return tf.zeros(shape=[t], dtype=thedtype) + tf.cast(1/lambdaval, thedtype)


def encoder_statistical(s1):
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



def get_predprobs(muT, TT, vardata, tdata):
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



def calculate_loss(thelist):
    with tf.name_scope('Initialize_encoding') as scope:
        reshaped_list = tf.reshape(thelist, (4, max_nwords))
        bothdata = tf.map_fn(encoder_statistical, reshaped_list, dtype=thedtype)
        tdata = bothdata[:, :, 0]
        vardata = bothdata[:, :, 1]

    with tf.name_scope('Initialize_priors') as scope:
        d = tf.cast(embedding_size, thedtype)
        mu0 = tf.zeros(shape=[1, d], dtype=thedtype)
        T0 = priorvariance * tf.ones(shape=[1, d], dtype=thedtype)
        lambdaval = tf.constant(10, dtype=thedtype)
        R0 = tf.reshape(tf.constant(0.0, dtype=thedtype), shape=(1,))
        muT = mu0
        TT = T0
        output0 = [R0, muT, TT]

    with tf.name_scope('Run_BOCD') as scope:
        output1 = run_iteration(output0, vardata, tdata, lambdaval, mu0, T0, 1)
        output2 = run_iteration(output1, vardata, tdata, lambdaval, mu0, T0, 2)
        output3 = run_iteration(output2, vardata, tdata, lambdaval, mu0, T0, 3)
        output4 = run_iteration(output3, vardata, tdata, lambdaval, mu0, T0, 4)

    with tf.name_scope('Final_calculation') as scope:
        R1 = tf.constant(np.log([.1, .9]), dtype=thedtype)
        R2 = output2[0]
        R3 = output3[0]
        R4 = output4[0]

        losselements = [R1[1], R2[2], R3[1], R4[2]]
        loss = tf.math.reduce_sum(tf.math.exp(losselements))

    return loss


#---------------------------------------------------------------------------------------------------------------

graph = tf.Graph()


with graph.as_default():
  #if True:
  with tf.name_scope('Dataset') as scope:
    dataset = tf.data.Dataset.from_generator(generate_batch, args=[nbatch], output_types=(tf.int32), output_shapes=(nbatch,2,2,max_nwords) )
    dataset = dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)
    iterator = dataset.make_one_shot_iterator()
    thebatch = iterator.get_next()
  #
  #
  with tf.name_scope('MainVariables') as scope:
    embeddings = tf.Variable(
	#embedfromfile,
  	tf.random_uniform([vocabulary_size, embedding_size], -1*initmean, initmean, dtype=thedtype),
  	dtype=thedtype)
    #
    variances = tf.Variable(  	
  	tf.random_uniform([vocabulary_size, embedding_size], .9*initdatavariance, 1.1*initdatavariance, dtype=thedtype),
  	dtype=thedtype)
    #
    priorvariance = tf.abs(tf.constant(initpriorvariance, dtype=thedtype))
  #
  #
  #
  #
  #
  #
  #
  with tf.name_scope('MapandOptimize') as scope:
    lossvec = tf.map_fn(calculateloss, thebatch, dtype=thedtype)
    optimizer = tf.train.AdagradOptimizer(learningrate).minimize(-1*tf.reduce_sum(lossvec))
    primaryloss = tf.reduce_mean(lossvec) #for displaying







testing=False
#testing=True
if testing:
  sess = tf.InteractiveSession(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=11))
  sess.run(tf.global_variables_initializer())
  thelist = thebatch[0].eval()
  #
  #
  #
  from tensorflow.python.client import timeline
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()
  _, new_loss  = sess.run([optimizer, primaryloss], options=options, run_metadata=run_metadata)
  # Create the Timeline object, and write it to a json file
  fetched_timeline = timeline.Timeline(run_metadata.step_stats)
  chrome_trace = fetched_timeline.generate_chrome_trace_format()
  with open('timeline_01_cluster.json', 'w') as f:
    f.write(chrome_trace)
  #
  #
  #
  tf_tensorboard_writer = tf.summary.FileWriter('./graphs', sess.graph)
  tf_tensorboard_writer.close()
  sess.close()
  exit()


#tensorboard --logdir="./graphs"



#---------------------------------------------------------------------------------------------------------------


import time
import math
import gc
import tensorflow as tf

with tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=nthreads)) as sess:
    sess.run(tf.global_variables_initializer())
    
    old_loss = []
    mysummarized = []
    start_time = time.time()
    
    for i in range(niter + 1):
        _, new_loss = sess.run([optimizer, primaryloss])
        loss_diff = np.abs(np.mean(old_loss[-150:]) - new_loss)
        old_loss.append(new_loss)
        runningaverage = np.mean(old_loss[-150:])
        
        print(runningaverage, "\t", new_loss, "\t", i % dispinterval)

        if math.isnan(new_loss):
            print('nans')
            break

        if i > 0 and i % dispinterval == 0:
            gc.collect()

            with open('rundetails', 'w') as h:
                h.write(f'Currently at {i} iterations.\n\n')
                elapsed_time = time.time() - start_time
                h.write(f'Elapsed time: {elapsed_time}')

                if i >= dispinterval:
                    mysummarized.append(np.mean(old_loss))
                    h.write('\n\n')
                    h.write(f"Current trace: {mysummarized}")

            save_obj(old_loss, f'lossdetails{i // dispinterval}')
            old_loss = []

            embed = sess.run([embeddings])
            save_obj(embed, f"final_embeddings{i // dispinterval}")
            del embed

            gc.collect()

            varia = sess.run([variances])
            save_obj(varia, f"final_variances{i // dispinterval}")
            del varia

            gc.collect()

elapsed_time = time.time() - start_time
print(f'Elapsed time: {elapsed_time}')






#---------------------------------------------------------------------------------------------------------------





import csv
import pickle
import numpy as np
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)




def writetoR(pos):
  if False: #Get the embedding
    embed1 = load_obj('final_embeddings'+ str(pos)+'.0')
    embed1 = embed1[0]
    with open('embed'+ str(pos)+'.Withvar.csv',"w") as f:
      wr = csv.writer(f, delimiter=" ")
      wr.writerows(embed1)
      
  if True: #Diagonal variance
    var1 = load_obj('final_variances'+ str(pos)+'.0')
    var1 = np.abs(var1[0])
    with open('var'+ str(pos)+'.csv',"w") as f:
      wr = csv.writer(f, delimiter=" ")
      wr.writerows(var1)
  #
  #
  if False:
    loss = []
    for i in [1,2,3,4,5,6]:
      loss = loss + load_obj('lossdetails'+str(i)+".0")
    #loss
    with open("loss"+str(1)+".csv","w") as f:
      wr = csv.writer(f)
      wr.writerows(map(lambda x: [x], loss))
  #
  #
  if False:
    loss = load_obj("lossdetails1.0")
    with open('loss1.csv',"w") as f:
      for item in loss:
        f.write("%s\n" % item)

writetoR(6)





