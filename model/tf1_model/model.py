import tensorflow as tf
from tf_utils import encoder_statistical, run_iteration
import numpy as np


class CAPEmodel:
    def __init__(self, loaded_data, HYPERPARAMS):
        self.thedtype = tf.float32
        self.vocabulary_size = len(loaded_data.tokenizer)
        self.init_data_variance = HYPERPARAMS['init_data_variance']
        self.embedding_size = HYPERPARAMS['embedding_size']
        self.init_prior_variance = HYPERPARAMS['init_prior_variance']
        self.init_mean = HYPERPARAMS['init_mean']
        self.nbatch = HYPERPARAMS['nbatch']
        self.max_nwords = HYPERPARAMS['max_nwords']
        self.learning_rate = HYPERPARAMS['learning_rate']

        self.loaded_data = loaded_data

        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_model()

    def _build_model(self):
        with tf.name_scope('Dataset') as scope:
            dataset = tf.data.Dataset.from_generator(self.loaded_data.generate_batch, args=[self.nbatch], output_types=tf.int32, output_shapes=(self.nbatch,2,2,self.max_nwords) )
            dataset = dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)
            iterator = dataset.make_one_shot_iterator()
            thebatch = iterator.get_next()
        
        with tf.name_scope('MainVariables') as scope:
            self.embeddings = tf.Variable(
            #embedfromfile,
            tf.random_uniform([self.vocabulary_size, self.embedding_size], -1*self.init_mean, self.init_mean, dtype=self.thedtype),
            dtype=self.thedtype)
            #
            self.variances = tf.Variable(  	
            tf.random_uniform([self.vocabulary_size, self.embedding_size], .9*self.init_data_variance, 1.1*self.init_data_variance, dtype=self.thedtype),
            dtype=self.thedtype)
            #
            self.priorvariance = tf.abs(tf.constant(self.init_prior_variance, dtype=self.thedtype))

        with tf.name_scope('MapandOptimize') as scope:
            lossvec = tf.map_fn(self.calculate_loss, thebatch, dtype=self.thedtype)
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(-1*tf.reduce_sum(lossvec))
            self.primaryloss = tf.reduce_mean(lossvec)

    def calculate_loss(self, thelist):
        embedding_size = tf.shape(self.embeddings)[0]
        with tf.name_scope('Initialize_encoding') as scope:
            reshaped_list = tf.reshape(thelist, (4, self.max_nwords))
            bothdata = tf.map_fn(encoder_statistical, reshaped_list, self.embeddings, self.variances, dtype=self.thedtype)
            tdata = bothdata[:, :, 0]
            vardata = bothdata[:, :, 1]

        with tf.name_scope('Initialize_priors') as scope:
            d = tf.cast(embedding_size, self.thedtype)
            mu0 = tf.zeros(shape=[1, d], dtype=self.thedtype)
            T0 = self.priorvariance * tf.ones(shape=[1, d], dtype=self.thedtype)
            lambdaval = tf.constant(10, dtype=self.thedtype)
            R0 = tf.reshape(tf.constant(0.0, dtype=self.thedtype), shape=(1,))
            muT = mu0
            TT = T0
            output0 = [R0, muT, TT]

        with tf.name_scope('Run_BOCD') as scope:
            output1 = run_iteration(output0, vardata, tdata, lambdaval, mu0, T0, 1)
            output2 = run_iteration(output1, vardata, tdata, lambdaval, mu0, T0, 2)
            output3 = run_iteration(output2, vardata, tdata, lambdaval, mu0, T0, 3)
            output4 = run_iteration(output3, vardata, tdata, lambdaval, mu0, T0, 4)

        with tf.name_scope('Final_calculation') as scope:
            R1 = tf.constant(np.log([.1, .9]), dtype=self.thedtype)
            R2 = output2[0]
            R3 = output3[0]
            R4 = output4[0]

            losselements = [R1[1], R2[2], R3[1], R4[2]]
            loss = tf.math.reduce_sum(tf.math.exp(losselements))

        return loss


    
