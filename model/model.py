import tensorflow as tf
from tf_utils import calculate_loss

class CAPEmodel:
    def __init__(self, loaded_data, HYPERPARAMS):
        self.thedtype = HYPERPARAMS['thedtype']
        self.vocabulary_size = HYPERPARAMS['thedtype']
        self.init_data_variance = HYPERPARAMS['init_data_variance']
        self.embedding_size = HYPERPARAMS['embedding_size']
        self.init_prior_variance = HYPERPARAMS['init_prior_variance']
        self.init_mean = HYPERPARAMS['init_mean']
        self.nbatch = HYPERPARAMS['nbatch']
        self.max_nwords = HYPERPARAMS['max_nwords']
        self.learning_rate = HYPERPARAMS['learning_rate']

        self.data_loader = loaded_data

        self.graph = tf.Graph()
        with self.graph.as_default():
            self._build_model()

    def _build_model(self):
        with tf.name_scope('Dataset') as scope:
            dataset = tf.data.Dataset.from_generator(self.data_loader.generate_batch, args=[self.nbatch], output_types=(tf.int32), output_shapes=(self.nbatch,2,2,self.max_nwords) )
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
            priorvariance = tf.abs(tf.constant(self.init_prior_variance, dtype=self.thedtype))

        with tf.name_scope('MapandOptimize') as scope:
            lossvec = tf.map_fn(calculate_loss, thebatch, self.embeddings, self.variances, priorvariance, self.max_nwords, dtype=self.thedtype)
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(-1*tf.reduce_sum(lossvec))
            self.primaryloss = tf.reduce_mean(lossvec)


    def predict(self):
        return
    
