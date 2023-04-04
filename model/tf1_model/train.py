import tensorflow as tf
import numpy as np
from utils import save_obj
import time
import math
import gc


class Trainer:
    def __init__(self, model, RUNTIME_SETTINGS):
        self.model = model
        self.n_iter = RUNTIME_SETTINGS['n_iter']
        self.n_threads = RUNTIME_SETTINGS['n_threads']
        self.save_interval = RUNTIME_SETTINGS['save_interval']
        
        self.sess = tf.Session(graph=model.graph, config=tf.ConfigProto(inter_op_parallelism_threads=self.n_threads))
        self.sess.run(tf.global_variables_initializer())
        
    def train(self):
        old_loss = []
        mysummarized = []
        start_time = time.time()
        
        for i in range(self.n_iter + 1):
            _, new_loss = self.sess.run([optimizer, primaryloss])
            old_loss.append(new_loss)
            runningaverage = np.mean(old_loss[-150:])
            
            print(runningaverage, "\t", new_loss, "\t", i % self.save_interval)

            if math.isnan(new_loss):
                print('nans')
                break

            if i > 0 and i % self.save_interval == 0:
                gc.collect()

                with open('rundetails', 'w') as h:
                    h.write(f'Currently at {i} iterations.\n\n')
                    elapsed_time = time.time() - start_time
                    h.write(f'Elapsed time: {elapsed_time}')

                    if i >= self.save_interval:
                        mysummarized.append(np.mean(old_loss))
                        h.write('\n\n')
                        h.write(f"Current trace: {mysummarized}")

                save_obj(old_loss, f'lossdetails{i // self.save_interval}')
                old_loss = []

                embed = self.sess.run([embeddings])
                save_obj(embed, f"final_embeddings{i // self.save_interval}")
                del embed

                gc.collect()

                varia = self.sess.run([variances])
                save_obj(varia, f"final_variances{i // self.save_interval}")
                del varia

                gc.collect()

        elapsed_time = time.time() - start_time
        print(f'Elapsed time: {elapsed_time}')
        self.sess.close()

