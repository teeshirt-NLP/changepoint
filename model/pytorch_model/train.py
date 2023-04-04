import torch
import numpy as np
import gc
import time
from .torch_utils import calculate_loss
from .utils import save_obj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thedtype = torch.float32

class Trainer:
    def __init__(self, model, RUNTIME_SETTINGS, HYPERPARAMS):
        self.model = model
        learning_rate = HYPERPARAMS['learning_rate']
        self.optimizer = torch.optim.Adagrad(model.get_params(), lr=learning_rate)
        self.primaryloss = []
        self.mysummarized = []
        self.start_time = time.time()
        self.RUNTIME_SETTINGS = RUNTIME_SETTINGS
        self.HYPERPARAMS = HYPERPARAMS

    def train(self, loaded_data):
        n_iter = self.RUNTIME_SETTINGS['n_iter']
        nbatch = self.HYPERPARAMS['nbatch']
        max_nwords = self.HYPERPARAMS['max_nwords']
        embedding_size = self.HYPERPARAMS['embedding_size']
        init_prior_variance = self.HYPERPARAMS['init_prior_variance']
        save_interval = self.RUNTIME_SETTINGS['save_interval']

        for i in range(n_iter + 1):
            thebatch = torch.tensor(loaded_data.generate_batch(nbatch), dtype=torch.int64, device=device)

            self.optimizer.zero_grad()
            lossvec = torch.stack([calculate_loss(thebatch[j], self.model.embeddings, self.model.variances,
                                                  max_nwords=max_nwords, embedding_size=embedding_size, init_prior_variance=init_prior_variance) for j in range(nbatch)])
            loss = (-lossvec.sum())
            loss.backward()
            self.optimizer.step()

            new_loss = lossvec.mean().item()
            self.primaryloss.append(new_loss)
            runningaverage = np.mean(self.primaryloss[-150:])

            print(runningaverage, "\t", new_loss, "\t", i % save_interval)

            if np.isnan(new_loss):
                print('nans')
                break

            if i > 0 and i % save_interval == 0:
                gc.collect()

                with open('rundetails', 'w') as h:
                    h.write(f'Currently at {i} iterations.\n\n')
                    elapsed_time = time.time() - self.start_time
                    h.write(f'Elapsed time: {elapsed_time}')

                    if i >= save_interval:
                        self.mysummarized.append(np.mean(self.primaryloss))
                        h.write('\n\n')
                        h.write(f"Current trace: {self.mysummarized}")

                save_obj(self.primaryloss, f'lossdetails{i // save_interval}')
                self.primaryloss = []

                embed = self.model.embeddings.weight.data.cpu().numpy()
                save_obj(embed, f"final_embeddings{i // save_interval}")
                del embed

                gc.collect()

                varia = self.model.variances.weight.data.cpu().numpy()
                save_obj(varia, f"final_variances{i // save_interval}")
                del varia

                gc.collect()

        elapsed_time = time.time() - self.start_time
        print(f'Elapsed time: {elapsed_time}')
