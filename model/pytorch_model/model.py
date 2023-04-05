import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thedtype = torch.float32

class CAPEmodel:
    def __init__(self, HYPERPARAMS):
        self.HYPERPARAMS = HYPERPARAMS

        vocabulary_size = self.HYPERPARAMS['vocabulary_size']
        embedding_size = self.HYPERPARAMS['embedding_size']
        init_mean = self.HYPERPARAMS['init_mean']

        self.embeddings = nn.Embedding(vocabulary_size, embedding_size).to(device)
        self.embeddings.weight.data.uniform_(-init_mean, init_mean)

        self.variances = nn.Embedding(vocabulary_size, embedding_size).to(device)
        init_data_variance = self.HYPERPARAMS['init_data_variance']
        self.variances.weight.data.uniform_(0.9 * init_data_variance, 1.1 * init_data_variance)

        init_prior_variance = self.HYPERPARAMS['init_prior_variance']
        self.priorvariance = torch.tensor(init_prior_variance, dtype=thedtype, device=device).abs()
        print("Model initialized")

    def get_params(self):
        return list(self.embeddings.parameters()) + list(self.variances.parameters())
