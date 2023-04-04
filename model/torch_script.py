import torch
import numpy as np
from torch import nn
import gc
import time
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thedtype = torch.float32



def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# Graph functions
def un_pad(s1):
    mask = s1 < int(1e6)
    return s1[mask]

def constant_hazard(lambdaval, t):
    return torch.zeros(t).to(device) + torch.tensor(1 / lambdaval, dtype=thedtype, device=device)

def encoder_statistical(s1):
    z = un_pad(s1)
    embed1 = embeddings(z)
    tdata = embed1.mean(dim=0)

    var1 = torch.abs(variances(z))
    vardata = var1.sum(dim=0) / (torch.tensor(var1.shape[0], dtype=thedtype, device=device) ** 2)

    return torch.stack([tdata, vardata], dim=1)

def update_r_vector(r, predprobs, lambdaval, tnow):
    H = constant_hazard(lambdaval, tnow)
    growthprobs = r + predprobs + torch.log(1 - H)
    changeprobs = (r + predprobs + torch.log(H)).logsumexp(dim=0)

    Rprime = torch.cat([changeprobs.unsqueeze(0), growthprobs], dim=0)
    return Rprime - Rprime.logsumexp(dim=0)


def get_predprobs(muT, TT, vardata, tdata):
    sigmadiag = vardata + TT
    A = torch.tensor(2 * np.pi, dtype=thedtype, device=device).log()
    B = sigmadiag.log().sum(dim=1)
    C = ((tdata - muT) ** 2 / sigmadiag).sum(dim=1)
    return (-0.5 * (A + B + C)) - (-0.5 * (A + B + C)).logsumexp(dim=0)


def run_iteration(package, vardata, tdata, lambdaval, mu0, T0, tnow):
    ra, muT, TT = package
    predprobs1 = get_predprobs(muT, TT, vardata[tnow - 1], tdata[tnow - 1])

    rb = update_r_vector(ra, predprobs1, lambdaval, tnow)

    TT0a = 1 / (1 / TT + 1 / vardata[tnow - 1])
    muT0a = TT0a * (1 / vardata[tnow - 1] * tdata[tnow - 1] + 1 / TT * muT)
    TT = torch.cat([T0, TT0a], dim=0)
    muT = torch.cat([mu0, muT0a], dim=0)

    return [rb, muT, TT]


def calculate_loss(thelist):
    reshaped_list = thelist.view(4, max_nwords)
    bothdata = torch.stack([encoder_statistical(reshaped_list[i]) for i in range(4)], dim=0)
    tdata = bothdata[:, :, 0]
    vardata = bothdata[:, :, 1]

    #d = torch.tensor(embedding_size, dtype=thedtype, device=device)
    mu0 = torch.zeros(1, embedding_size, dtype=thedtype, device=device)
    T0 = priorvariance * torch.ones(1, embedding_size, dtype=thedtype, device=device)
    lambdaval = torch.tensor(10, dtype=thedtype, device=device)
    R0 = torch.tensor(0.0, dtype=thedtype, device=device).unsqueeze(0)
    muT = mu0
    TT = T0
    output0 = [R0, muT, TT]

    output1 = run_iteration(output0, vardata, tdata, lambdaval, mu0, T0, 1)
    output2 = run_iteration(output1, vardata, tdata, lambdaval, mu0, T0, 2)
    output3 = run_iteration(output2, vardata, tdata, lambdaval, mu0, T0, 3)
    output4 = run_iteration(output3, vardata, tdata, lambdaval, mu0, T0, 4)

    R1 = torch.tensor(np.log([.1, .9]), dtype=thedtype, device=device)
    R2 = output2[0]
    R3 = output3[0]
    R4 = output4[0]

    losselements = [R1[1], R2[2], R3[1], R4[2]]
    loss = torch.exp(torch.stack(losselements)).sum()

    return loss





from .config import HYPERPARAMS, RUNTIME_SETTINGS
from data_loading import DataLoader

loaded_data = DataLoader(RUNTIME_SETTINGS)

vocabulary_size = len(loaded_data.tokenizer)

init_data_variance = HYPERPARAMS['init_data_variance']
embedding_size = HYPERPARAMS['embedding_size']
init_prior_variance = HYPERPARAMS['init_prior_variance']
init_mean = HYPERPARAMS['init_mean']
nbatch = HYPERPARAMS['nbatch']
max_nwords = HYPERPARAMS['max_nwords']
learning_rate = HYPERPARAMS['learning_rate']
save_interval = RUNTIME_SETTINGS['save_interval']

n_iter = RUNTIME_SETTINGS['n_iter']



embeddings = nn.Embedding(vocabulary_size, embedding_size).to(device)
embeddings.weight.data.uniform_(-init_mean, init_mean)

variances = nn.Embedding(vocabulary_size, embedding_size).to(device)
variances.weight.data.uniform_(0.9 * init_data_variance, 1.1 * init_data_variance)

priorvariance = torch.tensor(init_prior_variance, dtype=thedtype, device=device).abs()

optimizer = torch.optim.Adagrad(list(embeddings.parameters()) + list(variances.parameters()), lr=learning_rate)
primaryloss = []

mysummarized = []
start_time = time.time()

for i in range(n_iter + 1):
    thebatch = torch.tensor(loaded_data.generate_batch(nbatch), dtype=torch.int64, device=device)

    optimizer.zero_grad()
    lossvec = torch.stack([calculate_loss(thebatch[j]) for j in range(nbatch)])
    loss = (-lossvec.sum())
    loss.backward()
    optimizer.step()

    new_loss = lossvec.mean().item()
    primaryloss.append(new_loss)
    runningaverage = np.mean(primaryloss[-150:])

    print(runningaverage, "\t", new_loss, "\t", i % save_interval)

    if np.isnan(new_loss):
        print('nans')
        break

    if i > 0 and i % save_interval == 0:
        gc.collect()

        with open('rundetails', 'w') as h:
            h.write(f'Currently at {i} iterations.\n\n')
            elapsed_time = time.time() - start_time
            h.write(f'Elapsed time: {elapsed_time}')

            if i >= save_interval:
                mysummarized.append(np.mean(primaryloss))
                h.write('\n\n')
                h.write(f"Current trace: {mysummarized}")

        save_obj(primaryloss, f'lossdetails{i // save_interval}')
        primaryloss = []

        embed = embeddings.weight.data.cpu().numpy()
        save_obj(embed, f"final_embeddings{i // save_interval}")
        del embed

        gc.collect()

        varia = variances.weight.data.cpu().numpy()
        save_obj(varia, f"final_variances{i // save_interval}")
        del varia

        gc.collect()

elapsed_time = time.time() - start_time
print(f'Elapsed time: {elapsed_time}')
