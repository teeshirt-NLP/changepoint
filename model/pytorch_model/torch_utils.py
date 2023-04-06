import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thedtype = torch.float32


# Graph functions
def un_pad(s1):
    mask = s1 < int(1e6)
    return s1[mask]

def constant_hazard(lambdaval, t):
    return torch.zeros(t).to(device) + torch.tensor(1 / lambdaval, dtype=thedtype, device=device)

def encoder_statistical(s1, embeddings, variances):
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


def calculate_loss(thelist, embeddings, variances, max_nwords=100, embedding_size=100, init_prior_variance=1):
    reshaped_list = thelist.view(4, max_nwords)
    bothdata = torch.stack([encoder_statistical(reshaped_list[i], embeddings, variances) for i in range(4)], dim=0)
    tdata = bothdata[:, :, 0]
    vardata = bothdata[:, :, 1]

    mu0 = torch.zeros(1, embedding_size, dtype=thedtype, device=device)
    T0 = init_prior_variance * torch.ones(1, embedding_size, dtype=thedtype, device=device)
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


