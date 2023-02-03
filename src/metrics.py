import torch


def categ_cross_entropy(p_distr_hat, p_distr):

    numerical_stabilizer = 10**-20

    return - torch.sum(p_distr * torch.log(p_distr_hat+numerical_stabilizer), axis=(0,1)) / p_distr.shape[0]

def perplexity(ce):
    return torch.exp(ce)

def accuracy(p_distr_hat, p_distr):

    p_distr_hat_ = torch.argmax(p_distr_hat, axis=1)
    p_distr_ = torch.argmax(p_distr, axis=1)
    correct_predictions = torch.sum(p_distr_hat_ == p_distr_)

    return correct_predictions / p_distr.shape[0]