import torch


def reparameterize_gaussian(mu, logvar):
    """Sample from N(mu, var)

    Parameters
    ----------
    mu : torch.Tensor
        vector of mean parameters
    logvar : torch.Tensor
        vector of log variances; only mean field approximation is currently implemented

    Returns
    -------
    torch.Tensor
        sampled vector of shape (n_sequences, sequence_length, embedding_dim)

    """
    std = torch.exp(logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)
