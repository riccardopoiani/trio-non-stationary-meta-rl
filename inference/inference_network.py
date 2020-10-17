import torch
import torch.nn.functional as F


class InferenceNetwork(torch.nn.Module):
    """
    Input:
        Previous latent space
        sequence of data from the current task and old task at that point
        prior over the current task (expressed as vector mu and c)

    Output:
        - sample from the probability distribution over the latent space
        - mu and logvar from the posterior distribution
    """

    def __init__(self, n_in, z_dim, hidden_sizes=None):
        super(InferenceNetwork, self).__init__()
        self.z_dim = z_dim

        self.hidden_sizes = hidden_sizes
        if hidden_sizes is None:
            self.hidden_sizes = (32, 16)
        else:
            assert len(hidden_sizes) == 2, "Length of hidden sizes must be equal to 2 "

        self.n_in = n_in

        self.enc2 = torch.nn.GRU(input_size=n_in, hidden_size=self.hidden_sizes[0], num_layers=1, batch_first=True)
        self.enc3 = torch.nn.Linear(self.hidden_sizes[0] + z_dim * 2 + 1,
                                    self.hidden_sizes[1])  # hidden + the prior + seq_len
        self.enc41 = torch.nn.Linear(self.hidden_sizes[1], z_dim)
        self.enc42 = torch.nn.Linear(self.hidden_sizes[1], z_dim)

        self.h = None
        self.seq_len = 0

    def encode(self, context, prior, use_prev_state):
        # Compute batch number and lenght of the sequence
        n_batch = context.shape[0]
        seq_len = context.shape[1]

        # Data preparation
        original_prior = prior
        if len(original_prior.shape) == 1:
            original_prior = original_prior.unsqueeze(0)
        prior = prior.reshape(n_batch, 1, 2 * self.z_dim)
        prior = prior.repeat(1, seq_len, 1)

        context = torch.cat([context, prior], dim=2).view(n_batch, seq_len, self.n_in)

        # Data processing
        if use_prev_state and self.h is not None:
            t, self.h = self.enc2(context, self.h)
        else:
            t, self.h = self.enc2(context)
        t = t[:, -1, :]  # we are interested only in the last output of the sequence
        t = F.elu(t)

        self.seq_len += seq_len
        trust = torch.tensor([self.seq_len], dtype=t.dtype).repeat(n_batch, 1)
        t = torch.cat([t, original_prior, trust], 1)
        t = F.elu(self.enc3(t))

        # Return encoded mu and logvar
        return self.enc41(t), self.enc42(t)

    def reparameterize(self, mu, logvar):
        # Re-parametrization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, context, prior, use_prev_state=False):
        if not use_prev_state:
            self.h = None
            self.seq_len = 0

        mu, logvar = self.encode(context, prior, use_prev_state)
        z = self.reparameterize(mu=mu, logvar=logvar)
        return z, mu, logvar


class MujocoInferenceNetwork(torch.nn.Module):
    """
    Input:
        Previous latent space
        sequence of data from the current task and old task at that point
        prior over the current task (expressed as vector mu and c)

    Output:
        - sample from the probability distribution over the latent space
        - mu and logvar from the posterior distribution
    """

    def __init__(self, n_in, z_dim):
        super(MujocoInferenceNetwork, self).__init__()
        self.z_dim = z_dim

        self.n_in = n_in

        self.enc1_pre_rec = torch.nn.Linear(n_in, 32)
        self.enc2_pre_rec = torch.nn.Linear(32, 32)
        self.enc3_pre_rec = torch.nn.Linear(32, 16)

        self.enc2 = torch.nn.GRU(input_size=16, hidden_size=128, num_layers=1, batch_first=True)
        self.enc3 = torch.nn.Linear(128 + z_dim * 2 + 1,
                                    16)  # hidden + the prior + seq_len
        self.enc41 = torch.nn.Linear(16, z_dim)
        self.enc42 = torch.nn.Linear(16, z_dim)

        self.h = None
        self.seq_len = 0

    def encode(self, context, prior, use_prev_state):
        # Compute batch number and length of the sequence
        n_batch = context.shape[0]
        seq_len = context.shape[1]

        # Data preparation
        original_prior = prior
        if len(original_prior.shape) == 1:
            original_prior = original_prior.unsqueeze(0)
        prior = prior.reshape(n_batch, 1, 2 * self.z_dim)
        prior = prior.repeat(1, seq_len, 1)

        context = torch.cat([context, prior], dim=2).view(n_batch, seq_len, self.n_in)

        # Data processing
        context = self.enc1_pre_rec(context)
        context = F.elu(context)
        context = self.enc2_pre_rec(context)
        context = F.elu(context)
        context = self.enc3_pre_rec(context)
        context = F.elu(context)

        if use_prev_state and self.h is not None:
            t, self.h = self.enc2(context, self.h)
        else:
            t, self.h = self.enc2(context)
        t = t[:, -1, :]  # we are interested only in the last output of the sequence
        t = F.elu(t)

        self.seq_len += seq_len
        trust = torch.tensor([self.seq_len], dtype=t.dtype).repeat(n_batch, 1)
        t = torch.cat([t, original_prior, trust], 1)
        t = F.elu(self.enc3(t))

        # Return encoded mu and logvar
        return self.enc41(t), self.enc42(t)

    def reparameterize(self, mu, logvar):
        # Re-parametrization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, context, prior, use_prev_state=False):
        if not use_prev_state:
            self.h = None
            self.seq_len = 0

        mu, logvar = self.encode(context, prior, use_prev_state)
        z = self.reparameterize(mu=mu, logvar=logvar)
        return z, mu, logvar
