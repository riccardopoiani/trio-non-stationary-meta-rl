import torch
import torch.nn.functional as F

import numpy as np


class InferenceNetwork(torch.nn.Module):

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

    def forward(self, context, prior, use_prev_state=False, detach_every=None):
        if not use_prev_state:
            self.h = None
            self.seq_len = 0

        mu, logvar = self.encode(context, prior, use_prev_state)
        z = self.reparameterize(mu=mu, logvar=logvar)
        return z, mu, logvar


class EmbeddingInferenceNetwork(torch.nn.Module):

    def __init__(self, z_dim, action_dim, action_embedding_dim, state_dim, state_embedding_dim,
                 reward_embedding_dim, prior_embedding_dim, hidden_size_dim):
        super(EmbeddingInferenceNetwork, self).__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.n_in = action_embedding_dim + state_embedding_dim + reward_embedding_dim + prior_embedding_dim

        self.action_embedding_layer = torch.nn.Linear(action_dim, action_embedding_dim)
        self.state_embedding_layer = torch.nn.Linear(state_dim, state_embedding_dim)
        self.reward_embedding_layer = torch.nn.Linear(1, reward_embedding_dim)
        self.prior_embedding_layer = torch.nn.Linear(z_dim * 2, prior_embedding_dim)

        self.enc2 = torch.nn.GRU(input_size=self.n_in, hidden_size=hidden_size_dim, num_layers=1, batch_first=True)
        self.enc3 = torch.nn.Linear(hidden_size_dim + 1 + z_dim * 2,
                                    16)  # hidden + the prior + seq_len
        self.enc41 = torch.nn.Linear(16, z_dim)
        self.enc42 = torch.nn.Linear(16, z_dim)

        self.h = None
        self.seq_len = 0

    def encode(self, context, prior, use_prev_state, detach_every):
        # Compute batch number and length of the sequence
        n_batch = context.shape[0]
        seq_len = context.shape[1]

        # Data preparation
        original_prior = prior
        if len(original_prior.shape) == 1:
            original_prior = original_prior.unsqueeze(0)
        prior = prior.reshape(n_batch, 1, 2 * self.z_dim)
        prior = prior.repeat(1, seq_len, 1)

        # Embed values
        action = context[:, :, 0:self.action_dim]
        reward = context[:, :, self.action_dim:self.action_dim + 1]
        state = context[:, :, self.action_dim + 1: self.action_dim + 1 + self.state_dim]

        prior = F.elu(self.prior_embedding_layer(prior))
        action = F.elu(self.action_embedding_layer(action))
        state = F.elu(self.state_embedding_layer(state))
        reward = F.elu(self.reward_embedding_layer(reward))

        context = torch.cat([action, reward, state, prior], dim=2).view(n_batch, seq_len, self.n_in)

        if detach_every is not None:
            hidden_state = None
            for i in range(int(np.ceil(seq_len / detach_every))):
                curr_input = context[:, i * detach_every: i * detach_every + detach_every, :]
                if hidden_state is not None:
                    curr_output, hidden_state = self.enc2(curr_input, hidden_state)
                else:
                    curr_output, hidden_state = self.enc2(curr_input, hidden_state)

                hidden_state = hidden_state.detach()
            t = curr_output[:, -1, :]
        elif use_prev_state and self.h is not None:
            t, self.h = self.enc2(context, self.h)
            t = t[:, -1, :]  # we are interested only in the last output of the sequence
        else:
            t, self.h = self.enc2(context)
            t = t[:, -1, :]  # we are interested only in the last output of the sequence

        t = F.elu(t)

        self.seq_len += seq_len
        trust = 1 / self.seq_len

        trust = torch.tensor([trust], dtype=t.dtype).repeat(n_batch, 1)
        t = torch.cat([t, original_prior, trust], 1)
        t = F.elu(self.enc3(t))

        # Return encoded mu and logvar
        return self.enc41(t), self.enc42(t)

    def reparameterize(self, mu, logvar):
        # Re-parametrization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, context, prior, use_prev_state=False, detach_every=None):
        if not use_prev_state:
            self.h = None
            self.seq_len = 0

        mu, logvar = self.encode(context, prior, use_prev_state, detach_every)
        z = self.reparameterize(mu=mu, logvar=logvar)
        return z, mu, logvar

