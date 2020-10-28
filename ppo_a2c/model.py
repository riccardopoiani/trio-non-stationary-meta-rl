"""
Code taken from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
"""
import numpy as np
import torch

from ppo_a2c.distributions import Categorical, DiagGaussian, Bernoulli
from ppo_a2c.utils import init, xavier_weights_init


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(torch.nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(torch.nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = torch.nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    torch.nn.init.constant_(param, 0)
                elif 'weight' in name:
                    torch.nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.
                               constant_(x, 0), torch.nn.init.calculate_gain('relu'))

        self.main = torch.nn.Sequential(
            init_(torch.nn.Conv2d(num_inputs, 32, 8, stride=4)), torch.nn.ReLU(),
            init_(torch.nn.Conv2d(32, 64, 4, stride=2)), torch.nn.ReLU(),
            init_(torch.nn.Conv2d(64, 32, 3, stride=1)), torch.nn.ReLU(), Flatten(),
            init_(torch.nn.Linear(32 * 7 * 7, hidden_size)), torch.nn.ReLU())

        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(torch.nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64, use_elu=True, use_xavier=False):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.
                               constant_(x, 0), np.sqrt(2))

        if use_elu:
            self.actor = torch.nn.Sequential(
                init_(torch.nn.Linear(num_inputs, hidden_size)), torch.nn.ELU(),
                init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.ELU())

            self.critic = torch.nn.Sequential(
                init_(torch.nn.Linear(num_inputs, hidden_size)), torch.nn.ELU(),
                init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.ELU())
        else:
            self.actor = torch.nn.Sequential(
                init_(torch.nn.Linear(num_inputs, hidden_size)), torch.nn.Tanh(),
                init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.Tanh())

            self.critic = torch.nn.Sequential(
                init_(torch.nn.Linear(num_inputs, hidden_size)), torch.nn.Tanh(),
                init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.Tanh())

        self.critic_linear = init_(torch.nn.Linear(hidden_size, 1))

        if use_xavier:
            self.actor.apply(xavier_weights_init)
            self.critic.apply(xavier_weights_init)

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class ImprovedMLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(ImprovedMLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = torch.nn.Sequential(
            init_(torch.nn.Linear(num_inputs, hidden_size)), torch.nn.ELU(),
            init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.ELU(),
            init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.ELU())

        self.critic = torch.nn.Sequential(
            init_(torch.nn.Linear(num_inputs, hidden_size)), torch.nn.ELU(),
            init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.ELU(),
            init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.ELU())

        self.critic_linear = init_(torch.nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class MLPFeatureExtractor(NNBase):
    def __init__(self, num_inputs, latent_dim, latent_extractor_dim, state_dim, state_extractor_dim,
                 has_uncertainty, uncertainty_extractor_dim=None, hidden_size=64, use_elu=True, use_xavier=False):
        super(MLPFeatureExtractor, self).__init__(False, num_inputs, hidden_size)

        if has_uncertainty:
            assert state_extractor_dim + latent_extractor_dim + uncertainty_extractor_dim == hidden_size, \
                "Network sizes do not match: {} + {} + {} != {}".format(state_extractor_dim, latent_extractor_dim,
                                                                        uncertainty_extractor_dim, hidden_size)
        else:
            assert state_extractor_dim + latent_extractor_dim == hidden_size, \
                "Network sizes do not match: {} + {} != {}".format(state_extractor_dim, latent_extractor_dim, hidden_size)
        assert state_dim != 0 and latent_dim != 0

        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.has_uncertainty = has_uncertainty

        self.latent_extractor = torch.nn.Linear(latent_dim, latent_extractor_dim)
        self.state_extractor = torch.nn.Linear(state_dim, state_extractor_dim)

        if has_uncertainty:
            self.uncertainty_extractor = torch.nn.Linear(latent_dim, uncertainty_extractor_dim)

        self.extractor_activation_function = torch.nn.ELU() if use_elu else torch.nn.Tanh()

        if use_elu:
            self.actor = torch.nn.Sequential(
                init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.ELU(),
                init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.ELU())

            self.critic = torch.nn.Sequential(
                init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.ELU(),
                init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.ELU())
        else:
            self.actor = torch.nn.Sequential(
                init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.Tanh(),
                init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.Tanh())

            self.critic = torch.nn.Sequential(
                init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.Tanh(),
                init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.Tanh())

        self.critic_linear = init_(torch.nn.Linear(hidden_size, 1))

        if use_xavier:
            self.actor.apply(xavier_weights_init)
            self.critic.apply(xavier_weights_init)

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        # Break the input
        s = inputs[:, 0:self.state_dim]
        b = inputs[:, self.state_dim: self.state_dim + self.latent_dim]
        if self.has_uncertainty:
            u = inputs[:, self.state_dim + self.latent_dim: self.state_dim + self.latent_dim * 2]

        # Encode the input
        s = self.extractor_activation_function(self.state_extractor(s))
        b = self.extractor_activation_function(self.latent_extractor(b))
        if self.has_uncertainty:
            u = self.extractor_activation_function(self.uncertainty_extractor(u))
            x = torch.cat([s, b, u], 1)
        else:
            x = torch.cat([s, b], 1)

        # Apply policy network
        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
