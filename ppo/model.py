"""
Code taken and
modified from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
and https://github.com/lmzintgraf/varibad
"""
import numpy as np
import torch

from ppo.distributions import Categorical, DiagGaussian, Bernoulli
from ppo.utils import init, xavier_weights_init
from utilities.observation_utils import RunningMeanStd


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(torch.nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)
        if isinstance(self.base, RL2Base):
            output_size = base_kwargs['hidden_size']
        else:
            output_size = self.base.output_size

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(output_size, num_outputs)
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

    def update_rms(self, storage):
        if isinstance(self.base, MLPFeatureExtractor) or isinstance(self.base, MLPRL2FeatureExtractor):
            self.base.update_rms(storage)

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


class RL2Base(NNBase):
    def __init__(self, num_inputs, latent_dim=None, ext_hidden_sizes=None, hidden_size=64, use_elu=True,
                 use_env_obs=False, state_dim=0, use_xavier=False):
        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.
                               constant_(x, 0), np.sqrt(2))
        if ext_hidden_sizes is None:
            ext_hidden_sizes = (32, 16)
        else:
            assert len(ext_hidden_sizes) == 2, "Length of hidden sizes must be equal to 2 "

        super(RL2Base, self).__init__(recurrent=True, recurrent_input_size=num_inputs, hidden_size=ext_hidden_sizes[0])

        self.enc3 = init_(torch.nn.Linear(ext_hidden_sizes[0], ext_hidden_sizes[1]))
        self.enc4 = init_(torch.nn.Linear(ext_hidden_sizes[1], latent_dim))
        self.act_f = torch.nn.ELU() if use_elu else torch.nn.Tanh()

        if not use_env_obs:
            state_dim = 0
        self.state_dim = state_dim
        self.use_env_obs = use_env_obs

        if use_elu:
            self.actor = torch.nn.Sequential(
                init_(torch.nn.Linear(state_dim + latent_dim, hidden_size)), torch.nn.ELU(),
                init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.ELU())

            self.critic = torch.nn.Sequential(
                init_(torch.nn.Linear(state_dim + latent_dim, hidden_size)), torch.nn.ELU(),
                init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.ELU())
        else:
            self.actor = torch.nn.Sequential(
                init_(torch.nn.Linear(state_dim + latent_dim, hidden_size)), torch.nn.Tanh(),
                init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.Tanh())

            self.critic = torch.nn.Sequential(
                init_(torch.nn.Linear(state_dim + latent_dim, hidden_size)), torch.nn.Tanh(),
                init_(torch.nn.Linear(hidden_size, hidden_size)), torch.nn.Tanh())

        self.critic_linear = init_(torch.nn.Linear(hidden_size, 1))

        if use_xavier:
            self.enc3.apply(xavier_weights_init)
            self.enc4.apply(xavier_weights_init)
            self.actor.apply(xavier_weights_init)
            self.critic.apply(xavier_weights_init)

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x, rnn_hxs = self._forward_gru(inputs, rnn_hxs, masks)

        x = self.act_f(self.enc3(x))
        x = self.act_f(self.enc4(x))

        if self.use_env_obs and self.state_dim != 0:
            state = inputs[:, -self.state_dim:]
            x = torch.cat([state, x], 1)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)
        c = self.critic_linear(hidden_critic)

        return c, hidden_actor, rnn_hxs


class MLPRL2FeatureExtractor(NNBase):

    def __init__(self, num_inputs, state_dim, state_extractor_dim,
                 reward_extractor_dim, action_dim, action_extractor_dim,
                 norm_state, norm_action, norm_reward, has_done, done_extractor_dim=None,
                 recurrent=False, hidden_size=64, use_elu=False, use_xavier=False):
        rec_input_size = action_extractor_dim + state_extractor_dim + reward_extractor_dim + done_extractor_dim \
            if done_extractor_dim else action_extractor_dim + state_extractor_dim + reward_extractor_dim
        super(MLPRL2FeatureExtractor, self).__init__(recurrent, rec_input_size, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.has_done = has_done

        self.norm_state = norm_state
        self.norm_action = norm_action
        self.norm_reward = norm_reward

        # Normalisation
        if self.norm_state:
            self.state_rms = RunningMeanStd(shape=state_dim)
        if self.norm_action:
            self.action_rms = RunningMeanStd(shape=action_dim)
        if self.norm_reward:
            self.reward_rms = RunningMeanStd(shape=1)

        # Extractor
        self.action_extractor = init_(torch.nn.Linear(action_dim, action_extractor_dim))
        self.state_extractor = init_(torch.nn.Linear(state_dim, state_extractor_dim))
        self.reward_extractor = init_(torch.nn.Linear(1, reward_extractor_dim))
        if self.has_done:
            self.done_extractor = init_(torch.nn.Linear(1, done_extractor_dim))

        self.extractor_activation_function = torch.nn.ELU() if use_elu else torch.nn.Tanh()

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
            self.action_extractor.apply(xavier_weights_init)
            self.state_extractor.appply(xavier_weights_init)
            self.reward_extractor.apply(xavier_weights_init)
            if has_done:
                self.done_extractor.apply(xavier_weights_init)
            self.actor.apply(xavier_weights_init)
            self.critic.apply(xavier_weights_init)

        self.train()

    def update_rms(self, storage):
        obs = storage.obs[:-1]
        offset = 1 if self.has_done else 0

        action = obs[:, :, offset:offset + self.action_dim]
        reward = obs[:, :, offset + self.action_dim:1 + offset + self.action_dim]
        state = obs[:, :, 1 + offset + self.action_dim:]

        if self.norm_state:
            self.state_rms.update(state)
        if self.norm_action:
            self.action_rms.update(action)
        if self.norm_reward:
            self.reward_rms.update(reward)

    def forward(self, inputs, rnn_hxs, masks):
        if self.has_done:
            d = inputs[:, 0:1]
            a = inputs[:, 1: 1 + self.action_dim]
            r = inputs[:, 1 + self.action_dim: 2 + self.action_dim]
            s = inputs[:, 2 + self.action_dim:]
        else:
            a = inputs[:, 0: self.action_dim]
            r = inputs[:, self.action_dim:self.action_dim + 1]
            s = inputs[:, self.action_dim + 1:]

        if self.norm_state:
            s = (s - self.state_rms.mean) / torch.sqrt(self.state_rms.var + 1e-8)
        if self.norm_action:
            a = (a - self.action_rms.mean) / torch.sqrt(self.action_rms.var + 1e-8)
        if self.norm_reward:
            r = (r - self.reward_rms.mean) / torch.sqrt(self.reward_rms.var + 1e-8)

        s = self.extractor_activation_function(self.state_extractor(s))
        a = self.extractor_activation_function(self.action_extractor(a))
        r = self.extractor_activation_function(self.reward_extractor(r))

        if self.has_done:
            d = self.extractor_activation_function(d)
            x = torch.cat([d, a, r, s], 1)
        else:
            x = torch.cat([a, r, s], 1)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class MLPFeatureExtractor(NNBase):

    def __init__(self, num_inputs, latent_dim, latent_extractor_dim, state_dim, state_extractor_dim,
                 has_uncertainty, norm_state, norm_latent, uncertainty_extractor_dim=None,
                 decouple_latent_rms=False,
                 hidden_size=64, use_elu=True, use_xavier=False):
        super(MLPFeatureExtractor, self).__init__(False, num_inputs, hidden_size)

        # Check conditions
        if has_uncertainty:
            assert state_extractor_dim + latent_extractor_dim + uncertainty_extractor_dim == hidden_size, \
                "Network sizes do not match: {} + {} + {} != {}".format(state_extractor_dim, latent_extractor_dim,
                                                                        uncertainty_extractor_dim, hidden_size)
        else:
            assert state_extractor_dim + latent_extractor_dim == hidden_size, \
                "Network sizes do not match: {} + {} != {}".format(state_extractor_dim, latent_extractor_dim,
                                                                   hidden_size)
        assert state_dim != 0 and latent_dim != 0

        # Initialization
        init_ = lambda m: init(m, torch.nn.init.orthogonal_, lambda x: torch.nn.init.
                               constant_(x, 0), np.sqrt(2))

        # Generals
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.norm_state = norm_state
        self.norm_latent = norm_latent
        self.decouple_latent_rms = decouple_latent_rms
        self.has_uncertainty = has_uncertainty

        # Normalisation
        if self.norm_state:
            self.state_rms = RunningMeanStd(shape=state_dim)
        if self.norm_latent and self.has_uncertainty and not decouple_latent_rms:
            self.latent_rms = RunningMeanStd(shape=latent_dim * 2)
        elif self.norm_latent and self.has_uncertainty and decouple_latent_rms:
            self.latent_rms = RunningMeanStd(shape=latent_dim)
            self.uncertainty_rms = RunningMeanStd(shape=latent_dim)
        else:
            self.latent_rms = RunningMeanStd(shape=latent_dim)

        # Extractor
        self.latent_extractor = init_(torch.nn.Linear(latent_dim, latent_extractor_dim))
        self.state_extractor = init_(torch.nn.Linear(state_dim, state_extractor_dim))

        if has_uncertainty:
            self.uncertainty_extractor = init_(torch.nn.Linear(latent_dim, uncertainty_extractor_dim))

        self.extractor_activation_function = torch.nn.ELU() if use_elu else torch.nn.Tanh()

        # Layers
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

        # Additional init.
        if use_xavier:
            self.state_extractor.apply(xavier_weights_init)
            self.latent_extractor.apply(xavier_weights_init)
            if self.has_uncertainty:
                self.uncertainty_extractor.apply(xavier_weights_init)
            self.actor.apply(xavier_weights_init)
            self.critic.apply(xavier_weights_init)

        self.train()

    def update_rms(self, storage):
        obs = storage.obs[:-1]
        if self.norm_state:
            state = obs[:, :, 0:self.state_dim]
            self.state_rms.update(state)
        if self.norm_latent:
            latent = obs[:, :, self.state_dim:]
            if self.decouple_latent_rms:
                latent_mean = latent[:, :, 0:self.latent_dim]
                latent_u = latent[:, :, self.latent_dim:]

                self.latent_rms.update(latent_mean)
                self.uncertainty_rms.update(latent_u)
            else:
                self.latent_rms.update(latent)

    def forward(self, inputs, rnn_hxs, masks):
        # Break the input
        s = inputs[:, 0:self.state_dim]
        b = inputs[:, self.state_dim: self.state_dim + self.latent_dim]
        if self.has_uncertainty:
            u = inputs[:, self.state_dim + self.latent_dim: self.state_dim + self.latent_dim * 2]

        if self.norm_state:
            s = (s - self.state_rms.mean) / torch.sqrt(self.state_rms.var + 1e-8)

        if self.norm_latent and self.has_uncertainty and not self.decouple_latent_rms:
            l = torch.cat([b, u], 1)
            l = (l - self.latent_rms.mean) / torch.sqrt(self.latent_rms.var + 1e-8)
            b = l[:, 0:self.latent_dim]
            u = l[:, self.latent_dim:]
        elif self.norm_latent and self.has_uncertainty and self.decouple_latent_rms:
            b = (b - self.latent_rms.mean) / torch.sqrt(self.latent_rms.var + 1e-8)
            u = (u - self.uncertainty_rms.mean) / torch.sqrt(self.uncertainty_rms.var + 1e-8)
        elif self.norm_latent:
            b = (b - self.latent_rms.mean) / torch.sqrt(self.latent_rms.var + 1e-8)

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
