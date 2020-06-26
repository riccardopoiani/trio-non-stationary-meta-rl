import torch
from torch.distributions.normal import Normal
import torch.nn.functional as F

from collections import OrderedDict


class MetaNetwork(torch.nn.Module):

    def __init__(self, n_in, n_out):
        super(MetaNetwork, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        # Meta learning stuff
        self.named_meta_parameters = self.named_parameters
        self.meta_parameters = self.parameters

    def update_parameters(self, loss, params=None, step_size=0.5, first_order=False):
        if params is None:
            params = OrderedDict(self.named_meta_parameters())

        grads = torch.autograd.grad(loss, params.values(), create_graph=not first_order)

        update_params = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            update_params[name] = param - step_size * grad

        return update_params


class MetaNetworkWithPertubation(MetaNetwork):

    def __init__(self, n_in, n_out,
                 init_mean: torch.tensor, init_std: torch.tensor,
                 hidden_sizes=(),
                 nonlinearity=F.relu,
                 ):
        super(MetaNetworkWithPertubation, self).__init__(n_in=n_in, n_out=n_out)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (n_in,) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i), torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

        self.output_layer = torch.nn.Linear(layer_sizes[-1], n_out)

        # Perturbation stuff (ignore for now)
        self.perturbation_distribution = Normal(init_mean, init_std)

    def forward(self, x, params=None, perturbation=False):
        if params is None:
            params = OrderedDict(self.named_parameters())

        for i in range(1, self.num_layers):
            x = F.linear(x, weight=params['layer{0}.weight'.format(i)],
                         bias=params['layer{0}.bias'.format(i)])
            x = self.nonlinearity(x)

        if perturbation:
            new_w = params['output_layer.weight'] * self.perturbation_distribution.sample()
            x = F.linear(x, weight=new_w, bias=params['output_layer.bias'])
        else:
            x = F.linear(x, weight=params['output_layer.weight'], bias=params['output_layer.bias'])

        return x


class NoisyMetaNetwork(MetaNetwork):

    def __init__(self, n_in, n_out,
                 init_mean_w_output: torch.tensor, init_std_w_output: torch.tensor,
                 init_mean_b_output: torch.tensor, init_std_b_output: torch.tensor,
                 init_mean_w_input: torch.tensor, init_std_w_input: torch.tensor,
                 init_mean_b_input: torch.tensor, init_std_b_input: torch.tensor,
                 init_input_mean: torch.tensor, init_input_std: torch.tensor,
                 hidden_sizes=(),
                 nonlinearity=F.relu,
                 ):
        super(NoisyMetaNetwork, self).__init__(n_in=n_in, n_out=n_out)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (n_in,) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i), torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

        self.output_layer = torch.nn.Linear(layer_sizes[-1], n_out)

        # Perturbation stuff: noise layer and perturbation parameters
        self.add_module('noise_input', torch.nn.Linear(n_in, layer_sizes[1]))
        self.add_module('noise_output', torch.nn.Linear(layer_sizes[-1], n_out))
        self.w_perturbation_dist_output = Normal(init_mean_w_output, init_std_w_output)
        self.b_perturbation_dist_output = Normal(init_mean_b_output, init_std_b_output)
        # self.w_perturbation_dist_input = Normal(init_mean_w_input, init_std_w_input)
        # self.b_perturbation_dist_input = Normal(init_mean_b_input, init_std_b_input)
        self.input_perturbation = Normal(init_input_mean, init_input_std)

    def forward(self, x, params=None, perturbation=False):
        if params is None:
            params = OrderedDict(self.named_parameters())

        if perturbation:
            sample = self.input_perturbation.sample()
            x = x + sample

        for i in range(1, self.num_layers):
            x = F.linear(x, weight=params['layer{0}.weight'.format(i)], bias=params['layer{0}.bias'.format(i)])
            x = self.nonlinearity(x)

        if perturbation:
            standard_output = F.linear(x, weight=params['output_layer.weight'], bias=params['output_layer.bias'])

            samples_w = self.w_perturbation_dist_output.sample()
            new_w = params['noise_output.weight'] * samples_w

            samples_b = self.b_perturbation_dist_output.sample()
            new_b = params['noise_output.bias'] * samples_b

            pert_x = F.linear(x, weight=new_w, bias=new_b)
            x = standard_output + pert_x
        else:
            x = F.linear(x, weight=params['output_layer.weight'], bias=params['output_layer.bias'])

        return x

    def update_parameters(self, loss, params=None, step_size=0.5, first_order=False,
                          remove_noise_param=True):
        if params is None:
            params = OrderedDict(self.named_meta_parameters())

        if remove_noise_param:
            if params.__contains__('noise_output.bias'):
                del params['noise_output.bias']
            if params.__contains__('noise_output.weight'):
                del params['noise_output.weight']
            if params.__contains__('noise_input.bias'):
                del params['noise_input.bias']
            if params.__contains__('noise_input.weight'):
                del params['noise_input.weight']

        return super().update_parameters(loss=loss, params=params, step_size=step_size,
                                         first_order=first_order)

    def set_learn_net(self, state=True):
        params = OrderedDict(self.named_meta_parameters())
        for key in params:
            if key not in ['noise_output.weight', 'noise_output.bias', 'noise_input.weight', 'noise_input.bias']:
                params[key].requires_grad = state

    def set_learn_noise(self, state=True):
        params = OrderedDict(self.named_meta_parameters())
        params['noise_output.weight'].requires_grad = state
        params['noise_output.bias'].requires_grad = state
        params['noise_input.weight'].requires_grad = state
        params['noise_input.bias'].requires_grad = state

    def set_noise_layer(self, new_params):
        curr_params = OrderedDict(self.named_meta_parameters())
        curr_params['noise_output.weight'] = new_params['noise_output.weight']
        curr_params['noise_output.bias'] = new_params['noise_output.bias']
        curr_params['noise_input.weight'] = new_params['noise_input.weight']
        curr_params['noise_input.bias'] = new_params['noise_input.bias']

    def merge_parameters(self, params):
        curr_params = OrderedDict(self.named_meta_parameters())
        for key in curr_params:
            curr_params[key] += params[key]


class NoisyMetaNetwork_IO(MetaNetwork):

    def __init__(self, n_in, n_out,
                 init_mean_w_output: torch.tensor, init_std_w_output: torch.tensor,
                 init_mean_b_output: torch.tensor, init_std_b_output: torch.tensor,
                 init_mean_w_input: torch.tensor, init_std_w_input: torch.tensor,
                 init_mean_b_input: torch.tensor, init_std_b_input: torch.tensor,
                 hidden_sizes=(),
                 nonlinearity=F.relu,
                 w_init=1, b_init=1
                 ):
        super(NoisyMetaNetwork_IO, self).__init__(n_in=n_in, n_out=n_out)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (n_in,) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i), torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

        self.output_layer = torch.nn.Linear(layer_sizes[-1], n_out)

        # Perturbation stuff: noise layer and perturbation parameters
        self.add_module('noise_input', torch.nn.Linear(n_in, layer_sizes[1]))
        self.add_module('noise_output', torch.nn.Linear(layer_sizes[-1], n_out))

        p = OrderedDict(self.named_parameters())
        torch.nn.init.constant_(p['noise_input.weight'], w_init)
        torch.nn.init.constant_(p['noise_input.bias'], b_init)
        torch.nn.init.constant_(p['noise_output.weight'], w_init)
        torch.nn.init.constant_(p['noise_output.bias'], b_init)

        self.w_perturbation_dist_output = Normal(init_mean_w_output, init_std_w_output)
        self.b_perturbation_dist_output = Normal(init_mean_b_output, init_std_b_output)
        self.w_perturbation_dist_input = Normal(init_mean_w_input, init_std_w_input)
        self.b_perturbation_dist_input = Normal(init_mean_b_input, init_std_b_input)

    def forward(self, x, params=None, perturbation=False):
        if params is None:
            params = OrderedDict(self.named_parameters())

        for i in range(1, self.num_layers):
            if i == 1 and perturbation:
                standard_x = F.linear(x, weight=params['layer1.weight'], bias=params['layer1.bias'])

                samples_w = self.w_perturbation_dist_input.sample()
                samples_b = self.b_perturbation_dist_input.sample()
                new_w = params['noise_input.weight'] * samples_w
                new_b = params['noise_input.bias'] * samples_b
                pert_x = F.linear(x, weight=new_w, bias=new_b)

                x = standard_x + pert_x
                x = self.nonlinearity(x)
            else:
                x = F.linear(x, weight=params['layer{0}.weight'.format(i)], bias=params['layer{0}.bias'.format(i)])
                x = self.nonlinearity(x)

        if perturbation:
            standard_output = F.linear(x, weight=params['output_layer.weight'], bias=params['output_layer.bias'])

            samples_w = self.w_perturbation_dist_output.sample()
            new_w = params['noise_output.weight'] * samples_w

            samples_b = self.b_perturbation_dist_output.sample()
            new_b = params['noise_output.bias'] * samples_b

            pert_x = F.linear(x, weight=new_w, bias=new_b)
            x = standard_output + pert_x
        else:
            x = F.linear(x, weight=params['output_layer.weight'], bias=params['output_layer.bias'])

        return x

    def update_parameters(self, loss, params=None, step_size=0.5, first_order=False,
                          remove_noise_param=True):
        if params is None:
            params = OrderedDict(self.named_meta_parameters())

        if remove_noise_param:
            if params.__contains__('noise_output.bias'):
                del params['noise_output.bias']
            if params.__contains__('noise_output.weight'):
                del params['noise_output.weight']
            if params.__contains__('noise_input.bias'):
                del params['noise_input.bias']
            if params.__contains__('noise_input.weight'):
                del params['noise_input.weight']

        print(params)

        return super().update_parameters(loss=loss, params=params, step_size=step_size,
                                         first_order=first_order)

    def set_learn_net(self, state=True):
        params = OrderedDict(self.named_meta_parameters())
        for key in params:
            if key not in ['noise_output.weight', 'noise_output.bias', 'noise_input.weight', 'noise_input.bias']:
                params[key].requires_grad = state

    def set_learn_noise(self, state=True):
        params = OrderedDict(self.named_meta_parameters())
        params['noise_output.weight'].requires_grad = state
        params['noise_output.bias'].requires_grad = state
        params['noise_input.weight'].requires_grad = state
        params['noise_input.bias'].requires_grad = state

    def set_noise_layer(self, new_params):
        curr_params = OrderedDict(self.named_meta_parameters())
        curr_params['noise_output.weight'] = new_params['noise_output.weight']
        curr_params['noise_output.bias'] = new_params['noise_output.bias']
        curr_params['noise_input.weight'] = new_params['noise_input.weight']
        curr_params['noise_input.bias'] = new_params['noise_input.bias']

    def merge_parameters(self, params):
        curr_params = OrderedDict(self.named_meta_parameters())
        for key in curr_params:
            curr_params[key] += params[key]


class MetaContainer(MetaNetwork):

    def __init__(self, n_in, n_out,
                 init_mean_w_output, init_std_w_output,
                 init_mean_w_input, init_std_w_input,
                 init_mean_b_input, init_std_b_input,
                 init_mean_b_output, init_std_b_output,
                 new_space_size=10,
                 hidden_sizes=(),
                 nonlinearity=F.relu,
                 w_init=1,
                 b_init=1):
        super(MetaContainer, self).__init__(n_in=n_in, n_out=n_out)

        # Core network parameters
        self.nonlinearity = nonlinearity
        self.hidden_sizes = hidden_sizes

        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (n_in,) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i), torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

        self.output_layer = torch.nn.Linear(layer_sizes[-1], n_out)

        # Perturbation parameters
        self.add_module('noise_input', torch.nn.Linear(n_in, layer_sizes[1]))
        self.add_module('noise_output', torch.nn.Linear(layer_sizes[-1], n_out))

        p = OrderedDict(self.named_parameters())
        torch.nn.init.constant_(p['noise_input.weight'], w_init)
        torch.nn.init.constant_(p['noise_input.bias'], b_init)
        torch.nn.init.constant_(p['noise_output.weight'], w_init)
        torch.nn.init.constant_(p['noise_output.bias'], b_init)

        self.w_perturbation_dist_output = Normal(init_mean_w_output, init_std_w_output)
        self.b_perturbation_dist_output = Normal(init_mean_b_output, init_std_b_output)
        self.w_perturbation_dist_input = Normal(init_mean_w_input, init_std_w_input)
        self.b_perturbation_dist_input = Normal(init_mean_b_input, init_std_b_input)

        # Feature transformation: a small network at the beginning and after the meta one, to be meta-trained
        self.add_module("meta_in_1", torch.nn.Linear(n_in, new_space_size))
        self.add_module("meta_in_2", torch.nn.Linear(new_space_size, n_in))

        self.add_module("meta_out_1", torch.nn.Linear(n_out, new_space_size))
        self.add_module("meta_out_2", torch.nn.Linear(new_space_size, n_out))

        self.activate_meta = False

    def forward(self, x, params=None, perturbation=False):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Meta input
        if self.activate_meta:
            x = F.linear(x, weight=params['meta_in_1.weight'], bias=params['meta_in_1.bias'])
            x = F.linear(x, weight=params['meta_in_2.weight'], bias=params['meta_in_2.bias'])

        # Core network
        for i in range(1, self.num_layers):
            if i == 1 and perturbation:
                standard_x = F.linear(x, weight=params['layer1.weight'], bias=params['layer1.bias'])

                samples_w = self.w_perturbation_dist_input.sample()
                samples_b = self.b_perturbation_dist_input.sample()
                new_w = params['noise_input.weight'] * samples_w
                new_b = params['noise_input.bias'] * samples_b
                pert_x = F.linear(x, weight=new_w, bias=new_b)

                x = standard_x + pert_x
                x = self.nonlinearity(x)
            else:
                x = F.linear(x, weight=params['layer{0}.weight'.format(i)], bias=params['layer{0}.bias'.format(i)])
                x = self.nonlinearity(x)

        if perturbation:
            standard_output = F.linear(x, weight=params['output_layer.weight'], bias=params['output_layer.bias'])

            samples_w = self.w_perturbation_dist_output.sample()
            new_w = params['noise_output.weight'] * samples_w

            samples_b = self.b_perturbation_dist_output.sample()
            new_b = params['noise_output.bias'] * samples_b

            pert_x = F.linear(x, weight=new_w, bias=new_b)
            x = standard_output + pert_x
        else:
            x = F.linear(x, weight=params['output_layer.weight'], bias=params['output_layer.bias'])

        # Meta output
        if self.activate_meta:
            x = F.linear(x, weight=params['meta_out_1.weight'], bias=params['meta_out_1.bias'])
            x = F.linear(x, weight=params['meta_out_2.weight'], bias=params['meta_out_2.bias'])

        return x

    def set_meta(self, state):
        self.activate_meta = state

    def set_learn_net(self, state=True):
        params = OrderedDict(self.named_meta_parameters())
        for key in params:
            if key not in ['noise_output.weight', 'noise_output.bias', 'noise_input.weight', 'noise_input.bias',
                           'meta_in_1.weight', 'meta_in_1.bias', 'meta_in_2.weight', 'meta_in_2.weight',
                           'meta_out_1.weight', 'meta_out_1.bias', 'meta_out_2.bias', 'meta_out_2.weight']:
                params[key].requires_grad = state

    def set_learn_noise(self, state=True):
        params = OrderedDict(self.named_meta_parameters())
        params['noise_output.weight'].requires_grad = state
        params['noise_output.bias'].requires_grad = state
        params['noise_input.weight'].requires_grad = state
        params['noise_input.bias'].requires_grad = state

    def set_learn_meta(self, state=True):
        params = OrderedDict(self.named_meta_parameters())
        params['meta_in_1.weight'].requires_grad = state
        params['meta_in_1.bias'].requires_grad = state
        params['meta_in_2.weight'].requires_grad = state
        params['meta_in_2.bias'].requires_grad = state
        params['meta_out_1.weight'].requires_grad = state
        params['meta_out_1.bias'].requires_grad = state
        params['meta_out_2.weight'].requires_grad = state
        params['meta_out_2.bias'].requires_grad = state

    def update_parameters(self, loss, params=None, step_size=0.5, first_order=False,
                          remove_noise_param=True, keep_meta_only=True):
        if params is None:
            params = OrderedDict(self.named_meta_parameters())

        if remove_noise_param:
            if params.__contains__('noise_output.bias'):
                del params['noise_output.bias']
            if params.__contains__('noise_output.weight'):
                del params['noise_output.weight']
            if params.__contains__('noise_input.bias'):
                del params['noise_input.bias']
            if params.__contains__('noise_input.weight'):
                del params['noise_input.weight']

        prev_params = OrderedDict()
        if keep_meta_only:
            new_params = OrderedDict()
            for key in params:
                if key in ['meta_in_1.weight', 'meta_in_1.bias', 'meta_in_2.weight', 'meta_in_2.bias',
                           'meta_out_1.weight', 'meta_out_1.bias', 'meta_out_2.bias', 'meta_out_2.weight']:
                    new_params[key] = params[key]
                else:
                    prev_params[key] = params[key]
            params = new_params

        updated_meta_params = super().update_parameters(loss=loss, params=params, step_size=step_size,
                                                        first_order=first_order)

        final_params = OrderedDict()
        for key in updated_meta_params:
            final_params[key] = updated_meta_params[key]
        for key in prev_params:
            final_params[key] = prev_params[key]
        return final_params
