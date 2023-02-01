import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

# Actor NN class
class Actor(nn.Module):

    def __init__(self, state_n, action_n, layer_nodes, seed):
        """
        Initialise Q-Network layers

        :param state_n (Int): Dimensions of state space
        :param action_n (Int): Dimensions of the action space
        :param layer_nodes (List[Int]): List of number of nodes in each hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Initiate in layers
        layers = [nn.Linear(state_n, layer_nodes[0])]

        for i in range(1, len(layer_nodes)):
            layers.append(nn.Linear(layer_nodes[i-1], layer_nodes[i]))

        layers.append(nn.Linear(layer_nodes[-1], action_n))

        self.layers = nn.ModuleList(layers)

    def reset_params(self):
        for layer in self.layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))

        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Mapping input state to actions

        :param state: Current state
        :return: Action values
        """

        # Pass output of each layer as input to the next
        x = state

        for layer in self.layers[:-1]:
            x = f.relu(layer(x))

        return torch.tanh(self.layers[-1](x))


# Critic NN class
class Critic(nn.Module):

    def __init__(self, state_n, action_n, layer_nodes, seed):
        """
        Initialise Q-Network layers

        :param state_n (Int): Dimensions of state space
        :param action_n (Int): Dimensions of the action space
        :param layer_nodes (List[Int]): List of number of nodes in each hidden layer
        """
        if len(layer_nodes) < 3:
            raise Exception("Critic NN must have at least 3 layers")

        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Initiate in layers
        layers = [
            nn.Linear(state_n, layer_nodes[0]),
            nn.Linear(layer_nodes[0]+action_n, layer_nodes[1])
        ]

        for i in range(2, len(layer_nodes)):
            layers.append(nn.Linear(layer_nodes[i-1], layer_nodes[i]))

        layers.append(nn.Linear(layer_nodes[-1], 1))

        self.layers = nn.ModuleList(layers)

    def reset_params(self):
        for layer in self.layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))

        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Mapping input state to actions

        :param state: Current state
        :param action: Current action
        :return: Action values
        """

        # Pass output of each layer as input to the next
        state_x = f.leaky_relu(self.layers[0](state))
        x = torch.cat((state_x, action), dim=1)

        for layer in self.layers[1:-1]:
            x = f.leaky_relu(layer(x))

        return self.layers[-1](x)
