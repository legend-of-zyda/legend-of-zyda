"""Network for game units."""
import numpy as np
import torch
import torch.nn as nn


class Network(nn.Module):

    def __init__(self, layer_shapes):
        super().__init__()

        layers = []
        for i, shape in enumerate(layer_shapes):
            layers.append(nn.Linear(*shape))
            if i == len(layer_shapes) - 1:
                layers.append(nn.LogSoftmax(dim=1))
            else:
                layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def serialize(self):
        """Returns 1D array with all parameters in the actor."""
        return np.concatenate(
            [p.data.cpu().detach().numpy().ravel() for p in self.parameters()])

    def deserialize(self, array):
        """Loads parameters from 1D array."""
        array = np.copy(array)
        arr_idx = 0
        for param in self.parameters():
            shape = tuple(param.data.shape)
            length = np.product(shape)
            block = array[arr_idx:arr_idx + length]
            if len(block) != length:
                raise ValueError("Array not long enough!")
            block = np.reshape(block, shape)
            arr_idx += length
            param.data = torch.from_numpy(block).float()

        return self

    def forward(self, batch_of_states):
        return self.model(batch_of_states)

    def action(self, state):
        """Takes in single state and outputs single action index."""
        state = torch.FloatTensor(state)[None]
        return torch.argmax(self(state)[0]).cpu().detach().item()
