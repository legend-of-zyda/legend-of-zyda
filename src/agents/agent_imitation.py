"""Imitation Learning agent."""
import os

import numpy as np
import torch
import torch.nn.functional as F
from lux.game import Game
from torch import nn

try:
    from loguru import logger
except ImportError:
    logger = None

DEVICE = "cpu"

#
# Agent models.
#


# Neural Network for Lux AI
class BasicConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.conv = nn.Conv2d(input_dim,
                              output_dim,
                              kernel_size=kernel_size,
                              padding=(kernel_size[0] // 2,
                                       kernel_size[1] // 2))
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h) if self.bn is not None else h
        return h


class SerializationMixin:
    def serialize(self):
        """Returns 1D array with all parameters in the actor."""
        return np.concatenate(
            [p.data.cpu().detach().numpy().ravel() for p in self.parameters()])

    def deserialize(self, array):
        """Loads parameters from 1D array.

        Note that state params like averages for batch norm layers are not
        loaded here.
        """
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


class LuxNet(nn.Module, SerializationMixin):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32
        self.conv0 = BasicConv2d(20, filters, (3, 3), True)
        self.blocks = nn.ModuleList([
            BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)
        ])
        self.head_p = nn.Linear(filters, 5, bias=False)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:, :1]).view(h.size(0), h.size(1), -1).sum(-1)
        p = self.head_p(h_head)
        return p

    def serialize(self):
        """Returns 1D array with all parameters in the actor."""
        return np.concatenate(
            [p.data.cpu().detach().numpy().ravel() for p in self.parameters()])

    def deserialize(self, array):
        """Loads parameters from 1D array.

        Note that state params like averages for batch norm layers are not
        loaded here.
        """
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


class LuxController(nn.Module, SerializationMixin):
    """Outputs actions after being passed an encoded state."""
    def __init__(self):
        super().__init__()
        filters = 32
        self.head_p = nn.Linear(filters, 5, bias=False)

    def forward(self, h_head):
        return self.head_p(h_head)


class LuxSeparate:
    """A frozen encoder followed by a trainable controller."""
    def __init__(self):
        # Always use the pretrained encoder.
        here = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(here, "state_encoder.pth")
        self.encoder = torch.jit.load(filename, DEVICE)
        self.encoder.eval()

        self.head = LuxController()

    def to(self, device):
        self.encoder.to(device)
        self.head.to(device)
        return self

    def eval(self):
        self.encoder.eval()
        self.head.eval()

    def __call__(self, state):
        with torch.no_grad():
            encoded_state = self.encoder(state)
        return self.head(encoded_state)

    def serialize(self):
        return self.head.serialize()

    def deserialize(self, params):
        self.head.deserialize(params)
        return self

    def state_dict(self):
        return self.head.head_p.state_dict()

    def load_state_dict(self, state_dict):
        # We didn't save things properly in freeze_head.py, so we have to load
        # for head.head_p instead of just head.
        self.head.head_p.load_state_dict(state_dict)


#
# Agent
#


def call_func(obj, method, args=()):
    return getattr(obj, method)(*args)


class ImitationAgent:
    """Callable agent which maintains several state variables.

    This cleans up the global variable problem -- the class object still behaves
    like a function because of __call__().
    """

    UNIT_ACTIONS = [('move', 'n'), ('move', 's'), ('move', 'w'), ('move', 'e'),
                    ('build_city', )]

    def __init__(self, model):
        self.game_state = None

        # Controls worker actions.
        self.model = model.to(DEVICE)

    @staticmethod
    def make_input(obs, unit_id):
        width, height = obs['width'], obs['height']
        x_shift = (32 - width) // 2
        y_shift = (32 - height) // 2
        cities = {}

        b = np.zeros((20, 32, 32), dtype=np.float32)

        for update in obs['updates']:
            strs = update.split(' ')
            input_identifier = strs[0]

            if input_identifier == 'u':
                x = int(strs[4]) + x_shift
                y = int(strs[5]) + y_shift
                wood = int(strs[7])
                coal = int(strs[8])
                uranium = int(strs[9])
                if unit_id == strs[3]:
                    # Position and Cargo
                    b[:2, x, y] = (1, (wood + coal + uranium) / 100)
                else:
                    # Units
                    team = int(strs[2])
                    cooldown = float(strs[6])
                    idx = 2 + (team - obs['player']) % 2 * 3
                    b[idx:idx + 3, x,
                      y] = (1, cooldown / 6, (wood + coal + uranium) / 100)
            elif input_identifier == 'ct':
                # CityTiles
                team = int(strs[1])
                city_id = strs[2]
                x = int(strs[3]) + x_shift
                y = int(strs[4]) + y_shift
                idx = 8 + (team - obs['player']) % 2 * 2
                b[idx:idx + 2, x, y] = (1, cities[city_id])
            elif input_identifier == 'r':
                # Resources
                r_type = strs[1]
                x = int(strs[2]) + x_shift
                y = int(strs[3]) + y_shift
                amt = int(float(strs[4]))
                b[{
                    'wood': 12,
                    'coal': 13,
                    'uranium': 14
                }[r_type], x, y] = amt / 800
            elif input_identifier == 'rp':
                # Research Points
                team = int(strs[1])
                rp = int(strs[2])
                b[15 + (team - obs['player']) % 2, :] = min(rp, 200) / 200
            elif input_identifier == 'c':
                # Cities
                city_id = strs[2]
                fuel = float(strs[3])
                lightupkeep = float(strs[4])
                cities[city_id] = min(fuel / lightupkeep, 10) / 10

        # Day/Night Cycle
        b[17, :] = obs['step'] % 40 / 40
        # Turns
        b[18, :] = obs['step'] / 360
        # Map Size
        b[19, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

        return b

    def get_game_state(self, observation):
        if observation["step"] == 0:
            self.game_state = Game()
            self.game_state._initialize(observation["updates"])
            self.game_state._update(observation["updates"][2:])
            self.game_state.id = observation["player"]
        else:
            self.game_state._update(observation["updates"])
        return self.game_state

    def in_city(self, pos):
        try:
            city = self.game_state.map.get_cell_by_pos(pos).citytile
            return city is not None and city.team == self.game_state.id
        except:  # pylint: disable = bare-except
            return False

    def get_action(self, policy, unit, dest):
        for label in np.argsort(policy)[::-1]:
            act = self.UNIT_ACTIONS[label]
            pos = unit.pos.translate(act[-1], 1) or unit.pos
            if pos not in dest or self.in_city(pos):
                return call_func(unit, *act), pos

        return unit.move('c'), unit.pos

    def __call__(self, observation, configuration):
        self.game_state = self.get_game_state(observation)
        player = self.game_state.players[observation.player]
        actions = []

        # City Actions
        unit_count = len(player.units)
        for city in player.cities.values():
            for city_tile in city.citytiles:
                if city_tile.can_act():
                    if unit_count < player.city_tile_count:
                        actions.append(city_tile.build_worker())
                        unit_count += 1
                    elif not player.researched_uranium():
                        actions.append(city_tile.research())
                        player.research_points += 1

        # Worker Actions
        dest = []
        for unit in player.units:
            if unit.can_act() and (self.game_state.turn % 40 < 30
                                   or not self.in_city(unit.pos)):
                state = self.make_input(observation, unit.id)
                with torch.no_grad():
                    p = self.model(
                        torch.from_numpy(state).unsqueeze(0).to(
                            DEVICE)).detach().cpu()

                policy = p.squeeze(0).numpy()

                action, pos = self.get_action(policy, unit, dest)
                actions.append(action)
                dest.append(pos)

        return actions


def get_default_model(head_only):
    """Retrieves default imitation learning model."""
    here = os.path.dirname(os.path.abspath(__file__))
    if head_only:
        filename = os.path.join(here, "controller_weights.pth")
        head_state_dict = torch.load(filename, DEVICE)
        model = LuxSeparate()
        model.load_state_dict(head_state_dict)
        return model
    else:
        filename = os.path.join(here, "model_imitation_v2.pth")
        model = torch.jit.load(filename, DEVICE)
        model.eval()
        return model


CACHED_DEFAULT_MODEL = get_default_model(head_only=False)
CACHED_DEFAULT_HEAD_ONLY_MODEL = get_default_model(head_only=True)

# Default imitation learning agent instances.
default_imitation_agent_instance = ImitationAgent(CACHED_DEFAULT_MODEL)
default_imitation_agent_instance_head_only = ImitationAgent(
    CACHED_DEFAULT_HEAD_ONLY_MODEL)


def get_default_model_weights(head_only):
    if head_only:
        model = CACHED_DEFAULT_HEAD_ONLY_MODEL
        return model.serialize()
    else:
        model = CACHED_DEFAULT_MODEL
        return np.concatenate([
            p.data.cpu().detach().numpy().ravel() for p in model.parameters()
        ])


def imitation_agent_from_weights(params, head_only):
    new_model = LuxSeparate() if head_only else LuxNet()

    cached = (CACHED_DEFAULT_HEAD_ONLY_MODEL
              if head_only else CACHED_DEFAULT_MODEL)

    # Warning: It is very important that the state dict is loaded here since the
    # model contains state params like batch norm averages that are not captured
    # by the parameters(). Thus, we first load the state and then change the
    # params with deserialize.
    with torch.no_grad():
        new_model.load_state_dict(cached.state_dict())
        new_model.deserialize(params)

    new_model.eval()

    if logger is not None:
        logger.info("Model: {} ||Params||: {}", new_model,
                    np.linalg.norm(params))

    return ImitationAgent(new_model)


def imitation_agent_from_weights_file(file, head_only):
    return imitation_agent_from_weights(np.load(file), head_only)


def get_improved_imitation_agent(head_only):
    """Retrieves default imitation learning model."""
    here = os.path.dirname(os.path.abspath(__file__))

    if head_only:
        filename = os.path.join(here, "cma_es_imitation_head_only.npy")
    else:
        filename = os.path.join(here, "cma_es_imitation.npy")

    return imitation_agent_from_weights_file(filename, head_only)


# Kaggle requires that the agent be a function, so we make it a lambda here.

# pylint: disable = unnecessary-lambda, line-too-long

# Use this for the regular imitation agent.
#  agent_imitation = (lambda observation, configuration:
#                     default_imitation_agent_instance(observation, configuration))

# Use this for the regular imitation agent with head only.
#  agent_imitation = (
#      lambda observation, configuration:
#      default_imitation_agent_instance_head_only(observation, configuration))

# Use this for the version improved by imitation learning.
#  improved_imitation_agent_instance = get_improved_imitation_agent(
#      head_only=False)
#  agent_imitation = (
#      lambda observation, configuration: improved_imitation_agent_instance(
#          observation, configuration))

# Use this for the version improved by imitation learning with only the head
# improved.
improved_imitation_agent_instance_head_only = get_improved_imitation_agent(
    head_only=True)
agent_imitation = (
    lambda observation, configuration:
    improved_imitation_agent_instance_head_only(observation, configuration))
