"""The nn agent."""
import torch
from helpers import tile_helper
from helpers.Network import Network
from helpers.StateManager import StateManager
from lux.constants import Constants
from lux.game import Game
from lux.game_constants import GAME_CONSTANTS
from lux.game_map import Cell, Position
from torch import nn

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

DIRECTIONS = Constants.DIRECTIONS

GAME_STATE = None

WORKER_INPUTS = 16
WORKER_OUTPUTS = 6
worker_net = Network([
    (WORKER_INPUTS, 32),
    (32, WORKER_OUTPUTS),
])

TOTAL_PARAMS = len(worker_net.serialize())


def get_inputs(game_state, state_manager, unit):
    #  # Map representation from the DQN notebook
    #  # The shape of the map
    #  w, h = game_state.map.width, game_state.map.height
    #  # The map of resources
    #  M = [[
    #      0 if game_state.map.map[j][i].resource == None else
    #      game_state.map.map[j][i].resource.amount for i in range(w)
    #  ] for j in range(h)]

    #  M = np.array(M).reshape((h, w, 1))

    #  # The map of units features
    #  U = [[[0, 0, 0, 0, 0] for i in range(w)] for j in range(h)]
    #  units = game_state.players[0].units
    #  for i in units:
    #      U[i.pos.y][i.pos.x] = [
    #          i.type, i.cooldown, i.cargo.wood, i.cargo.coal, i.cargo.uranium
    #      ]

    #  U = np.array(U)

    #  # The map of cities features
    #  e = game_state.players[1].cities
    #  C = [[[0, 0, 0, 0] for i in range(w)] for j in range(h)]
    #  for k in e:
    #      citytiles = e[k].citytiles
    #      for i in citytiles:
    #          C[i.pos.y][i.pos.x] = [
    #              i.cooldown, e[k].fuel, e[k].light_upkeep, e[k].team
    #          ]

    #  C = np.array(C)

    # TODO test adding relevant opponent info to state, other resource
    # locations, and worker-resource assignments
    S = [
        float(state_manager.closest_resource_tile_dist_for_unit(unit)),
        float(state_manager.closest_empty_tile_dist_for_unit(unit)),
        float(state_manager.closest_city_tile_dist_for_unit(unit)),
        float(state_manager.get_cities_count()),
        float(len(state_manager.get_units())),
        float(unit.get_cargo_space_left()),
        float(unit.can_build(game_state.map)),
        float(unit.can_act()),
        float(unit.cooldown),
        float(state_manager.turn()),
        float(state_manager.turns_until_nightfall()),
        float(state_manager.research_points()),
        float(len(state_manager.get_units_with_full_cargo())),
        float(state_manager.get_workers_count()),
        float(game_state.players[0].researched_coal()),
        float(game_state.players[0].researched_uranium())
    ]
    # print(M.shape,U.shape,C.shape)
    # stacking all in one array
    # E = np.dstack([M,U,C])

    # try just using handpicked state for now
    return S


def neural_agent(
        observation,
        configuration,  # pylint: disable = unused-argument
        weights,  # TODO: Load weights from file by default.
):
    global GAME_STATE

    worker_net.eval()
    worker_net.deserialize(weights)

    ### Do not edit ###
    if observation["step"] == 0:
        GAME_STATE = Game()
        GAME_STATE._initialize(observation["updates"])
        GAME_STATE._update(observation["updates"][2:])
        GAME_STATE.id = observation.player
    else:
        GAME_STATE._update(observation["updates"])

    actions = []

    ### AI Code goes down here! ###
    player = GAME_STATE.players[observation.player]
    opponent = GAME_STATE.players[(observation.player + 1) % 2]
    width, height = GAME_STATE.map.width, GAME_STATE.map.height

    # initialize statemanager with latest data
    state_manager = StateManager(GAME_STATE, player, opponent)

    for unit in player.units:
        # TODO: Handle carts (we currently don't build any carts in the cities).
        state = get_inputs(GAME_STATE, state_manager, unit)
        #  logger.info("state: {}", state)
        with torch.no_grad():
            action_idx = worker_net.action(state)
        #  logger.info("action_idx: {}", action_idx)

        if (action_idx == 0):
            actions.append(unit.move(Constants.DIRECTIONS.CENTER))
        elif (action_idx == 1):
            actions.append(unit.move(Constants.DIRECTIONS.NORTH))
        elif (action_idx == 2):
            actions.append(unit.move(Constants.DIRECTIONS.SOUTH))
        elif (action_idx == 3):
            actions.append(unit.move(Constants.DIRECTIONS.WEST))
        elif (action_idx == 4):
            actions.append(unit.move(Constants.DIRECTIONS.EAST))
        elif (action_idx == 5):
            if (unit.can_build(GAME_STATE.map)):
                actions.append(unit.build_city())
            else:
                actions.append(unit.move(Constants.DIRECTIONS.CENTER))

    # Handle citytiles manually for now since the strategy is pretty
    # straightforward.
    num_city_tiles = state_manager.get_city_tiles_count()
    for city_id in player.cities.keys():
        for city_tile in player.cities[city_id].citytiles:
            if city_tile.can_act():
                if num_city_tiles > len(
                        player.units):  # Unit cap not reached yet
                    actions.append(city_tile.build_worker())
                elif not player.researched_uranium():
                    actions.append(city_tile.research())
    return actions
