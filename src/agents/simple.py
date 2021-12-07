"""Super simple agent based on starter code."""
import math
import time
from typing import List

from lux import annotate
from lux.constants import Constants
from lux.game import Game
from lux.game_constants import GAME_CONSTANTS
from lux.game_map import RESOURCE_TYPES, Cell

from .utils import debug

DIRECTIONS = Constants.DIRECTIONS
MAX_RESEARCH = 200  # Needed for uranium; no point in researching after this.
GAME_STATE = None


def simple_agent(observation, configuration):
    global GAME_STATE  # pylint: disable = global-statement
    start_time = time.time()

    ### Do not edit ###
    if observation["step"] == 0:
        GAME_STATE = Game()
        GAME_STATE._initialize(observation["updates"])
        GAME_STATE._update(observation["updates"][2:])
        GAME_STATE.id = observation.player
    else:
        GAME_STATE._update(observation["updates"])

    actions = []

    debug("===== Turn", observation["step"] + 1, "=====")

    ### AI Code goes down here! ###
    player = GAME_STATE.players[observation.player]
    opponent = GAME_STATE.players[(observation.player + 1) % 2]
    width, height = GAME_STATE.map.width, GAME_STATE.map.height

    # Time.
    cycle_turn = observation["step"] % 40
    is_day_turn = cycle_turn < 30  # First 30 turns are day, last 10 are night.

    n_units = len(player.units)
    new_units = 0
    n_citytiles = sum(len(city.citytiles) for city in player.cities.values())

    # Resource info.
    resource_tiles: List[Cell] = []
    for y in range(height):
        for x in range(width):
            cell = GAME_STATE.map.get_cell(x, y)
            if cell.has_resource():
                resource_tiles.append(cell)

    # Do something with cities.
    for _, city in player.cities.items():
        for tile in city.citytiles:
            if tile.can_act():
                # Prioritize building units, then researching.
                if n_units + new_units < n_citytiles:
                    # TODO: Decide between workers and carts.
                    actions.append(tile.build_worker())
                    new_workers += 1
                elif player.research_points < MAX_RESEARCH:
                    actions.append(tile.research())
                else:
                    pass  # Do nothing -- save this city's cooldown.

    # We iterate over all our units and do something with them.
    for unit in player.units:
        if unit.is_worker() and unit.can_act():
            if unit.get_cargo_space_left() > 0:
                # If the unit is a worker and we have space in cargo, let's find
                # the nearest resource tile and try to mine it.
                closest_dist = math.inf
                closest_resource_tile = None
                for resource_tile in resource_tiles:
                    if (resource_tile.resource.type
                            == Constants.RESOURCE_TYPES.COAL and
                            not player.researched_coal()):
                        continue
                    if (resource_tile.resource.type
                            == Constants.RESOURCE_TYPES.URANIUM and
                            not player.researched_uranium()):
                        continue
                    dist = resource_tile.pos.distance_to(unit.pos)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_resource_tile = resource_tile
                if closest_resource_tile is not None:
                    actions.append(
                        unit.move(
                            unit.pos.direction_to(closest_resource_tile.pos)))
            else:
                # If unit is a worker and there is no cargo space left, and we
                # have cities, let's return to them.
                if len(player.cities) > 0:
                    closest_dist = math.inf
                    closest_city_tile = None
                    for k, city in player.cities.items():
                        for city_tile in city.citytiles:
                            dist = city_tile.pos.distance_to(unit.pos)
                            if dist < closest_dist:
                                closest_dist = dist
                                closest_city_tile = city_tile
                    if closest_city_tile is not None:
                        move_dir = unit.pos.direction_to(closest_city_tile.pos)
                        actions.append(unit.move(move_dir))

    # You can add debug annotations using the functions in the annotate object
    # actions.append(annotate.circle(0, 0))

    debug(actions)
    debug("Walltime:", time.time() - start_time)
    return actions
