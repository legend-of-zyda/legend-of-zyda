"""The CARB agent."""
# yapf: disable
# ^^ Added by Bryon so I don't mess with the formatting.
import math

from lux import annotate
from lux.constants import Constants
from lux.game import Game
from lux.game_map import Cell
from helpers.StateManager import StateManager
from helpers.UnitManager import UnitManager
from helpers import tile_helper

DIRECTIONS = Constants.DIRECTIONS
game_state = None
unit_manager = UnitManager()

def CARB_agent(observation, configuration):
    global game_state
    global unit_manager

    ### Do not edit ###
    if observation["step"] == 0:
        game_state = Game()
        game_state._initialize(observation["updates"])
        game_state._update(observation["updates"][2:])
        game_state.id = observation.player
    else:
        game_state._update(observation["updates"])

    actions = []

    ### AI Code goes down here! ###
    player = game_state.players[observation.player]
    opponent = game_state.players[(observation.player + 1) % 2]
    width, height = game_state.map.width, game_state.map.height

    # initialize statemanager with latest data
    state_manager = StateManager(game_state, player, opponent)
    unit_manager.clean_up_data_structures(player)

    unit_manager.update_worker_assignments(player, game_state, actions)

    unit_manager.update_unit_movement_paths(game_state, player, opponent)

    for unit in player.units:
        if unit.id in unit_manager.unassigned_workers:
            actions.append(annotate.circle(unit.pos.x, unit.pos.y))

    for unit in player.units:
        target_cell = None
        if unit.id in unit_manager.worker_assignments:
            target_coordinates = unit_manager.worker_assignments[unit.id]
            target_cell = game_state.map.get_cell(x=target_coordinates[0], y=target_coordinates[1])
        if unit.is_worker() and unit.can_act():
            if target_cell is not None and unit.pos != target_cell.pos:
                unit_next_tile = unit_manager.get_next_tile(game_state, unit)
                if unit_next_tile is not None:
                    actions.append(unit.move(unit.pos.direction_to(unit_next_tile)))
            else:
                if unit.get_cargo_space_left() == 0:
                    if not game_state.map.get_cell_by_pos(unit.pos).citytile and state_manager.turns_until_nightfall() != 0: # Don't build cities at night
                        actions.append(unit.build_city())

    num_city_tiles = state_manager.get_city_tiles_count()
    num_units = state_manager.get_workers_count()
    num_newly_built_units = 0
    num_new_research_points = 0
    for city_id in player.cities:
        for city_tile in player.cities[city_id].citytiles:
            if city_tile.can_act():
                if num_city_tiles > num_units + num_newly_built_units:
                    actions.append(city_tile.build_worker())
                    num_newly_built_units += 1
                elif player.research_points + num_new_research_points < 200:
                    actions.append(city_tile.research())
                    num_new_research_points += 1

    return actions
