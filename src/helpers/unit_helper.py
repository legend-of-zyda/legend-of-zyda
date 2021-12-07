# yapf: disable
from lux.constants import Constants
from lux.game_constants import GAME_CONSTANTS

def get_num_nearby_workers(player, tilePos, dist):
    num_nearby_workers = 0
    for unit in player.units:
        if unit.pos.distance_to(tilePos) < dist and unit.pos.distance_to(tilePos) != 0:
            num_nearby_workers += 1
    return num_nearby_workers

def get_units_with_empty_cargo(units):
    empty_units = []
    for u in units:
        if (u.get_cargo_space_left() == 100):
            empty_units.append(u)

    return empty_units

def get_units_with_full_cargo(units):
    full_units = []
    for u in units:
        if (u.get_cargo_space_left() == 0):
            full_units.append(u)

    return full_units

def get_carts_count(units):
    return sum(u.type == Constants.UNIT_TYPES.CART for u in units)

def get_workers_count(units):
    return sum(u.type == Constants.UNIT_TYPES.WORKER for u in units)

def get_units_in_danger(units, turns_until_nightfall):
    # can start with simply checking if units will last throughout nightfall at current resource level

    night_duration = GAME_CONSTANTS.PARAMETERS.NIGHT_LENGTH

    u_in_danger = []
    for u in units:
        if (u.type == Constants.UNIT_TYPES.CART):
            burn_rate = GAME_CONSTANTS.PARAMETERS.LIGHT_UPKEEP.WORKER
        else:
            burn_rate = GAME_CONSTANTS.PARAMETERS.LIGHT_UPKEEP.CART

        if (u.get_cargo_space_left() < burn_rate * (turns_until_nightfall + night_duration)):
            u_in_danger.append(u)

    return u_in_danger

def get_closest_unit(unit, all_units):
    closest_unit = None
    closest_dist = 1_000_000_000
    for u in all_units:
        if u.pos == unit.pos: # Same unit.
            continue

        dist = u.pos.distance_to(unit.pos)
        if dist < closest_dist:
            closest_unit = u
            closest_dist = dist

    # None if no other unit on the board.
    return closest_unit
