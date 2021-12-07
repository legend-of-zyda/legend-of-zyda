"""StateManager provides info on the game state."""
# yapf: disable
# ^^ Added by Bryon so I don't mess with the formatting.
from . import tile_helper, unit_helper


class StateManager:
    def __init__(self, game_state, player, opponent):
        self.game_state = game_state
        self.player = player
        self.opponent = opponent

        self.max_x_dist = game_state.map_width - 1
        self.max_y_dist = game_state.map_height - 1
        self.max_dist = self.max_x_dist + self.max_y_dist

    ########
    # General
    ########
    def turn(self):
        return self.game_state.turn

    def turns_left(self):
        return 360 - self.game_state.turn

    def turns_until_nightfall(self):
        mod = self.turn() % 40
        diff = 30 - mod

        if diff >= 0:
            return diff + 1
        else:
            return 0

    def research_points(self):
        return self.player.research_points

    ########
    # Opponent
    ########

    def opponent_research_points(self):
        return self.opponent.research_points

    def get_opponents_with_empty_cargo(self):
        return unit_helper.get_units_with_empty_cargo(self.opponent.units)

    def get_opponents_with_full_cargo(self):
        return unit_helper.get_units_with_full_cargo(self.opponent.units)

    def get_opponent_carts_count(self):
        return unit_helper.get_carts_count(self.opponent.units)

    def get_opponent_workers_count(self):
        return unit_helper.get_workers_count(self.opponent.units)

    def get_opponent_city_with_lowest_fuel(self):
        return tile_helper.get_city_with_lowest_fuel(self.opponent.cities)

    def get_opponents_in_danger(self):
        return unit_helper.get_units_in_danger(self.opponent.units, self.turns_until_nightfall())

    ########
    # Units
    ########
    def _dist(self, tile, unit, norm):
        """Calculates Manhattan dist between tile and unit.

        Args:
            norm: Whether to normalize by max dist.
        """

        # TODO - better solution when tile doesn't exist?
        if tile is not None:
            dist = tile.pos.distance_to(unit.pos)
        else:
            dist = 999

        return dist / self.max_dist if norm else dist

    def _coord_dist(self, tile, unit, sign, norm):
        """Calculates x and y dist between tile and unit.

        Args:
            sign: Whether to give back signed distances (instead of taking abs
                value).
            norm: Whether to normalize by max dist.
        """
        dist = (tile.pos.x - unit.pos.x, tile.pos.y - unit.pos.y)
        if norm:
            dist = (dist[0] / self.max_x_dist,
                    dist[1] / self.max_y_dist)
        return dist if sign else (abs(dist[0]), abs(dist[1]))

    def get_units(self):
        return self.player.units

    def get_units_with_empty_cargo(self):
        return unit_helper.get_units_with_empty_cargo(self.player.units)

    def get_units_with_full_cargo(self):
        return unit_helper.get_units_with_full_cargo(self.player.units)

    def get_carts_count(self):
        return unit_helper.get_carts_count(self.player.units)

    def get_workers_count(self):
        return unit_helper.get_workers_count(self.player.units)

    def get_units_in_danger(self):
        return unit_helper.get_units_in_danger(self.player.units, self.turns_until_nightfall())

    def closest_city_tile_for_unit(self, unit):
        return tile_helper.find_closest_city_tile(unit.pos, self.player)

    def closest_city_tile_dist_for_unit(self, unit, norm=False):
        tile = self.closest_city_tile_for_unit(unit)
        return self._dist(tile, unit, norm)

    def closest_resource_tile_for_unit(self, unit):
        resource_tiles = tile_helper.find_resource_tiles(self.game_state, self.player, True)
        tile = tile_helper.find_closest_tile(unit.pos, resource_tiles)

        return tile

    def closest_resource_tile_dist_for_unit(self, unit, norm=False):
        tile = self.closest_resource_tile_for_unit(unit)

        return self._dist(tile, unit, norm)

    def closest_resource_tile_coord_dist_for_unit(self, unit, sign=False,
                                                  norm=False):
        tile = self.closest_resource_tile_for_unit(unit)

        return self._coord_dist(tile, unit, sign, norm)

    def closest_empty_tile_for_unit(self, unit):
        empty_tiles = tile_helper.find_empty_tiles(self.game_state)
        tile = tile_helper.find_closest_tile(unit.pos, empty_tiles)

        return tile

    def closest_empty_tile_dist_for_unit(self, unit, norm=False):
        tile = self.closest_empty_tile_for_unit(unit)

        return self._dist(tile, unit, norm)

    def closest_empty_tile_coord_dist_for_unit(self, unit, sign=False,
                                               norm=False):
        tile = self.closest_empty_tile_for_unit(unit)

        return self._coord_dist(tile, unit, sign, norm)

    def closest_unit_to_unit(self, unit):
        return unit_helper.get_closest_unit(unit, self.player.units)

    def closest_unit_dist_to_unit(self, unit, norm=False):
        closest_unit = unit_helper.get_closest_unit(unit, self.player.units)

        # The default value of 0 is debatable.
        return (0 if closest_unit is None else
                self._dist(closest_unit, unit, norm))

    def closest_unit_coord_dist_to_unit(self, unit, sign=False, norm=False):
        closest_unit = unit_helper.get_closest_unit(unit, self.player.units)

        # The default value of (0, 0) is debatable.
        return ((0, 0) if closest_unit is None else
                self._coord_dist(closest_unit, unit, sign, norm))

    ########
    # Cities
    ########
    def get_cities(self):
        return self.player.cities

    def get_cities_count(self):
        return len(self.player.cities)

    def get_city_tiles_count(self):
        count = 0

        for k, city in self.player.cities.items():
            count = count + len(city.citytiles)

        return count
    def get_city_with_lowest_fuel(self):
        return tile_helper.get_city_with_lowest_fuel(self.player.cities)
