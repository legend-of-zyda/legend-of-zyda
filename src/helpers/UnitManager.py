from lux.game_objects import Player, Unit
import math
import copy
from . import tile_helper, unit_helper
from helpers.mapf import MAPF_Map, MAPF_Agent, plan
from lux import annotate

MAX_NEARBY_WORKERS_FOR_DISTANT_ASSIGNMENTS = 4
RANGE_OF_NEARBY_WORKER_CHECK = 5
MINIMUM_DISTANCE_FOR_DISTANT_CLUSTERS = 10

class UnitManager:
    def __init__(self):
        self.worker_assignments = dict() # Map of unit ID --> (X,Y) of that unit's assigned tile
        self.tile_worker_assignments = dict() # Map of (X,Y) of tile --> unit ID assigned to that tile
        self.unassigned_workers = set() # Set of unit IDs
        self.workers = dict() # Map of unit ID --> Unit Object
        self.unit_movement_paths = dict() # Map of {unit_id --> [path]}

    def clean_up_data_structures(self, player):
        self.workers = dict()
        self.unassigned_workers = set()
        for unit in player.units:
            self.workers[unit.id] = unit
        self.clean_up_assignments(player)
        for unit in player.units:
            if unit.id not in self.worker_assignments:
                self.unassigned_workers.add(unit.id)

    def clean_up_assignments(self, player): # Remove assignments for dead units and TODO: assignments to blocked tiles
        still_alive_worker_ids = set(map(lambda unit : unit.id, player.units))
        dead_worker_ids = []
        for unit_id in self.worker_assignments:
            if unit_id not in still_alive_worker_ids:
                dead_worker_ids.append(unit_id)

        for dead_worker_id in dead_worker_ids:
            target_cell_tuple = self.worker_assignments[dead_worker_id]
            del self.worker_assignments[dead_worker_id]
            del self.tile_worker_assignments[target_cell_tuple]

    # TODO: Match resources to units instead of units to resources
    def update_worker_assignments(self, player, game_state, actions):
        for unit in player.units:
            if unit.id not in self.worker_assignments:
                assignment_made = self.assign_unit(unit, player, game_state)
                if not assignment_made:
                    continue
            else:
                target_coordinates = self.worker_assignments[unit.id]
                target_cell = game_state.map.get_cell(x=target_coordinates[0], y=target_coordinates[1])
                actions.append(annotate.line(unit.pos.x, unit.pos.y, target_cell.pos.x, target_cell.pos.y))

                # Now check if an assigned worker is in a city with no more adjacent resources, in which case, reassign
                if len(list(filter(lambda tile : tile.has_resource(), tile_helper.get_adjacent_tiles(target_cell, game_state)))) == 0:
                    del self.worker_assignments[unit.id]
                    del self.tile_worker_assignments[(target_cell.pos.x, target_cell.pos.y)]
                    self.assign_unit(unit, player, game_state)

        for unit_id in copy.deepcopy(self.unassigned_workers):
            self.assign_unit(self.workers[unit_id], player, game_state, False)

    def assign_unit(self, unit: Unit, player: Player, game_state, skip_unresearched_resources=True):
        resource_adjacent_tiles = tile_helper.get_empty_resource_adjacent_tiles(game_state, player, skip_unresearched_resources)
        if self.get_num_nearby_assignments(game_state, unit.pos, RANGE_OF_NEARBY_WORKER_CHECK) < MAX_NEARBY_WORKERS_FOR_DISTANT_ASSIGNMENTS:
            assigned_tile = self.get_closest_available_resource_adjacent_tile(unit, resource_adjacent_tiles)
        else: # Send it at least ten tiles away
            assigned_tile = self.get_distant_available_resource_adjacent_tile(unit, game_state, resource_adjacent_tiles, MINIMUM_DISTANCE_FOR_DISTANT_CLUSTERS)
            if assigned_tile is None: # In case there isn't a distant available resource, take any of them.
                assigned_tile = self.get_closest_available_resource_adjacent_tile(unit, resource_adjacent_tiles)

        if assigned_tile is not None:
            target_cell_tuple = (assigned_tile.pos.x, assigned_tile.pos.y)
            self.worker_assignments[unit.id] = target_cell_tuple
            self.tile_worker_assignments[target_cell_tuple] = unit.id
            self.unassigned_workers.discard(unit.id)
            return assigned_tile
        else:
            return False

    def get_closest_available_resource_adjacent_tile(self, unit: Unit, resource_adjacent_tiles):
        closest_dist = math.inf
        closest_resource_adjacent_tile = None # resource adjacent = adjacent to a resource tile but not a resource tile itself

        for resource_adjacent_tile in resource_adjacent_tiles:
            dist = resource_adjacent_tile.pos.distance_to(unit.pos)
            if dist < closest_dist and (resource_adjacent_tile.pos.x, resource_adjacent_tile.pos.y) not in self.tile_worker_assignments:
                closest_dist = dist
                closest_resource_adjacent_tile = resource_adjacent_tile
        return closest_resource_adjacent_tile

    def get_distant_available_resource_adjacent_tile(self, unit: Unit, game_state, resource_adjacent_tiles, minimum_distance):
        closest_dist = math.inf
        closest_resource_adjacent_tile = None # resource adjacent = adjacent to a resource tile but not a resource tile itself

        for resource_adjacent_tile in resource_adjacent_tiles:
            dist = resource_adjacent_tile.pos.distance_to(unit.pos)
            if (dist < closest_dist and
                    (resource_adjacent_tile.pos.x, resource_adjacent_tile.pos.y) not in self.tile_worker_assignments
                    and dist >= minimum_distance
                    and self.get_num_nearby_assignments(game_state, resource_adjacent_tile.pos, RANGE_OF_NEARBY_WORKER_CHECK) == 0):
                closest_dist = dist
                closest_resource_adjacent_tile = resource_adjacent_tile
        return closest_resource_adjacent_tile

    def get_num_nearby_assignments(self, game_state, tile_pos, _range):
        num_nearby_assignments = 0
        for unit_id in self.worker_assignments:
            assignment_tuple = self.worker_assignments[unit_id]
            assignment_cell = game_state.map.get_cell(x=assignment_tuple[0], y=assignment_tuple[1])
            if assignment_cell.pos.distance_to(tile_pos) < _range:
                num_nearby_assignments += 1
        return num_nearby_assignments

    def get_mapf_map_and_agents(self, game_state, player: Player, opponent: Player):
        units = []
        for unit in player.units:
            target_tile = (self.worker_assignments[unit.id] if
                unit.id in self.worker_assignments.keys() else (unit.pos.x, unit.pos.y))
            units.append(MAPF_Agent(
                agent_id=unit.id,
                start_tile=((unit.pos.y * game_state.map.width) + unit.pos.x), # TODO: write a function that translates 2d to 1d
                goal_tile=((target_tile[1] * game_state.map.width) + target_tile[0]),
            ))

        opponent_city_tiles = set()
        opponent_unit_tiles = set()
        for unit in opponent.units:
            opponent_city_tiles.add((unit.pos.x, unit.pos.y))
        for city_id in opponent.cities:
            for city_tile in opponent.cities[city_id].citytiles:
                opponent_city_tiles.add((city_tile.pos.x, city_tile.pos.y))

        # Allow vertex collisions on a player's city tiles
        stackable_tiles = set()
        for city_id in player.cities:
            for city_tile in player.cities[city_id].citytiles:
                stackable_tiles.add((city_tile.pos.y * game_state.map.width) + city_tile.pos.x)

        mapf_map = []
        for row_index, row in enumerate(game_state.map.map): #y
            mapf_map.append([])
            for col_index, col in enumerate(game_state.map.map[row_index]): #x
                current_tile_tuple = (col_index, row_index)
                if current_tile_tuple in opponent_city_tiles or current_tile_tuple in opponent_unit_tiles: # This tile is blocked
                    mapf_map[row_index].append("@")
                else:
                    mapf_map[row_index].append(".")
        return (MAPF_Map(mapf_map), units, stackable_tiles)

    def update_unit_movement_paths(self, game_state, player, opponent):
        mapf_map, mapf_agents, stackable_tiles = self.get_mapf_map_and_agents(game_state, player, opponent)
        self.unit_movement_paths = plan(mapf_map, mapf_agents, "MAPF_Prioritized_Planning", stackable_tiles)

    def get_next_tile(self, game_state, unit: Unit):
        unit_mapf_path = self.unit_movement_paths[unit.id]
        if unit_mapf_path is not None: # Check that a path was found
            unit_next_mapf_tile = unit_mapf_path[1]
            tile_row = math.floor(unit_next_mapf_tile / game_state.map_width)
            tile_col = unit_next_mapf_tile % game_state.map_width
            tile_pos = game_state.map.get_cell(x=tile_col, y=tile_row).pos
            return tile_pos
