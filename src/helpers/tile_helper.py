import math
from lux.game_map import Cell
from lux.constants import Constants

def find_empty_tiles(game_state):
    empty_resource_tiles: list[Cell] = []
    width, height = game_state.map_width, game_state.map_height
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            if not cell.has_resource() and not cell.citytile:
                empty_resource_tiles.append(cell)
    return empty_resource_tiles

def find_resource_tiles(game_state, player, skip_unresearched_resources=True):
    resource_tiles: list[Cell] = []
    width, height = game_state.map_width, game_state.map_height
    for y in range(height):
        for x in range(width):
            cell = game_state.map.get_cell(x, y)
            
            if cell.has_resource() and not cell.citytile:
                if skip_unresearched_resources: # Skip this tile if player hasn't researched this type yet
                    if cell.resource.type == Constants.RESOURCE_TYPES.COAL and not player.researched_coal(): continue
                    if cell.resource.type == Constants.RESOURCE_TYPES.URANIUM and not player.researched_uranium(): continue

                resource_tiles.append(cell)
    return resource_tiles

def find_closest_tile(pos, tiles):
    closest_dist = math.inf
    closest_tile = None
    for tile in tiles:
        dist = tile.pos.distance_to(pos)
        if dist < closest_dist:
            closest_dist = dist
            closest_tile = tile
    return closest_tile

def find_closest_city_tile(pos, player):
    closest_city_tile = None
    if len(player.cities) > 0:
        closest_dist = math.inf
        # the cities are stored as a dictionary mapping city id to the city object, which has a citytiles field that
        # contains the information of all citytiles in that city
        for city in player.cities.values():
            for city_tile in city.citytiles:
                dist = city_tile.pos.distance_to(pos)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_city_tile = city_tile
    return closest_city_tile

def get_city_with_lowest_fuel(cities):
    lowest_fuel = math.inf
    c = None

    for city in cities.values():
        lowest_fuel = city.fuel if city.fuel < lowest_fuel else lowest_fuel
        c = city

    return c

def get_adjacent_tiles(cell: Cell, game_state):
    adjacent_tiles = []
    north = (cell.pos.x, cell.pos.y - 1)
    south = (cell.pos.x, cell.pos.y + 1)
    east = (cell.pos.x + 1, cell.pos.y)
    west = (cell.pos.x - 1, cell.pos.y)
    for adjacent_tile in [north, south, east, west]:
        if 0 <= adjacent_tile[0] < game_state.map.width and 0 <= adjacent_tile[1] < game_state.map.height:
            adjacent_tiles.append(game_state.map.get_cell(x=adjacent_tile[0], y=adjacent_tile[1]))
    return adjacent_tiles

def get_adjacent_resource_tiles(cell: Cell, game_state):
    adjacent_tiles = get_adjacent_tiles(cell, game_state)
    return list(filter(lambda tile : tile.has_resource(), adjacent_tiles))

def get_empty_resource_adjacent_tiles(game_state, player, skip_unresearched_resources=True):
    resource_adjacent_tiles = set()
    resource_tiles = find_resource_tiles(game_state, player, skip_unresearched_resources)
    for resource_tile in resource_tiles:
        adjacent_tiles = get_adjacent_tiles(resource_tile, game_state)
        # Get all tiles adjacent to the resource tile which are not resource tiles
        resource_adjacent_tiles |= set(filter(lambda tile : (not tile.has_resource() and not tile.citytile), adjacent_tiles))
    return resource_adjacent_tiles

def get_cluster(game_state, resource_tile, new_cluster, tiles_in_clusters):
    adjacent_resource_tiles = get_adjacent_resource_tiles(resource_tile, game_state)
    for adjacent_resource_tile in adjacent_resource_tiles:
        if adjacent_resource_tile not in tiles_in_clusters:
            new_cluster.append(adjacent_resource_tile)
            tiles_in_clusters.add(adjacent_resource_tile)
            get_cluster(game_state, adjacent_resource_tile, new_cluster, tiles_in_clusters)

# Returns a list of lists [[tile1, tile2], [tile3], [tile4, tile5]]
def get_resource_clusters(game_state, player):
    clusters = []
    tiles_in_clusters = set()
    resource_tiles = find_resource_tiles(game_state, player, True)
    for resource_tile in resource_tiles:
        if resource_tile in tiles_in_clusters:
            continue
        new_cluster = []
        get_cluster(game_state, resource_tile, new_cluster, tiles_in_clusters)
        clusters.append(new_cluster)
    return clusters
