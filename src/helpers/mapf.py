from queue import PriorityQueue
import math
from typing import List, Any, Set
from dataclasses import dataclass, field

MAX_TIMESTEPS = 50 # TODO: 360 # 10 is not enough to find the goal... 50 times out really fast...

# data input is a 2-d array where '@' represents an obstacle and '.' represents open space
# Assumes input is correctly formatted
class MAPF_Map:
    def __init__(self, two_dimensional_representation):
        self.data = self.flatten_data(two_dimensional_representation)
        self.num_cols = len(two_dimensional_representation[0])
        self.num_rows = len(two_dimensional_representation)

    def flatten_data(self, two_dimensional_representation):
        one_dimensional_representation = []
        for rowIndex, row in enumerate(two_dimensional_representation):
            for colIndex, col in enumerate(two_dimensional_representation[rowIndex]):
                one_dimensional_representation.append(two_dimensional_representation[rowIndex][colIndex])
        return one_dimensional_representation

class MAPF_Agent:
    def __init__(self, agent_id, start_tile, goal_tile):
        self.agent_id = agent_id
        self.start_tile = start_tile
        self.goal_tile = goal_tile
    
    def __str__(self) -> str:
        return "MAPF_Agent: agent_id={},start_tile={},end_tile={}".format(self.agent_id, self.start_tile, self.goal_tile)

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

class AStarNode:
    def __init__(self, _tile, _g, _h, _t, _parent):
        self.tile = _tile
        self.g = _g
        self.h = _h
        self.t = _t
        self.parent = _parent

    def __str__(self) -> str:
        return "AStarNode: tile={},g={},h={},t={}".format(self.tile, self.g, self.h, self.t)

class Constraint:
    def __init__(self, agent_id, tile, _t): # TODO: Add edge constraints? Not necessary for Lux-AI spec
        self.agent_id = agent_id
        self.tile = tile
        self.t = _t

def a_star(_map, agent_id, start_tile, goal_tile, constraints: List[Constraint], timesteps):
    start_node = AStarNode(
        _tile=start_tile,
        _g=0, 
        _h=manhattan_distance_between_tiles(_map, start_tile, goal_tile),
        _t=0,
        _parent=None
    )
    open_set = PriorityQueue()
    open_set.put(PrioritizedItem(start_node.h, start_node))

    all_nodes = dict()

    path = None
    while not open_set.empty():
        current: AStarNode = open_set.get().item

        if current.tile == goal_tile and current.t > timesteps: # Reached the goal state (and max # of timesteps)
            path = get_a_star_path(current)
            break
        if current.t > MAX_TIMESTEPS:
            break
        
        for next_tile in get_adjacent_tiles(_map, current.tile):
            if (next_tile, current.t + 1) not in all_nodes: # This tile hasn't been visited at the next timestamp
                next_node: AStarNode = AStarNode(
                    _tile=next_tile,
                    _g=current.g + 1,
                    _h=manhattan_distance_between_tiles(_map, next_tile, goal_tile),
                    _t=current.t + 1,
                    _parent=current
                )

                prune = False
                for constraint in constraints:
                    if constraint.agent_id == agent_id and constraint.tile == next_node.tile and constraint.t == next_node.t:
                        prune = True
                    if constraint.agent_id == agent_id and constraint.tile == next_node.tile and constraint.t < 0 and constraint.t * -1 <= next_node.t:
                        prune = True # Future Vertex Prunes
                if not prune:
                    open_set.put(PrioritizedItem(next_node.h, next_node))
                    all_nodes[(next_tile, current.t + 1)] = next_node
    return path

# Returns dict of {agent_id: [path]}, where [path] is None if no path is found for that agent
def prioritized_planning(_map: MAPF_Map, agents: List[MAPF_Agent], stackable_tiles: Set):
    plan = dict()
    constraints = []
    for agent in agents:
        path = a_star(
            _map=_map,
            agent_id=agent.agent_id,
            start_tile=agent.start_tile,
            goal_tile=agent.goal_tile,
            constraints=constraints,
            # timesteps=MAX_TIMESTEPS
            timesteps=-1
        )
        plan[agent.agent_id] = path

        # Generate constraints from the newly created path
        for other_agent in agents:
            if other_agent.agent_id == agent.agent_id or path is None:
                continue
            else:
                # Add vertex constraints
                for t, tile in enumerate(path):
                    if tile not in stackable_tiles:
                        constraints.append(Constraint(agent_id=other_agent.agent_id, tile=tile, _t=t))
                        if t == len(path) - 1:
                            constraints.append(Constraint(agent_id=other_agent.agent_id, tile=tile, _t=(t * -1))) # Future Vertex Constraint
                 # TODO: Add edge constraints? Not necessary for Lux-AI spec
    return plan

# Returns dict of {agent_id: [path]}
def plan(_map: MAPF_Map, agents: List[MAPF_Agent], algorithm: str, stackable_tiles: Set):
    plan = dict()
    if algorithm == "MAPF_Time_Space_A_Star":
        for agent in agents:
            plan[agent.agent_id] = a_star(
                _map=_map,
                agent_id=agent.agent_id,
                start_tile=agent.start_tile,
                goal_tile=agent.goal_tile,
                constraints=[],
                timesteps=-1
            )
    elif algorithm == "MAPF_Prioritized_Planning":
        plan = prioritized_planning(
            _map=_map,
            agents=agents,
            stackable_tiles=stackable_tiles
        )
    return plan

############## Helper functions ##############

# Returns true if the tile is blocked, false otherwise
def is_blocked(_map: MAPF_Map, tile): 
    return _map.data[tile] == '@'

# Returns the row number of the tile
def get_row(_map: MAPF_Map, tile):
    return math.floor(tile / _map.num_cols)

# Returns the col number of the tile
def get_col(_map: MAPF_Map, tile):
    return tile % _map.num_cols

# Returns the tiles that are reachable from a given tile (including the current tile)
def get_adjacent_tiles(_map: MAPF_Map, tile):
    adjacent_tiles = []
    # Check the tile to the right
    if (get_col(_map, tile) != _map.num_cols - 1) and (not is_blocked(_map, tile + 1)):
        adjacent_tiles.append(tile + 1)
    # Check the tile to the left
    if (get_col(_map, tile) != 0) and (not is_blocked(_map, tile - 1)):
        adjacent_tiles.append(tile - 1)
    # Check the tile above
    if (get_row(_map, tile) != 0) and (not is_blocked(_map, tile - _map.num_cols)):
        adjacent_tiles.append(tile - _map.num_cols)
    # Check the tile below
    if (get_row(_map, tile) != _map.num_rows - 1) and (not is_blocked(_map, tile + _map.num_cols)):
        adjacent_tiles.append(tile + _map.num_cols)
    # Add the current tile (wait move)
    adjacent_tiles.append(tile)

    return adjacent_tiles

def manhattan_distance(start_x, start_y, end_x, end_y):
    return abs(end_x - start_x) + abs(end_y - start_y)

def manhattan_distance_between_tiles(_map: MAPF_Map, start_tile, end_tile):
    return manhattan_distance(
        get_row(_map, start_tile),
        get_col(_map, start_tile),
        get_row(_map, end_tile),
        get_col(_map, end_tile)
    )

def get_a_star_path(goal_node: AStarNode):
    path = []
    current_node = goal_node
    while current_node is not None:
        path.append(current_node.tile)
        current_node = current_node.parent
    path.reverse()
    return path
