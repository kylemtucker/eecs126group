import math
import copy
import itertools
import numpy as np

### GLOBALS
ROLL_PROBABILITIES = np.array([1/36,  #2 
                               2/36,  #3
                               3/36,  #4
                               4/36,  #5
                               5/36,  #6
                               6/36,  #7
                               5/36,  #8
                               4/36,  #9
                               3/36,  #10
                               2/36,  #11
                               1/36]) #12

SETTLEMENT = 0
CARD = 1
CITY = 2
ROAD = 3
COSTS = np.array([[2, 1, 1],
                  [1, 2, 2],
                  [0, 3, 3],
                  [1, 1, 0]])

WOOD = 0
BRICK = 1
GRAIN = 2
RESOURCES = range(3)

MAX_POINTS = 10
START_RESOURCES = 3
ROBBER_MAX_RESOURCES = 7

class Goal:
    def __init__(self, type, num_turns=None, end=None, start=None):
        self.type = type
        self.num_turns = num_turns
        self.end = end
        self.start = start

    #For debugging.
    def __str__(self):
        if self.type == 0:
            type = ("settlement")
        if self.type == 1:
            type = ("card")
        if self.type == 2:
            type = ("city")
        if self.type == 3:
            type = ("road")
        return "Goal: {0} from {1} to {2}".format(type, self.start, self.end)


"""
Returns the Manhattan distance between START and END, two 2-tuples.
"""
def manhattan_distance(start, end):
    return abs(start[0] - end[0]) + abs(start[1] - end[1])

"""
Returns a numpy 3-array representing the number of resources that the player
with PLAYER_ID is expected to gain from the BOARD after each dice roll, ordered
by: [wood, brick, grain].
"""
def get_expected_resources_per_turn(player_id, board):
    return ROLL_PROBABILITIES.dot(board.get_resources(player_id))

"""
Returns a numpy 3-array representing the exchange rate that the player
with PLAYER_ID has on [wood, brick, grain] on BOARD.

The exchange rate of a resource is how much of that resource is required
to trade for another resouce.
"""
def get_exchange_rate(player_id, board):
    #Find which ports the player owns.
    discounts = []
    for settlement in board.get_player_settlements(player_id):
        if board.is_port(settlement):
            discounts.append(board.which_port(settlement))
    for city in board.get_player_cities(player_id):
        if board.is_port(city):
            discounts.append(board.which_port(city))
    #Calculate the exchange rate based on which ports the player owns.
    if 3 in discounts:
        exchange_rate = [3, 3, 3]
        discounts.remove(3)
    else:
        exchange_rate = [4, 4, 4]
    for resource in discounts:
        exchange_rate[resource] = 2
    return np.array(exchange_rate)

"""
Returns an admissible heuristic for the minimum expected number of turns it takes
for the player with PLAYER_ID to earn the REQUIRED resources on the board BOARD,
assuming that dumping resources upon a dice roll of 7 has a negligible impact on
resource aquisition.
"""
def get_turns_for_resources(required, player_id, board):
    exchange_rate = get_exchange_rate(player_id, board)
    expected_resources_per_turn = get_expected_resources_per_turn(player_id, board)
    possible_resources = [r for r in range(3) if expected_resources_per_turn[r] > 0]
    if len(possible_resources) == 0:
        return float("inf")
    #Take the array of required resources and figure out if trades are necessary.
    #If so, consider all possible maximal trades for each necessary trade. A trade
    #is defined to be maximal if all of one of the desired resources come from
    #trading only one other resource.
    #This implementation uses DFS to search that space.
    equivalent_required = []
    stack = [required]
    while stack:
        required = stack.pop()
        unobtainable_resource = -1
        #Find children based on the first resource must be traded for.
        for resource in range(3):
            if required[resource] != 0 and expected_resources_per_turn[resource] == 0:
                unobtainable_resource = resource
                break
        if unobtainable_resource != -1:
            #We consider trades here.
            for resource in possible_resources:
                subsitute = required.copy()
                subsitute[resource] += exchange_rate[resource] * required[unobtainable_resource]
                subsitute[unobtainable_resource] = 0
                stack.append(subsitute)
        else:
            #No need to consider trades for this node.
            equivalent_required.append(required)
    #Compute the best equivalent equivalent resource array.
    impossible_resources = [r for r in range(3) if expected_resources_per_turn[r] <= 0]
    expected_resources_per_turn = np.delete(expected_resources_per_turn, impossible_resources)
    equivalent_required = np.delete(np.array(equivalent_required), impossible_resources, axis=1)

    #Number of turns for each equivalent resource array.
    num_turns = np.amax(equivalent_required / expected_resources_per_turn, axis=1)
    return np.amin(num_turns)

"""
Returns a the expected number of turns it will take for the player with PLAYER_ID on
BOARD to win the game by buying victory cards, given that they currently have NUM_POINTS
points.
"""
def get_turns_for_cards(num_points, player_id, board):
    if num_points >= MAX_POINTS:
        return 0
    return get_turns_for_resources((MAX_POINTS - num_points) * COSTS[CARD], player_id, board)

def get_first_settlement_goal(player_id, board):
    #Criteria for best first settlement. Smaller is better.
    def num_turns(settlement):
        #Assume state after purchasing settlement.
        board.settlements[settlement] = player_id
        #Calculate expected time to win.
        result = get_turns_for_cards(0, player_id, board)
        #Restore state.
        del board.settlements[settlement]
        return result
    best_end = board.get_vertex_location(min(range(board.max_vertex), key=num_turns))

    return Goal(SETTLEMENT, end=best_end)

"""
Insert some comment.
"""
def get_next_settlement_goal(num_points, player_id, board):
    #Get current buildings.
    current_buildings = set(map(board.get_vertex_location, board.get_player_settlements(player_id)))
    current_buildings.update(map(board.get_vertex_location, board.get_player_cities(player_id)))

    #Coordinates where we may begin a road to the next city.
    start_locations = set(map(board.get_vertex_location, itertools.chain(*board.get_player_roads(player_id))))
    start_locations.update(current_buildings)

    #Places where it is invalid to build another settlement.
    invalid_locations = current_buildings.copy()
    invalid_locations.update([(x + 1, y) for x, y in current_buildings])
    invalid_locations.update([(x - 1, y) for x, y in current_buildings])
    invalid_locations.update([(x, y + 1) for x, y in current_buildings])
    invalid_locations.update([(x, y - 1) for x, y in current_buildings])

    #A dict of expected time it takes to build a settlement that requires
    #some number of roads, indexed by the number of roads.
    settlement_costs = {}

    #Compute best next settlement.
    best_end, best_start, min_num_turns = None, None, float("inf")
    for settlement in range(board.max_vertex):
        end = board.get_vertex_location(settlement)
        if end not in invalid_locations:
            start = min(start_locations, key=lambda start:manhattan_distance(start, end))
            #Compute heuristic of number of turns for building settlement.
            num_roads = manhattan_distance(start, end)
            if num_roads not in settlement_costs:
                required = num_roads * COSTS[ROAD] + COSTS[SETTLEMENT]
                settlement_costs[num_roads] = get_turns_for_resources(required, player_id, board)
            num_turns = settlement_costs[num_roads].copy()
            #Add heuristic for time to win after building settlement.
            board.settlements[settlement] = player_id
            num_turns += get_turns_for_cards(num_points, player_id, board)
            del board.settlements[settlement]
            #Compute (arg)minimums.
            if num_turns < min_num_turns:
                best_end = end
                best_start = start
                min_num_turns = num_turns
    #Check if no settlements left to build.
    if best_end == None:
        return None
    else:
        return Goal(SETTLEMENT, num_turns=min_num_turns, end=best_end, start=best_start)

"""
Insert some comment.
"""
def get_next_city_goal(num_points, player_id, board):
    #Saves some computation below.
    num_points += 1

    #Compute best next city.
    best_city, min_num_turns = None, float("inf")
    for city in board.get_player_settlements(player_id):
        #Computes heuristic number of turns to win after building city.
        board.cities[city] = player_id
        del board.settlements[city]
        #Implicitly, here num_points is +1 from the original argument to save
        #minor computation, since building a city gains +1 victory points.
        num_turns = get_turns_for_cards(num_points + 1, player_id, board)
        del board.cities[city]
        board.settlements[city] = player_id
        #Compute (arg)minumums.
        if num_turns < min_num_turns:
            best_city, min_num_turns = city, num_turns

    min_num_turns += get_turns_for_resources(COSTS[CITY], player_id, board)

    if best_city == None:
        return None
    else:
        best_end = board.get_vertex_location(best_city)
        return Goal(CITY, num_turns=min_num_turns, end=best_end)

################## ABOVE IS DONE ##############################


################## POORLY DOCUMENTED BELOW ##############################
def action(self):
    while len(self.preComp) > 0:
        goal = self.preComp.pop()
        exchange_rate = get_exchange_rate(self.player_id, self.board)
        #Resources that we need more of (deficits) are positive; 
        #resources that we can use less of (surpluses) are negative.
        required = COSTS[goal.type] - self.resources
        #Trading policy. While we have deficits and surpluses, trade.
        while (required > 0).any() and (required <= -exchange_rate).any():
            self.trade(np.argmin(required), np.argmax(required))
            required = COSTS[goal.type] - self.resources
        #Buy, if possible.
        if (required <= 0).all():
            if goal.type == CARD:
                self.buy("card")
            elif goal.type == ROAD:
                self.buy("road", x=goal.start, y=goal.end)
            elif goal.type == SETTLEMENT:
                self.buy("settlement", x=goal.end[0], y=goal.end[1])
            elif goal.type == CITY:
                self.buy("city", x=goal.end[0], y=goal.end[1])
        else:
            self.preComp.append(goal)
            break

def planBoard(baseBoard):
    plan = []
    num_points = 0
    board = copy.deepcopy(baseBoard)

    #Plan to settle at the best first settlement location.
    player_id = board.num_players
    first_goal = get_first_settlement_goal(player_id, board)
    board.settlements[board.get_vertex_number(first_goal.end[0], first_goal.end[1])] = player_id
    plan.append(first_goal)

    #Decide next goals.
    while num_points < MAX_POINTS:
        #Default goal is to buy cards.
        next_goal = Goal(CARD, get_turns_for_cards(num_points, player_id, board))
        #Then check if settlement building is better.
        goal = get_next_settlement_goal(num_points, player_id, board)
        if goal != None and goal.num_turns < next_goal.num_turns:
            next_goal = goal
        #Finally, check if city building is better.
        goal = get_next_city_goal(num_points, player_id, board)
        if goal != None and goal.num_turns < next_goal.num_turns:
            next_goal = goal

        #Add concrete goals to plan and update the planning board.
        if next_goal.type == CARD:
            while num_points < MAX_POINTS:
                plan.append(next_goal)
                num_points += 1
        elif next_goal.type == SETTLEMENT:
            start = next_goal.start
            end = next_goal.end
            #Add road building goals.
            while start != end:
                if start[0] < end[0]:
                    next = (start[0] + 1, start[1])
                elif start[0] > end[0]:
                    next = (start[0] - 1, start[1])
                elif start[1] < end[1]:
                    next = (start[0], start[1] + 1)
                else:
                    next = (start[0], start[1] - 1)

                plan.append(Goal(ROAD, end=next, start=start))
                start_num = board.get_vertex_number(start[0], start[1])
                next_num = board.get_vertex_number(next[0], next[1])
                board.roads[(start_num, next_num)] = player_id
                start=next
            #Add settlement goals.
            plan.append(next_goal)
            board.settlements[board.get_vertex_number(end[0], end[1])] = player_id
        elif next_goal.type == CITY:
            plan.append(next_goal)
            city_num = board.get_vertex_number(next_goal.end[0], next_goal.end[1])
            board.cities[city_num] = player_id
            del board.settlements[city_num]
            num_points += 1
    #So we can push and pop effectively.
    plan.reverse()
    return plan

############################# TO DO BELOW ##################################
def dumpPolicy(self, max_resources):
    new_resources = np.minimum(self.resources, max_resources // 3)
    return self.resources - new_resources
