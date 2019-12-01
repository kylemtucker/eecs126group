import math
import copy
import itertools
import numpy as np

# Average turns to win: 75.119 #

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

MAX_POINTS = 10

class Goal:
    def __init__(self, type, end=None, start=None, num_turns=None):
        self.type = type
        self.end = end
        self.start = start
        self.num_turns = num_turns

    #For debugging.
    def __str__(self):
        if self.type == 0:
            type = "settlement"
        if self.type == 1:
            type = "card"
        if self.type == 2:
            type = "city"
        if self.type == 3:
            type = "road"
        return "Goal: {0} from {1} to {2} in {3}.".format(type, self.start, self.end, self.num_turns)


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
Returns a numpy 3-array representing the number of each resources that the
player with PLAYER_ID needs to trade for any other resource, ordered by:
[wood, brick, grain].
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
Returns the heuristic for the minimum expected number of turns it takes for the
player with PLAYER_ID to earn the REQUIRED resources on the board BOARD. In other
words, this function returns the proxy for hitting time to obtain some list of
resources.

This implementation assumes that dumping resources upon a dice roll of 7 has a
negligable impact on hitting time.
"""
def get_turns_for_resources(player_id, board, required):
    exchange_rate = get_exchange_rate(player_id, board)
    expected_resources_per_turn = get_expected_resources_per_turn(player_id, board)

    if (expected_resources_per_turn == 0).all():
        return float("inf")
    #Iteratively finds the number of turns it takes to achieve REQUIRED.
    num_turns = 0
    required = required.astype(float)
    while (required > 0).any():
        num_turns += 1
        required -= expected_resources_per_turn
        trades = -(required / exchange_rate).astype(int)
        for resource in range(3):
            for _ in range(trades[resource]):
                required[np.argmax(required)] -= 1
                required[resource] += exchange_rate[resource]
    return num_turns

"""
Returns the hitting time for the player with PLAYER_ID on BOARD to obtain
MAX_POINTS - NUM_POINTS victory cards.
"""
def get_turns_for_cards(player_id, board, num_points):
    if num_points >= MAX_POINTS:
        return 0
    return get_turns_for_resources(player_id, board, (MAX_POINTS - num_points) * COSTS[CARD])

"""
Returns a Goal for the first settlement.

The current implementation chooses the settlement that minimizes
the hitting time until the player with PLAYER_ID on BOARD wins via
buying victory cards after having purchased that first settlement.
"""
def get_first_settlement_goal(player_id, board):
    #Criteria for best first settlement.
    def num_turns(settlement):
        board.settlements[settlement] = player_id
        result = get_turns_for_cards(player_id, board, 0)
        del board.settlements[settlement]
        return result
    return Goal(SETTLEMENT, end=board.get_vertex_location(min(range(board.max_vertex), key=num_turns)))

"""
Returns a set of (x, y) coordinates where the player with PLAYER_ID
cannot build a settlement on BOARD, even if they have a road to such
locations.

This set is generated from coordinates where the player has an existing
settlement or city and from coordinates which are adjacent to an existing
settlement or city, as per the rules.
"""
def get_invalid_settlement_locations(player_id, board):
    building_locations = set()
    building_locations.update(map(board.get_vertex_location, board.get_player_settlements(player_id)))
    building_locations.update(map(board.get_vertex_location, board.get_player_cities(player_id)))

    invalid_locations = set(building_locations)
    invalid_locations.update((x + 1, y) for x, y in building_locations if board.is_tile(x, y))
    invalid_locations.update((x - 1, y) for x, y in building_locations if board.is_tile(x, y))
    invalid_locations.update((x, y + 1) for x, y in building_locations if board.is_tile(x, y))
    invalid_locations.update((x, y - 1) for x, y in building_locations if board.is_tile(x, y))

    return invalid_locations

"""
Returns a set of (x, y) coordinates where the player with PLAYER_ID
can begin a road.
"""
def get_road_start_locations(player_id, board):
    road_locations = set()
    road_locations.update(map(board.get_vertex_location, itertools.chain(*board.get_player_roads(player_id))))
    road_locations.update(map(board.get_vertex_location, board.get_player_settlements(player_id)))
    road_locations.update(map(board.get_vertex_location, board.get_player_cities(player_id)))
    return road_locations

"""
Returns a Goal for the next (not first) settlement.

The current implementation chooses the settlement that minimizes
the hitting time until the player with PLAYER_ID on BOARD wins via
buying victory cards after having built the shortest possible road
to such a settlement location and then building that settlement.
"""
def get_next_settlement_goal(player_id, board, num_points):
    road_start_locations = get_road_start_locations(player_id, board)
    invalid_settlement_locations = get_invalid_settlement_locations(player_id, board)

    #This stores the expected time it takes to build a settlement that requires
    #some number of roads, indexed by the number of roads to reduce compute.
    settlement_costs = {}

    best_end, best_start, min_num_turns = None, None, float("inf")
    for settlement in range(board.max_vertex):
        end = board.get_vertex_location(settlement)
        if end not in invalid_settlement_locations:
            start = min(road_start_locations, key=lambda start:manhattan_distance(start, end))
            num_roads = manhattan_distance(start, end)

            if num_roads not in settlement_costs:
                required = num_roads * COSTS[ROAD] + COSTS[SETTLEMENT]
                settlement_costs[num_roads] = get_turns_for_resources(player_id, board, required)
            num_turns = settlement_costs[num_roads]

            #Compute the number of turns it takes to buy the necessary victory points,
            #after having built the settlement.
            board.settlements[settlement] = player_id
            num_turns += get_turns_for_cards(player_id, board, num_points)
            del board.settlements[settlement]

            #Keep track of relevent (arg)mins.
            if num_turns < min_num_turns:
                best_end = end
                best_start = start
                min_num_turns = num_turns

    #If settlements can no longer be built, return None.
    if best_end == None:
        return None
    else:
        return Goal(SETTLEMENT, end=best_end, start=best_start, num_turns=min_num_turns)

"""
Returns a Goal for the next city.

The current implementation chooses the city that minimizes the
hitting time until the player with PLAYER_ID on BOARD wins via
buying victory cards after having built the city.
"""
def get_next_city_goal(player_id, board, num_points):
    #Compute best next city.
    best_city, min_num_turns = None, float("inf")
    for city in board.get_player_settlements(player_id):
        board.cities[city] = player_id
        del board.settlements[city]
        num_turns = get_turns_for_cards(player_id, board, num_points + 1)
        del board.cities[city]
        board.settlements[city] = player_id
        
        if num_turns < min_num_turns:
            best_city, min_num_turns = city, num_turns

    #We only compute this once, after the loop, since it is the same for all city locations.
    min_num_turns += get_turns_for_resources(player_id, board, COSTS[CITY])
    
    #If cities can no longer be built, return None.
    if best_city == None:
        return None
    else:
        best_end = board.get_vertex_location(best_city)
        return Goal(CITY, end=best_end, num_turns=min_num_turns)

"""
This method pops goals off the plan returned by planBoard(baseBoard)
and then trades/rolls the dice to achieve each goal incrementally.
"""
def action(self):
    while len(self.preComp) > 0:
        #Get next goal.
        goal = self.preComp.pop()
        exchange_rate = get_exchange_rate(self.player_id, self.board)
        #Required resources that we need more of are positive; 
        #required resources that we can use less of are negative.
        #Use this information to to consider trades.
        required = COSTS[goal.type] - self.resources
        while (required > 0).any() and (required <= -exchange_rate).any():
            #np.argmin(required / exchange_rate) calculates the resource with the
            #highest ~relative~ surplus, rather than highest absolute surplus.
            self.trade(np.argmin(required / exchange_rate), np.argmax(required))
            required = COSTS[goal.type] - self.resources
        #Buy (and achieve the goal), if possible.
        if (required <= 0).all():
            #print(goal)
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

"""
In the current implementation, this function retuns a plan (a list of goals)
which the bot uses to take actions on BASEBOARD. 

See proposal and/or above functions and methods for more information on how
the plan is generated.
"""
def planBoard(baseBoard):
    plan = []
    num_points = 0
    board = copy.deepcopy(baseBoard)
    player_id = board.num_players

    #Plan to settle as first goal.
    first_goal = get_first_settlement_goal(player_id, board)

    plan.append(first_goal)
    board.settlements[board.get_vertex_number(first_goal.end[0], first_goal.end[1])] = player_id
    
    #Decide next goals.
    while num_points < MAX_POINTS:
        #1) Default goal is to buy cards.
        next_goal = Goal(CARD, num_turns=get_turns_for_cards(player_id, board, num_points))
        #2) Decide if settlement building is better.
        goal = get_next_settlement_goal(player_id, board, num_points)
        if goal != None and goal.num_turns < next_goal.num_turns:
            next_goal = goal
        #3) Decide if city building is better.
        goal = get_next_city_goal(player_id, board, num_points)
        if goal != None and goal.num_turns < next_goal.num_turns:
            next_goal = goal

        if next_goal.type == CARD:
            #Plan to buy cards until the game is over.
            while num_points < MAX_POINTS:
                plan.append(next_goal)
                num_points += 1
        elif next_goal.type == SETTLEMENT:
            start = next_goal.start
            end = next_goal.end
            #Plan to build roads to next settlement.
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
            #Plan to build settlement.
            plan.append(next_goal)
            board.settlements[board.get_vertex_number(end[0], end[1])] = player_id
        else:
            #Plan to build city.
            plan.append(next_goal)
            city_num = board.get_vertex_number(next_goal.end[0], next_goal.end[1])
            board.cities[city_num] = player_id
            del board.settlements[city_num]
            num_points += 1
    #print(get_num_turns(player_id, baseBoard, plan))
    #Reverse the list so we can push (append) and pop goals from the plan efficiently.
    plan.reverse()
    return plan

def get_num_turns(player_id, board, goals):
    board = copy.deepcopy(board)
    num_turns = 0

    goal = goals[0]
    board.settlements[board.get_vertex_number(goal.end[0], goal.end[1])] = player_id
    for goal in goals[1:]:
        if goal.type == ROAD:
            end_vertex_number = board.get_vertex_number(goal.end[0], goal.end[1])
            start_vertex_number = board.get_vertex_number(goal.start[0], goal.start[1])
            board.roads[(start_vertex_number, end_vertex_number)] = player_id
            num_turns += get_turns_for_resources(player_id, board, COSTS[ROAD])
        elif goal.type == SETTLEMENT:
            end_vertex_number = board.get_vertex_number(goal.end[0], goal.end[1])
            board.settlements[end_vertex_number] = player_id
            num_turns += get_turns_for_resources(player_id, board, COSTS[SETTLEMENT])
        elif goal.type == CITY:
            end_vertex_number = board.get_vertex_number(goal.end[0], goal.end[1])
            board.cities[end_vertex_number] = player_id
            del board.settlements[end_vertex_number]
            num_turns += get_turns_for_resources(player_id, board, COSTS[CITY])
        else:
            num_turns += get_turns_for_resources(player_id, board, COSTS[CARD])
    return num_turns

############################# TO DO BELOW ##################################
def dumpPolicy(self, max_resources):
    goal = self.preComp.pop()
    surplus = self.resources - COSTS[goal.type]
    surplus[surplus < 0] = 0
    self.preComp.append(goal)

    num_resources = np.sum(self.resources)
    dump = np.zeros(3)

    while num_resources > max_resources:
        resource = np.argmax(surplus)
        dump[resource] += 1
        surplus[resource] -= 1
        num_resources -= 1
    return dump
