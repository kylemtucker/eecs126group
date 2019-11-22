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
ROBBER_MAX_RESOURCES = 7

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
        unobtainable_resource = None
        #Find children based on the first resource must be traded for.
        for resource in range(3):
            if required[resource] != 0 and expected_resources_per_turn[resource] == 0:
                unobtainable_resource = resource
                break
        if unobtainable_resource != None:
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
Returns a the expected number of turns it will take for the player with PLAYER_ID
on BOARD to obtain MAX_POINTS - NUM_POINTS victory cards using only maximal required
trades (see: get_turns_for_resources(required, player_id, board)).
"""
def get_turns_for_cards(num_points, player_id, board):
    if num_points >= MAX_POINTS:
        return 0
    return get_turns_for_resources((MAX_POINTS - num_points) * COSTS[CARD], player_id, board)

"""
Returns a Goal for the first settlement that minimizes the expected number of turns it takes
for the player with PLAYER_ID on BOARD to:
1) Build a settlements at some new location, without building roads;
2) Buy victory cards until the player has MAX_POINTS - NUM_POINTS points.
Returns None if there is no valid location on which to build a settlement.
"""
def get_first_settlement_goal(player_id, board):
    #Criteria for best first settlement.
    def num_turns(settlement):
        board.settlements[settlement] = player_id
        result = get_turns_for_cards(0, player_id, board)
        del board.settlements[settlement]
        return result
    best_end = board.get_vertex_location(min(range(board.max_vertex), key=num_turns))
    return Goal(SETTLEMENT, end=best_end)

"""
Returns a Goal that minimizes the expected number of turns it takes
for the player with PLAYER_ID on BOARD to:
1) Build a road to some location that is not adjacent to pre-existing
   settlements or cities;
2) Build a settlements at this new location;
3) Buy victory cards until the player has MAX_POINTS - NUM_POINTS points.
Returns None if there is no valid location on which to build a settlement
or if there is no existing settlement on the board beloning to the player.
"""
def get_next_settlement_goal(num_points, player_id, board):
    #Get current buildings.
    current_buildings = set(map(board.get_vertex_location, board.get_player_settlements(player_id)))
    current_buildings.update(map(board.get_vertex_location, board.get_player_cities(player_id)))
    #Get coordinates where we may begin a road to the next city.
    start_locations = set(map(board.get_vertex_location, itertools.chain(*board.get_player_roads(player_id))))
    start_locations.update(current_buildings)
    #Get places where it is invalid to build another settlement.
    invalid_locations = current_buildings.copy()
    invalid_locations.update([(x + 1, y) for x, y in current_buildings])
    invalid_locations.update([(x - 1, y) for x, y in current_buildings])
    invalid_locations.update([(x, y + 1) for x, y in current_buildings])
    invalid_locations.update([(x, y - 1) for x, y in current_buildings])
    #This function only computes NEXT settlements. Use 
    #get_first_settlement_goal(player_id, board) instead.
    if len(current_buildings) == 0:
        return None
    #This stores the expected time it takes to build a settlement that requires
    #some number of roads, indexed by the number of roads to reduce compute.
    settlement_costs = {}
    #Compute best next settlement.
    best_end, best_start, min_num_turns = None, None, float("inf")
    for settlement in range(board.max_vertex):
        end = board.get_vertex_location(settlement)
        if end not in invalid_locations:
            #Compute (or retrieve from the above dict) the number of turns it takes
            #to build the road to the new settlement and the settlement itself.
            start = min(start_locations, key=lambda start:manhattan_distance(start, end))
            num_roads = manhattan_distance(start, end)
            if num_roads not in settlement_costs:
                required = num_roads * COSTS[ROAD] + COSTS[SETTLEMENT]
                settlement_costs[num_roads] = get_turns_for_resources(required, player_id, board)
            num_turns = settlement_costs[num_roads].copy()
            #Compute the number of turns it takes to buy the necessary victory points,
            #after having built the settlement.
            board.settlements[settlement] = player_id
            num_turns += get_turns_for_cards(num_points, player_id, board)
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
Returns a Goal that minimizes the expected number of turns it takes
for the player with PLAYER_ID on BOARD to:
1) Build a city at some existing settlement;
2) Buy victory cards until the player has MAX_POINTS - NUM_POINTS points.
Returns None if there is no existing settlement on which to build a city.
"""
def get_next_city_goal(num_points, player_id, board):
    #Saves some computation below.
    num_points += 1
    #Compute best next city.
    best_city, min_num_turns = None, float("inf")
    for city in board.get_player_settlements(player_id):
        #Computes the number of turns it takes to the necessary victory points,
        #after having built the city.
        board.cities[city] = player_id
        del board.settlements[city]
        #Implicitly, here num_points is +1 from the original argument to save
        #minor computation, since building a city gains +1 victory points.
        num_turns = get_turns_for_cards(num_points, player_id, board)
        del board.cities[city]
        board.settlements[city] = player_id
        #Keep track of relevent (arg)mins.
        if num_turns < min_num_turns:
            best_city, min_num_turns = city, num_turns
    #We only compute this once, since it is the same for all city locations.
    min_num_turns += get_turns_for_resources(COSTS[CITY], player_id, board)
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
        next_goal = Goal(CARD, num_turns=get_turns_for_cards(num_points, player_id, board))
        #2) Decide if settlement building is better.
        goal = get_next_settlement_goal(num_points, player_id, board)
        if goal != None and goal.num_turns < next_goal.num_turns:
            next_goal = goal
        #3) Decide if city building is better.
        goal = get_next_city_goal(num_points, player_id, board)
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
    #Reverse the list so we can push (append) and pop goals from the plan efficiently.
    plan.reverse()
    return plan

############################# TO DO BELOW ##################################
def dumpPolicy(self, max_resources):
    goal = self.preComp.pop()
    surplus = self.resources - COSTS[goal.type]
    surplus[surplus < 0] = 0
    self.preComp.append(goal)

    num_resources = np.sum(self.resources)
    dump = np.zeros(3)

    while num_resources > ROBBER_MAX_RESOURCES:
        resource = np.argmax(surplus)
        dump[resource] += 1
        surplus[resource] -= 1
        num_resources -= 1
    return dump
