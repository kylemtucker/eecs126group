import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import math

### GLOBALS

SETTLEMENT = 0
CARD = 1
CITY = 2
ROAD = 3
MAX_POINTS = 10
ROBBER_MAX_RESOURCES = 7
START_RESOURCES = 3

costs = np.array([[2, 1, 1],
                  [1, 2, 2],
                  [0, 3, 3],
                  [1, 1, 0]])

class CatanException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class Catan:
    def __init__(self, dice, resources, settlements = {}, cities = {}, roads = {}):
        self.width = dice.shape[1]
        self.height = dice.shape[0]
        self.dice = dice
        self.resources = resources
        self.settlements = settlements
        self.cities = cities
        self.roads = roads
        self.max_vertex = (self.width+1)*(self.height+1) - 1
        self.num_players = 0

        self._clear_robber_resources()

    def clear_board(self):
        self.settlements = {}
        self.cities = {}
        self.roads = {}
        self.num_players = 0

    def register_player(self):
        self.num_players += 1
        return self.num_players

    def _clear_robber_resources(self):
        for x in range(self.width):
            for y in range(self.height):
                if self.dice[y, x] == 7:
                    self.resources[y, x] = -1

    def is_port(self, vertex):
        return vertex == 0 or vertex == self.width or vertex == self.max_vertex or vertex == self.max_vertex - self.width

    ## 0 - 2:1 wood
    ## 1 - 2:1 brick
    ## 2 - 2:1 grain
    ## 3 - 3:1 general
    def which_port(self, vertex):
        if vertex == 0:
            return 0
        elif vertex == self.width:
            return 1
        elif vertex == self.max_vertex - self.width:
            return 2
        elif vertex == self.max_vertex:
            return 3
        else:
            raise CatanException("{0} is not a port".format(vertex))

    def get_vertex_number(self, x, y):
        return (self.height + 1) * y + x

    def get_vertex_location(self, n):
        return (n % (self.height+1), n // (self.height+1))

    def get_player_settlements(self, player_id):
        return [v for v in self.settlements if self.settlements[v] == player_id]

    def get_player_cities(self, player_id):
        return [v for v in self.cities if self.cities[v] == player_id]

    def get_player_roads(self, player_id):
        return [road for road in self.roads if self.roads[road] == player_id]

    def is_tile(self, x, y):
        """returns whether x,y is a valid tile"""
        return x >= 0 and x < self.width and y >= 0 and y < self.width

    def build_road(self, c0, c1, player_id=0):
        v0 = self.get_vertex_number(c0[0], c0[1])
        v1 = self.get_vertex_number(c1[0], c1[1])
        if self.if_can_build_road(v0, v1, player_id):
            self.roads[(v0, v1)] = player_id
        else:
            raise CatanException("({0},{1}) is an invalid road".format(c0, c1))

    def if_can_build_road(self, start, end, player_id=0):
        ##order the road vertices
        temp = max(start, end)
        v1 = min(start, end)
        v2 = temp
        """returns true if road is valid, false otherwise"""
        #check if road vertices are on the map
        if v1 < 0 or v2 < 0 or v1 > self.max_vertex or v2 > self.max_vertex:
            raise CatanException("({0},{1}) is an invalid road".format(v1, v2))
        if v1 == v2: return False
        #first let's check that the spot is empty:
        if (v1, v2) in self.roads or (v2, v1) in self.roads:
            return False

        #now let's check if the proposed road is valid.
        #CORNER CASES
        if v1 == 0 or v2 == 0:
            if not (v1 + v2 == 1 or v1 + v2 == self.width+1):
                return False
        if v1 == self.width or v2 == self.width:
            if not (v1 + v2 == 2*self.width - 1 or v1 + v2 == 3*self.width+ 1):
                return False
        if v1 == (self.width + 1)*self.height or v2 == (self.width + 1)*self.height:
            if not (v1 + v2 == 2*(self.width + 1)*self.height + 1 or v1 + v2 == (self.width + 1)*(2*self.height - 1)):
                return False
        if v1 == self.max_vertex or v2 == self.max_vertex:
            if not (v1 + v2== 2*self.max_vertex - 1 or v1 + v2== (2 * self.max_vertex - (self.width + 1))):
                return False
        #EDGE CASES... literally --
        ## left edge
        if v1%(self.width + 1) == 0 or v2%(self.width + 1) == 0:
            if not (v2 - v1 == self.width + 1 or v2 - v1 == 1):
                return False
        ## bottom edge
        if v1 in range(1, self.width + 1) or v2 in range(1, self.width + 1):
            if not (v2 - v1 == self.width + 1 or v2 - v1 == 1):
                return False
        ## right edge
        if v1 in range(self.width, self.max_vertex + 1, self.width + 1) or v2 in range(self.width, self.max_vertex + 1, self.width + 1):
            if not (v2 - v1 == self.width + 1 or (v2 - v1 and v2%(self.width + 1) != 0) == 1):
                return False
        ## top edge
        if v1 in range(self.max_vertex - self.width + 1, self.max_vertex) or v2 in range(self.max_vertex - self.width + 1, self.max_vertex):
            if not (v2 - v1 == self.width + 1 or v2 - v1 == 1):
                return False
        #GENERAL CASE
        if not (v2 - v1 == self.width + 1 or v2 - v1 == 1): return False

        #If there are no roads, it must be connected to a settlement or a city
        if len(self.get_player_roads(player_id)) == 0:
            if (v1 not in self.settlements or self.settlements[v1] != player_id) and \
                    (v2 not in self.settlements or self.settlements[v2] != player_id) and \
                    (v1 not in self.cities or self.cities[v1] != player_id) and \
                    (v2 not in self.cities or self.cities[v2] != player_id):
                return False

        #Otherwise, it must be connected to another road
        else:
            connected_players = set([self.roads[road] for road in self.roads if v1 in road or v2 in road])
            if player_id not in connected_players:
                return False
        return True


    def build(self, x, y, building, player_id=0):
        """build either a city or a settlement"""
        if self.if_can_build(building, x, y, player_id):
            vertex = self.get_vertex_number(x, y)
            if building == "settlement":
                self.settlements[vertex] = player_id
            elif building == "city":
                if vertex not in self.settlements:
                    raise CatanException("A settlement must be built first.")
                self.cities[vertex] = player_id
                self.settlements.pop(vertex)
            else:
                raise CatanException("{0} is an unknown building. Please use 'city' or 'settlement'.".format(building))
        else:
            raise CatanException("Cannot build {0} here. Please check if_can_build before building".format(building))


    def if_can_build(self, building, x, y, player_id=0):
        """returns true if spot (x,y) is available, false otherwise"""
        if x< 0 or y<0 or x > self.width+1 or y > self.height + 1:
            raise CatanException("({0},{1}) is an invalid vertex".format(x,y))

        v = self.get_vertex_number(x,y)
        #first let's check that the spot is empty:
        if v in self.cities:
            return False

        ## upgrading first settlment into a city
        if (building == "city"):
            return v in self.settlements and self.settlements[v] == player_id
        ## If no cities, or settlements, build for freebies, otherwise need road connecting.
        if (len(self.get_player_settlements(player_id)) + len(self.get_player_cities(player_id)) != 0 and \
                v not in set([element for tupl in self.get_player_roads(player_id) for element in tupl])):
            return False
        for x1 in range(x-1,x+2):
            for y1 in range(y-1,y+2):
                if x1+y1 < x+y-1 or x1+y1 > x+y+1 or y1-x1 < y-x-1 or y1-x1 > y-x+1: ## only interested in up, down, left, and right
                    pass
                elif x1 < 0 or x1 > self.width or y1 < 0 or y1 > self.height: ## only interested in valid tiles
                    pass
                elif self.get_vertex_number(x1, y1) in self.settlements or self.get_vertex_number(x1, y1) in self.cities:
                    return False
        return True

    def get_resources(self, player_id=0):
        """Returns array r where:
        r[i, :] = resources gained from throwing a (i+2)"""
        r = np.zeros((11, 3))
        for vertex in self.get_player_settlements(player_id):
            x, y = self.get_vertex_location(vertex)
            for dx in [-1, 0]:
                for dy in [-1, 0]:
                    xx = x + dx
                    yy = y + dy
                    if self.is_tile(xx, yy):
                        die = self.dice[yy, xx]
                        if die != 7:
                            resource = self.resources[yy, xx]
                            r[die - 2, resource] += 1
        for vertex in self.get_player_cities(player_id):
            x, y = self.get_vertex_location(vertex)
            for dx in [-1, 0]:
                for dy in [-1, 0]:
                    xx = x + dx
                    yy = y + dy
                    if self.is_tile(xx, yy):
                        die = self.dice[yy, xx]
                        if die != 7:
                            resource = self.resources[yy, xx]
                            r[die - 2, resource] += 2
        return r

    def draw(self):
        print("Drawing...")
        color_map = plt.cm.get_cmap('hsv')
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim(-0.02,self.width+0.02)
        ax.set_ylim(-0.02,self.height+0.02)
        ax.set_frame_on(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        for x in range(self.width):
            for y in range(self.height):
                color = ["brown", "red", "green", "khaki"][self.resources[y, x]]
                ax.add_patch(patches.Rectangle((x, y),1,1,
                                               facecolor=color, ec = "black"))
                if self.dice[y,x] != 0:
                    ax.text(x+0.5, y+0.5, str(self.dice[y, x]), fontsize=15)
        ## draw roads
        for road in self.roads:
            x0, y0 = self.get_vertex_location(road[0])
            x1, y1 = self.get_vertex_location(road[1])
            #vertical road
            if x0 == x1:
                # ax.add_patch(patches.Rectangle((x0 - 0.05, (y0 + y1)/2 + 0.05), 0.1, 0.9,
                #                                facecolor = "white"))
                ax.add_patch(patches.Rectangle((x0 - 0.05, min(y0,y1) + 0.05), 0.1, 0.9,
                                               facecolor = color_map(self.roads[road]), ec = "black"))
            #horizontal road
            elif y0 == y1:
                # ax.add_patch(patches.Rectangle(((x0 + x1)/2 + 0.05, y0 - 0.05), 0.9, 0.1,
                #                                 facecolor = "white"))
                ax.add_patch(patches.Rectangle((min(x0,x1) + 0.05, y0 - 0.05), 0.9, 0.1,
                                                facecolor = color_map(self.roads[road]), ec = "black"))
        for vertex in self.settlements:
            x, y = self.get_vertex_location(vertex)
            ax.add_patch(patches.Rectangle((x-0.1, y-0.1),0.2,0.2,
                                           facecolor=color_map(self.settlements[vertex]), ec = "black"))
            ax.text(x-0.1, y-0.09, "1", fontsize=15, color="white")
        for vertex in self.cities:
            x, y = self.get_vertex_location(vertex)
            ax.add_patch(patches.Rectangle((x-0.1, y-0.1),0.2,0.2,
                                           facecolor=color_map(self.cities[vertex]), ec = "black"))
            ax.text(x-0.1, y-0.09, "2", fontsize=15, color="white")



class Player:
    def __init__(self, player_name, action, dumpPolicy, planBoard, resources=np.array([START_RESOURCES, START_RESOURCES, START_RESOURCES]), points = 0):
        self.name = player_name
        self.action = action
        self.dumpPolicy = dumpPolicy
        self.planBoard = planBoard
        self.init_resources = resources
        self.init_points = points

    def join_board(self, board):
        self.board = board
        self.preComp = self.planBoard(self.board)
        self.player_id = self.board.register_player()
        self.resources = self.init_resources[:]
        self.points = self.init_points

    def get_settlements(self):
        if not self.player_id:
            raise CatanException("Must join a board.")
        return self.board.get_player_settlements(self.player_id)

    def get_cities(self):
        if not self.player_id:
            raise CatanException("Must join a board.")
        return self.board.get_player_cities(self.player_id)

    def get_roads(self):
        if not self.player_id:
            raise CatanException("Must join a board.")
        return self.board.get_player_roads(self.player_id)


    def if_can_buy(self, item):
        if item == "card":
            return np.all(self.resources >= costs[CARD,:])
        elif item == "settlement":
            return np.all(self.resources >= costs[SETTLEMENT,:])
        elif item == "city":
            return np.all(self.resources >= costs[CITY,:])
        elif item == "road":
            return np.all(self.resources >= costs[ROAD,:])
        else:
            raise CatanException("Unknown item: {0}".format(item))

    def buy(self, item, x=-1,y=-1):
        if not self.if_can_buy(item):
            raise CatanException("Cannot afford item. Please check with if_can_buy")
        if item == "card":
            self.points += 1
            self.resources = np.subtract(self.resources,costs[1])
        elif item == "road": #input should be of format board.buy("road", (1,1), (1,2))
            v0 = self.board.get_vertex_number(x[0], x[1])
            v1 = self.board.get_vertex_number(y[0], y[1])
            if self.board.if_can_build_road(v0, v1, self.player_id):
                self.board.build_road(x, y, self.player_id)
                self.resources = np.subtract(self.resources, costs[ROAD,:])
            else:
                raise CatanException("Cannot build road here. Please check with if_can_build")
        elif (item == "settlement" or item == "city"):
            if self.board.if_can_build(item,x,y,self.player_id):
                self.board.build(x,y,item,self.player_id)
                if item == "settlement":
                    self.points += 1
                    self.resources = np.subtract(self.resources,costs[SETTLEMENT,:])
                else:
                    self.points += 1
                    self.resources = np.subtract(self.resources,costs[CITY,:])
            else:
                raise CatanException("Cannot build {} here. Please check with if_can_build".format(item))
        else:
            raise CatanException("Item {} is not recognized. Check your spelling".format(item))

    #Trading
    def trade(self, r_in, r_out):
        required = 4
        ports = []
        for e in self.get_settlements():
            if self.board.is_port(e):
                ports.append(self.board.which_port(e))
        for e in self.get_cities():
            if self.board.is_port(e):
                ports.append(self.board.which_port(e))
        if r_in in ports:
            required = 2
        if 3 in ports:
            required = min(required, 3)
        if self.resources[r_in] < required:
            raise CatanException("Invalid trade.")
        self.resources[r_in] -= required
        self.resources[r_out] += 1

class Game:
    def __init__(self, board, players):
        self.board = board
        self.players = players
        for player in self.players:
            player.join_board(self.board)
        self.turn_counter = 0
        self.order = list(range(len(self.players)))

    def initial_build(self):
        for player_i in self.order:
            player = self.players[player_i]
            try:
                player.action(player)
            except CatanException as e:
                print(e)
                raise CatanException("Exception thrown by {}'s action.".format(player.name))

    def play_round(self):
        dice_rolls = []
        for player_i in self.order:
            player = self.players[player_i]
            dice_roll = np.random.randint(1,7) + np.random.randint(1,7)
            dice_rolls.append(dice_roll)
            if dice_roll == 7:
                for d_player in self.players:
                    if sum(d_player.resources) > ROBBER_MAX_RESOURCES:
                        dumped_resources = d_player.dumpPolicy(d_player, ROBBER_MAX_RESOURCES)
                        if any([r < 0 for r in dumped_resources]):
                            raise CatanException("Invalid dump policy. {} tried to dump a negative number of resources.".format(d_player.name))
                        d_player.resources -= dumped_resources
                        if sum(d_player.resources) > ROBBER_MAX_RESOURCES:
                            raise CatanException("Invalid dump policy. {} did not dump enough resources.".format(d_player.name))
            # collect resources
            else:
                collected_resources = self.board.get_resources(player.player_id)[dice_roll-2,:]
                player.resources = np.add(player.resources,collected_resources)
            # perform action
            try:
                player.action(player)
            except CatanException as e:
                print(e)
                raise CatanException("Exception thrown by {}'s action.".format(player.name))
        # update the turn counter
        self.turn_counter += 1
        return dice_rolls

    def check_win(self):
        for player_i in self.order:
            player = self.players[player_i]
            if player.points >= MAX_POINTS:
                return player.name, self.turn_counter
        return False

    # returns winning_player, turn_counter
    def run_game_to_completion(self):
        # shuffles players to randomize turn order
        while self.turn_counter < 1000000:
            self.initial_build()
            self.play_round()
            is_won = self.check_win()
            if is_won:
                return is_won
        raise CatanException("possible infinite loop (over 1M turns)")

    def restart_game(self):
        self.turn_counter = 0
        self.board.clear_board()
        for player in self.players:
            player.join_board(self.board)
        np.random.shuffle(self.order)

    def simulate_game(self, num_trials):
        """Simulates 'num_trials' games with policy 'action' and
        returns average length of won games and win rate for each player"""
        results = {}
        for player in self.players:
            results[player.name] = [0, 0]
        for _ in range(num_trials):
            self.restart_game()
            winner, turns = self.run_game_to_completion()
            results[winner][0] += turns
            results[winner][1] += 1
        for player in self.players:
            results[player.name][0] = results[player.name][0] / results[player.name][1] if results[player.name][1] else 0
            results[player.name][1] /= float(num_trials)
        return results

    def simulate_one_game_with_data(self):
        self.restart_game()

        settlements = []
        cities = []
        roads = []
        hands = []
        live_points = []
        dice_rolls = []

        self.initial_build()
        while self.turn_counter < 1000000:
            round_dice_rolls = self.play_round()
            dice_rolls.append(round_dice_rolls)
            settlements.append(dict(self.board.settlements))
            cities.append(dict(self.board.cities))
            roads.append(dict(self.board.roads))
            hands.append([player.resources[:] for player in self.players])
            live_points.append([player.points for player in self.players])
            if self.check_win():
                return settlements, cities, roads, hands, live_points, dice_rolls
        raise CatanException("possible infinite loop (over 1M turns)")

def simulate_1p_game(action, dumpPolicy, planBoard, board, num_trials):
    player = Player("Player 1", action, dumpPolicy, planBoard)
    game = Game(board, [player])
    return game.simulate_game(num_trials)[player.name][0]

def simulate_1p_game_with_data(action, dumpPolicy, planBoard, board):
    """Simulates 1 game with policy 'action' and returns data about game"""
    player = Player("Player 1", action, dumpPolicy, planBoard)
    game = Game(board, [player])
    settlements, cities, roads, hands, live_points, dice_rolls = game.simulate_one_game_with_data()
    hands = [h[0] for h in hands]
    live_points = [l[0] for l in live_points]
    return settlements, cities, roads, hands, live_points, dice_rolls

def get_random_dice_arrangement(width, height):
    """returns a random field of dice"""
    ns = (list(range(2, 7)) + list(range(8, 13))) * int(width * height / 10 + 1)
    np.random.shuffle(ns)
    ns = ns[:width*height]
    ns[np.random.choice(np.arange(width*height))] = 7
    ns = np.reshape(ns, (height, width))
    return ns
