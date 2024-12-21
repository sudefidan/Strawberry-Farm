import mesa
from random import randint, sample, seed
from farm.agents import *

SEED_NUMBER = 42

class StrawberryFarm(mesa.Model):
    def __init__(self,n_picker, n_explorer, width, height, strawberry_percentage, mode):
        """ Create a Strawberry Farm Model."""
        super().__init__(seed=SEED_NUMBER)
        self.n_picker = n_picker # Number of picker robots
        self.n_explorer = n_explorer # Number of explorer drones
        self.width = width # Width of the grid
        self.height = height # Height of the grid
        self.strawberry_percentage = strawberry_percentage # Percentage of trees with strawberries
        self.mode = mode # Operation mode of the model
        self.farm_tokens = 0 # Farm tokens
        self.assigned_trees = [] # List of trees with strawberries assigned to picker robots
        self.reserved_charging_stations = [] # List of reserved charging stations
        self.robots_need_charging = []  # List of robots sent signal for charging


        # Set the seed for reproducibility
        self.seed = SEED_NUMBER
        random.seed(SEED_NUMBER)
        np.random.seed(SEED_NUMBER)
        self.random.seed(SEED_NUMBER)

        #KPIs
        self.collected_strawberries = 0 # Total number of strawberries collected
        self.dropped_off_strawberries = 0 # Total number of strawberries collected and dropped off
        self.total_distance_traveled_by_picker = 0 # Total distance traveled by all pickers
        self.total_distance_traveled_by_explorer = 0 # Total distance traveled by all explorers
        self.total_battery_consumed_by_picker = 0 # Total battery consumed by all all pickers
        self.total_battery_consumed_by_explorer = 0 # Total battery consumed by all explorers

        # Initialise DataCollector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "strawberries_picked": lambda m: m.collected_strawberries,
                "strawberries_in_basket": lambda m: m.dropped_off_strawberries,
                "total_distance_traveled_by_picker": lambda m: m.total_distance_traveled_by_picker,
                "total_distance_traveled_by_explorer": lambda m: m.total_distance_traveled_by_explorer,
                "total_battery_consumed_by_picker": lambda m: m.total_battery_consumed_by_picker,
                "total_battery_consumed_by_explorer": lambda m: m.total_battery_consumed_by_explorer,
            }
        )

        # Create a grid and schedule
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = mesa.time.SimultaneousActivation(self)

        # Place agents
        agent_id = 0  # Initialise agent ID counter

        # Place rivers and trees
        tree_positions = []
        for x in range(width):
            for y in range(height):
                # Place rivers
                if (x % 15 < 3 and x > 3):  # Place rivers in the first 3 columns of every 15 columns block
                    self.place_river(agent_id, x, y)
                    agent_id += 1
                # Place trees and gaps in empty cells
                elif (x % 5) < 3:  # Place trees in the first 3 columns of every 5 columns block
                    if (x // 5) % 2 == 0 and y >= 2 and self.grid.is_cell_empty((x, y)):  # First set of 3 columns, rows 2 and above if cell is empty
                        tree_positions.append((x, y))
                    elif (x // 5) % 2 != 0 and y < height - 2 and self.grid.is_cell_empty((x, y)):  # Next set of 3 columns, rows below top 2 rows if cell is empty
                        tree_positions.append((x, y))

        #print("Rivers placed")

        # Calculate the number of trees that should have strawberries
        total_trees = len(tree_positions)
        num_strawberry_trees = int(total_trees * (self.strawberry_percentage / 100))
        strawberry_tree_positions = sample(tree_positions, num_strawberry_trees)

        # Place trees with the correct assignment of strawberries
        for pos in tree_positions:
            has_strawberries = pos in strawberry_tree_positions
            self.place_tree(agent_id, pos[0], pos[1], has_strawberries)
            agent_id += 1
        #print("Trees placed")

        # Place collection points at random border positions
        n_collection_points = (self.n_picker + 2) // 3  # Number of collection points
        self.collection_points = []
        self.place_points(n_collection_points, CollectionPoint, self.collection_points, agent_id)
        agent_id += n_collection_points
        #print("Collection points placed")

        # Place charging stations at random border positions
        n_charging_stations = (self.n_explorer + self.n_picker + 2) // 3  # Number of collection points
        self.charging_stations = []
        self.place_points(n_charging_stations, ChargingStation, self.charging_stations, agent_id)
        agent_id += n_charging_stations
        #print("Charging stations placed")

        self.picker_robots = []  # List to store picker robots

        if self.mode in ["Novel","Extended"]:
            # Place house station at a random border position
            #self.house_station = self.place_house(House, agent_id)
            self.house_stations = []
            self.place_points(1, House, self.house_stations, agent_id)
            agent_id += 1
            #print("House station placed")
            # Place picker robots at the house station
            self.place_robot_in_the_house(PickerRobot, self.n_picker, agent_id)
            agent_id += self.n_picker
            #print("Picker robots placed in the house")
        else:
            # Place picker robots at random positions
            self.place_robot_randomly(PickerRobot, self.n_picker, agent_id)
            agent_id += self.n_picker
            #print("Picker robots placed randomly")

        # Place explorer drones at random positions
        self.place_robot_randomly(ExplorerDrone, self.n_explorer, agent_id)
        agent_id += self.n_explorer
        #print("Explorer drones placed")

        # Place charger robots at random positions (only for Novel mode)
        if self.mode == "Novel":
            n_charger = (self.n_picker + 2) // 1  # Number of charger robots
            self.place_robot_in_the_house(ChargerRobot, n_charger, agent_id)
            agent_id += n_charger
        #print("Charger robots placed")

        self.running = True  # Set the model to running
        #print("Model initialised")

    def place_tree(self, agent_id, x, y, has_strawberries):
        tree = Tree(agent_id, (x, y), self, has_strawberries)
        self.grid.place_agent(tree, (x, y))
        self.schedule.add(tree)

    def place_river(self, agent_id, x, y):
        existing_agents = self.grid.get_cell_list_contents((x, y))  # Remove any existing agents before placing the river
        for agent in existing_agents:
            self.grid.remove_agent(agent)
            self.schedule.remove(agent)
        river = River(agent_id, (x, y), self)
        self.grid.place_agent(river, (x, y))
        self.schedule.add(river)

    def place_points(self, n_points, agent_class, agent_list, start_id):
        for _ in range(n_points):
            while True:
                if self.random.choice([True, False]):
                    x = self.random.randrange(self.width)
                    y = self.random.choice([0, self.height - 1])
                else:
                    x = self.random.choice([0, self.width - 1])
                    y = self.random.randrange(self.height)

                cell_contents = self.grid.get_cell_list_contents((x, y))
                if all(not isinstance(agent, (Tree, River, CollectionPoint, ChargingStation)) for agent in cell_contents):
                    agent = agent_class(start_id, (x, y), self)
                    self.grid.place_agent(agent, (x, y))
                    self.schedule.add(agent)
                    agent_list.append(agent)
                    start_id += 1
                    break

    def place_house(self, agent_class, start_id, max_attempts=1000):
        attempts = 0
        while attempts < max_attempts:
            x = random.randrange( self.width - 1)
            y = random.randrange(self.height - 1)
            cell_contents = self.grid.get_cell_list_contents((x, y))
            if all(not isinstance(agent, (Tree, River, CollectionPoint, ChargingStation, House)) for agent in cell_contents):
                agent = agent_class(start_id, (x, y), self)
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)
                return agent
            #else:
                #print(f"Position ({x}, {y}) is occupied by: {[type(agent) for agent in cell_contents]}")
            attempts += 1
        raise Exception("Failed to place house station after maximum attempts")


    def place_robot_randomly(self, agent_class, count, start_id):
        for _ in range(count):
            while True:
                x = self.random.randrange(self.width)
                y = self.random.randrange(self.height)
                if self.grid.is_cell_empty((x, y)):
                    agent = agent_class(start_id, (x, y), self)
                    if agent_class == PickerRobot:
                        self.picker_robots.append(agent)
                    self.grid.place_agent(agent, (x, y))
                    self.schedule.add(agent)
                    start_id += 1
                    break

    def place_robot_in_the_house(self, agent_class, count, start_id):
        for _ in range(count):
            agent = agent_class(start_id, (self.house_stations[0].x, self.house_stations[0].y), self)
            if agent_class == PickerRobot:
                        self.picker_robots.append(agent)
            self.grid.place_agent(agent, (self.house_stations[0].x, self.house_stations[0].y))
            self.schedule.add(agent)
            start_id += 1

    def check_all_collected(self):
        """Check if all strawberries are collected and dropped off."""
        for agent in self.schedule.agents:
            if isinstance(agent, Tree) and agent.is_strawberry_ready:
                return False
        #print("All strawberries collected")
        return True

    def check_all_dropped_off(self):
        for agent in self.schedule.agents:
            if isinstance(agent, PickerRobot) and agent.is_Payload_Empty:
                #print("All strawberries dropped off")
                return True
        #print("Not all strawberries dropped off")
        return False

    def count_trees(self):
        """Count the number of Tree agents in the model."""
        return len([agent for agent in self.schedule.agents if isinstance(agent, Tree)])

    def if_any_battery_level_below_zero(self):
        """Check if all robots have enough battery to return to the house."""
        for agent in self.schedule.agents:
            if isinstance(agent, (PickerRobot, ExplorerDrone)) and agent.battery < 0:
                #print("Not all robots have enough battery")
                return True
        #print("All robots have enough battery")
        return False

    def step(self):
        """Advance the model by one step."""
        self.schedule.step()
        self.datacollector.collect(self)  # Collect data at each step
        #print(f"Step {self.schedule.steps}: Collected {self.collected_strawberries} strawberries, {self.dropped_off_strawberries} in basket")
        if self.check_all_collected() and self.check_all_dropped_off() and self.mode == "Basic":
            self.running = False
        if self.if_any_battery_level_below_zero() and self.mode == "Extended":
            self.running = False


    def run_auction(self, tree_pos):
        """Run the auction for the picker robots for a specific tree."""
        for robot in self.picker_robots:
            if any(assigned_tree == (tree_pos) for _, assigned_tree in self.assigned_trees):
                #print(f"Tree at {tree_pos} is already assigned to robot {robot.id}.")
                return robot.id

        self.bids = []
        # Receive bid from all free picker robots
        for robot in self.picker_robots:
            if robot.is_Free:
                robot_bid = robot.bid()
                #print("Robot's bid:", robot_bid)
                self.bids.append((robot.id, robot_bid))

       # print(f"Bids for tree at {tree_pos}: {self.bids}")
        # Determine the winner
        if self.bids:
            winner = max(self.bids, key=lambda x: x[1])  # Winner is the robot with the highest bid or the first robot if there is a tie
            winner_payment = winner[1]
            winner_id = winner[0]
            # Winner pays the tokens to the model
            for robot in self.picker_robots:
                if robot.id == winner_id:
                    robot.pay(winner_payment)  # Winner pays the amount they bid
                    self.receive_payment(winner_payment)  # Model receives the payment
                    self.assigned_trees.append((winner_id, tree_pos))  # Mark the tree as assigned
                    #print("Winner:", winner_id, "Payment:", winner_payment, "Position:", tree_pos, "Tokens:", self.farm_tokens)
                    return winner_id
        return None

    def receive_payment(self,payment):
        self.farm_tokens += payment

    def update_collected_strawberries(self, collected):
        self.collected_strawberries += collected

    def update_dropped_off_strawberries(self, dropped_off):
        self.dropped_off_strawberries += dropped_off

    def update_distance_traveled(self, agent_type, distance):
        if agent_type == PickerRobot:
            self.total_distance_traveled_by_picker += distance
        elif agent_type == ExplorerDrone:
            self.total_distance_traveled_by_explorer += distance

    def update_battery_consumed(self, agent_type, battery_consumed):
        if agent_type == PickerRobot:
            self.total_battery_consumed_by_picker += battery_consumed
        elif agent_type == ExplorerDrone:
            self.total_battery_consumed_by_explorer += battery_consumed

