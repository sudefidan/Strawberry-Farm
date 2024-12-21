import mesa
import random
from collections import deque
import math
import numpy as np

# Payload States
PAYLOAD_EMPTY = 0
PAYLOAD_NOT_FULL = 1
PAYLOAD_FULL = 2

# Battery States
LOW_BATTERY = 0
CHARGING = 1
ENOUGH_BATTERY = 2

# Tree States
READY_STRAWBERRY = 1
NOT_READY_STRAWBERRY = 0

# Work States
FREE = 0
BUSY = 1
SIGNALED = 2

# Constants
BATTERY_CHARGING_RATE = 10 # Battery charging rate of the robots
MAX_BATTERY = 100 # Maximum battery level of the robots
LOW_BATTERY_LEVEL = 25 # Low battery level of the robots
BATTERY_CONSUMPTION_RATE = 0.25 # Battery consumption rate of the robots
LOW_BATTERY_CONSUMPTION_RATE = 0.1 # Low battery consumption rate of the robots for waiting and while going to the charging station
RIVER_CROSSING_BATTERY_CONSUMPTION_RATE = 0.5 # Battery consumption rate of the robots while crossing the river
RIVER_CROSSING_DELAY = 5 # Delay to slow down crossing the river
MAX_PAYLOAD = 10 # Maximum number of strawberries a picker robot can carry
MAX_TOKEN = 200 # Maximum number of tokens for picker robots
LEARNING_RATE = 0.1 # Learning rate for the Q-learning algorithm
DISCOUNT_FACTOR = 0.9 # Discount factor for the Q-learning algorithm
EXPLORATION_RATE = 0.9 # Exploration rate for Q-learning
MIN_EXPLORATION_RATE = 0.1 # Minimum exploration rate for Q-learning
EXPLORATION_DECAY_RATE = 0.995 # Decay rate for exploration in Q-learning

class FarmRobot(mesa.Agent):
    """Base class for farm agents with common functionalities."""
    def __init__(self, id, pos, model):
        super().__init__(id, model)
        self.mode = self.model.mode
        self.id = id # ID of the robot
        self.x, self.y = pos # Position of the robot
        self.next_x, self.next_y = None, None # Next position of the robot
        self.battery = MAX_BATTERY  # Battery level of the robot
        self.battery_state = ENOUGH_BATTERY # Battery state of the robot
        self.crossing_river = False  # Flag to indicate if the robot is crossing a river
        self.crossing_direction = (0, 0)  # Direction in which the robot is crossing the river

        if self.mode == "Novel":
            self.initialise_q_table() # Initialise the Q-table for the robots
            self.learning_rate = LEARNING_RATE  # How much new information overrides old information
            self.discount_factor = DISCOUNT_FACTOR  # Balance the trade-off between short-term and long-term rewards.
            self.exploration_rate = EXPLORATION_RATE # Probability of choosing a random action vs. the best action
            self.min_exploration_rate = MIN_EXPLORATION_RATE  # Minimum exploration rate
            self.exploration_decay_rate = EXPLORATION_DECAY_RATE  # Decay rate for exploration

    @property
    def is_charging(self):
        return self.battery_state == CHARGING

    @property
    def is_low_battery(self):
        return self.battery_state == LOW_BATTERY

    @property
    def is_enough_battery(self):
        return self.battery_state == ENOUGH_BATTERY

    @property
    def is_Free(self):
        return self.work_state == FREE

    @property
    def is_Busy(self):
        return self.work_state == BUSY

    @property
    def is_Signaled(self):
        return self.work_state == SIGNALED

    def step(self):
        self.update_battery_state()
        action = self.deliberate()
        if action:
            getattr(self, action)()
        self.advance()

    def advance(self):
        """Advance the robot to the next position."""
        if self.next_x is not None and self.next_y is not None:
            if self.x != self.next_x or self.y != self.next_y: # Check if the robot is moving
                self.model.update_distance_traveled(type(self),1) # Update the distance traveled by the robot
            self.model.grid.move_agent(self, (self.next_x, self.next_y))
            self.x, self.y = self.next_x, self.next_y
            self.next_x, self.next_y = None, None

    def consume_battery(self, rate):
        self.battery -= rate
        self.model.update_battery_consumed(type(self), rate) # Update the battery consumed by the robot

    def update_battery_state(self):
        """Update the battery state of the robot."""
        # Check if the robot is at a reserved charging station
        at_reserved_charging_station = any(
            station.x == self.x and station.y == self.y and robot_id == self.id
            for station, robot_id in self.model.reserved_charging_stations
        )

        max_battery = MAX_BATTERY if not isinstance(self, ChargerRobot) else 4*MAX_BATTERY
        if at_reserved_charging_station and self.battery < max_battery:
            #print(f"Robot {self.id} is at a reserved charging station.")
            self.battery_state = CHARGING
        elif self.battery <= LOW_BATTERY_LEVEL and self.battery_state != CHARGING:
            #print(f"Robot {self.id} has low battery.")
            self.battery_state = LOW_BATTERY
        else:
            #print(f"Robot {self.id} has enough battery.")
            self.battery_state = ENOUGH_BATTERY

    def recharge(self):
        """Recharge the battery of the robot."""
        max_battery = MAX_BATTERY if not isinstance(self, ChargerRobot) else 4*MAX_BATTERY
        if self.is_charging:
            self.battery = min(max_battery, self.battery + BATTERY_CHARGING_RATE)
            if self.battery == max_battery:
                self.battery_state = ENOUGH_BATTERY
                self.release_charging_station()
            else:
                self.wait()
        else:
            self.move_randomly()

    def move_to_charging_station(self):
        # Check if the robot's ID is in the reserved charging stations list
        reserved_station = next((station for station, robot_id in self.model.reserved_charging_stations if robot_id == self.id), None)

        if reserved_station:
            charging_station = reserved_station
        else:
            charging_station = self.find_nearest_point("charging")
            if charging_station and charging_station.is_free():
                self.model.reserved_charging_stations.append((charging_station, self.id))

        if charging_station:
            path = self.bfs((self.x, self.y), (charging_station.x, charging_station.y))
            if path:
                next_position = path[1] if len(path) > 1 else path[0]
                self.next_x, self.next_y = next_position
                self.consume_battery(LOW_BATTERY_CONSUMPTION_RATE)  # Low battery mode
                self.advance()  # Move to the next position
        else:
            self.wait()

    def find_nearest_point(self, point_type):
        """Find the nearest point (charging station or collection point) to the robot."""
        min_distance = float('inf')  # Initialize the minimum distance
        nearest_point = None

        if point_type == 'charging':
            points = self.model.charging_stations
        elif point_type == 'collection':
            points = self.model.collection_points

        for point in points:  # Iterate over all points
            if point_type == 'charging':
                is_free = point.is_free() # Check if the charging station is free
            else:
                is_free = lambda x: True # All collection points are considered free

            if is_free: # Check if the point is free
                path = self.bfs((self.x, self.y), (point.x, point.y))  # Find the shortest path
                distance = len(path)  # Calculate the distance
                if distance < min_distance:  # Update the nearest point if the distance is less
                    min_distance = distance
                    nearest_point = point

        if nearest_point:
            #print(f"Robot {self.id} at {self.x, self.y} found a nearest {point_type} at position {nearest_point.x, nearest_point.y}")
            return nearest_point

    def release_charging_station(self,):
        """Release the reservation on the charging station."""
        for station, robot_id in self.model.reserved_charging_stations:
            if robot_id == self.id:
                self.model.reserved_charging_stations.remove((station, robot_id))
                break

    def is_next_to_tree(self, position):
        """Check if the robot is next to a tree."""
        x, y = position
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Possible directions to check
        for dx, dy in directions:
            neighbor_position = (x + dx, y + dy)
            if not self.model.grid.out_of_bounds(neighbor_position):
                cell_contents = self.model.grid.get_cell_list_contents(neighbor_position)
                if any(isinstance(agent, Tree) for agent in cell_contents):
                    return True
        return False

    def move_randomly(self):
        """Move the robot to a random adjacent position."""
        if self.crossing_river:
            self.cross_river()
            #print(f"Robot {self.id} is crossing the river. Reward: -0.5")
            return -0.5 # Negative reward for crossing the river

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Possible directions to move
        random.shuffle(directions)  # Shuffle to ensure random selection
        for dx, dy in directions:
            new_position = (self.x + dx, self.y + dy)
            if not self.model.grid.out_of_bounds(new_position):
                cell_contents = self.model.grid.get_cell_list_contents(new_position)
                if self.is_cell_empty(cell_contents):
                    if any(isinstance(agent, River) for agent in cell_contents):  # Check if the cell contains a river
                        if (dx, dy) in [(1, 0), (-1, 0)]:  # Restrict movement to left or right
                            self.crossing_river = True
                            self.crossing_direction = (dx, dy)
                            self.next_x, self.next_y = new_position
                            self.advance()
                            #print(f"Robot {self.id} is crossing the river. Reward: -0.5")
                            return -0.5  # Negative reward for crossing the river
                    else:
                        self.consume_battery(BATTERY_CONSUMPTION_RATE)  # Consume battery
                        self.crossing_river = False
                        self.next_x, self.next_y = new_position
                        self.advance()
                        if isinstance(self, PickerRobot) and self.is_next_to_tree(new_position):
                            #print(f"Robot {self.id} moved next to a tree. Reward: 1")
                            return 1  # Positive reward for moving next to a tree
                        if isinstance(self, ExplorerDrone) and any(isinstance(agent, Tree) for agent in cell_contents):
                            #print(f"Drone {self.id} moved on the top of a tree. Reward: 1")
                            return 1  # Positive reward for moving towards a tree
                        #print(f"Robot {self.id} moved to an empty cell. Reward: -1")
                        return -1  # Negative reward for moving into an empty cell
        #print(f"Robot {self.id} could not find a valid move. Reward: -1")
        return -1  # Negative reward if no valid move is found


    def wait(self):
        """Wait in the same position."""
        self.consume_battery(LOW_BATTERY_CONSUMPTION_RATE)  # Consume battery
        self.next_x, self.next_y = self.x, self.y  # Stay in the same position

    def cross_river(self):
        """Continue moving in the same direction until the robot exits the river."""
        self.consume_battery(RIVER_CROSSING_BATTERY_CONSUMPTION_RATE if isinstance(self, PickerRobot) else BATTERY_CONSUMPTION_RATE) # Consume battery
        if hasattr(self, 'crossing_delay') and self.crossing_delay > 0:
            self.crossing_delay -= 1
            return
        dx, dy = self.crossing_direction
        new_position = (self.x + dx, self.y + dy)  # Calculate the new position
        if not self.model.grid.out_of_bounds(new_position):  # Check if the new position is out of bounds
            cell_contents = self.model.grid.get_cell_list_contents(new_position)  # Get the contents of the new position
            if all(not isinstance(agent, (Tree, PickerRobot)) for agent in cell_contents):  # Check if the cell is empty
                self.next_x, self.next_y = new_position  # Move to the new position
                if hasattr(self, 'crossing_delay'):
                    self.crossing_delay = RIVER_CROSSING_DELAY  # Delay movement to slow down crossing the river, every RIVER_CROSSING_DELAY steps it goes ahead
                if not any(isinstance(agent, River) for agent in cell_contents):  # Check if the cell contains a river
                    self.crossing_river = False  # Exit the river
                return
        self.crossing_river = False  # Exit the river if out of bounds or blocked

    def is_in_the_house(self):
        """Check if the robot is in the house."""
        cell_contents = self.model.grid.get_cell_list_contents((self.x, self.y))
        return any(isinstance(agent, House) for agent in cell_contents)


    def bfs(self, start, goal, within_range=False):
        """Perform BFS to find the shortest path from start to goal.
        If within_range is True, find the closest position within 3 cells of the goal."""
        queue = deque([start]) # Initialise the queue
        came_from = {start: None} # Store the path, key is the current position and value is the previous position
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Up, down, right, left

        while queue:
            current = queue.popleft() # Get the current position

            # If within_range is True, check if we are within 3 cells of the goal
            if within_range:
                for dx, dy in directions:
                    for dist in range(1, self.max_distance + 1): # Check within 3 cells
                        if (current[0] + dx * dist, current[1] + dy * dist) == goal: # Check if the goal is within range
                            #print(f"Path found within range from {start} to {goal}.")
                            return self.reconstruct_path(came_from, start, current)

            # If within_range is False, check if we have reached the exact goal
            if not within_range and current == goal:
                #print(f"Exact path found from {start} to {goal}.")
                return self.reconstruct_path(came_from, start, goal)

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy) # Get the neighbor position
                if neighbor not in came_from and not self.model.grid.out_of_bounds(neighbor): # Check if the neighbor is not visited and within bounds
                    if isinstance(self, PickerRobot) and not any(isinstance(agent, Tree) for agent in self.model.grid.get_cell_list_contents(neighbor)): # Skip if the neighbor is not a tree
                        queue.append(neighbor)
                        came_from[neighbor] = current
                    if isinstance(self, (ExplorerDrone, ChargerRobot)):
                        queue.append(neighbor)
                        came_from[neighbor] = current

        return None  # No path found

    def reconstruct_path(self, came_from, start, goal):
        """Reconstruct the path from start to goal."""
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        #print("Path: ", path)
        return path


    def generate_unique_id(self):
        """Generate a unique ID that is not already taken."""
        existing_ids = {agent.unique_id for agent in self.model.schedule.agents}
        new_id = self.model.next_id()
        while new_id in existing_ids:
            new_id = self.model.next_id()
        return new_id

    def initialise_q_table(self):
        state_space_size = self.get_state_space_size()
        action_space_size = self.get_action_space_size()
        self.q_table = np.zeros((state_space_size, action_space_size))
        #print(f"Initialised Q-table with size: {self.q_table.shape}")

    def get_state_space_size(self):
        # Define the size of the state space, which is the total number of cells in the grid
        return self.model.width * self.model.height

    def get_action_space_size(self):
        # Define the size of the action space, which is the total number of possible actions: up, down, right, left = 4
        return 4

    def get_state(self):
        # Define the state representation: (x, y) position of the robot
        state = self.pos[0] * self.model.width + self.pos[1]
        #print(f"State: {state}")
        return state

    def get_action(self):
        # epsilon-greedy action selection
        state = self.get_state()
        if state >= self.get_state_space_size():
            return np.random.randint(self.get_action_space_size())  # Explore if state is out of bounds
        if np.random.rand() < self.exploration_rate: # Choose random action with probability exploration_rate
            return np.random.randint(self.get_action_space_size())  # Explore
        return np.argmax(self.q_table[state])  # Exploit best action based on Q-table

    def update_q_table(self, state, action, reward, next_state):
        if state >= self.get_state_space_size() or next_state >= self.get_state_space_size():
            #print(f"State {state} or next state {next_state} is out of bounds for Q-table with size {self.get_state_space_size()}")
            return  # Skip update if state is out of bounds
        best_next_action = np.argmax(self.q_table[next_state])
        #print(f"Best next action: {best_next_action}")
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        #print(f"Updated Q-table: {self.q_table}")

    def update_rewards(self):
        if self.mode == "Novel":
            state = self.get_state()
            action = self.get_action()
            reward = self.move_randomly()
            next_state = self.get_state()
            self.update_q_table(state, action, reward, next_state)
            #print(f"Reward: {reward}")

            # Decay exploration_rate
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay_rate)
            #print(f"exploration_rate: {self.exploration_rate})

    def signal_charger_robot(self):
        if self not in self.model.robots_need_charging:
            for agent in self.model.schedule.agents:
                if isinstance(agent, ChargerRobot) and agent.battery_state == ENOUGH_BATTERY and agent.is_Free:
                    self.model.robots_need_charging.append(self)
                    #print(f"Picker {self.id} signaled charger robot {agent.id}.")
                    agent.signaled_by = self
                    agent.work_state = SIGNALED
                    break
                #if isinstance(agent, ChargerRobot) and agent.battery_state == ENOUGH_BATTERY and not agent.is_Free:
                    #print(f"Charger robot {agent.id} is busy and cannot be signaled.")
        #else:
            #print(f"Picker {self.id} already signaled to a picker.")

class PickerRobot(FarmRobot):
    """Represents the picker robot of the farm."""
    def __init__(self, id, pos, model):
        super().__init__(id,pos, model)
        self.payload = []  # id of the strawberry that the robot carries
        self.payload_state = PAYLOAD_EMPTY  # State of the payload
        self.max_distance = 3  # It can reach max 3 strawberry trees away
        self.crossing_delay = 0  # Counter to slow down movement when crossing the river
        self.target_tree = None  # Target tree to pick strawberries from
        self.target_tree_pos = None  # Position of the target tree
        self.work_state = FREE # State of the robot for the auction

        self.visible = False if self.mode == "Extended" else True   # Visibility of the robot
        if self.mode in ["Novel", "Extended"]:
            self.tokens = MAX_TOKEN # Token for the robot

    @property
    def is_Payload_Empty(self):
        return self.payload_state == PAYLOAD_EMPTY

    @property
    def is_Payload_Not_Full(self):
        return self.payload_state == PAYLOAD_NOT_FULL

    @property
    def is_Payload_Full(self):
        return self.payload_state == PAYLOAD_FULL

    def deliberate(self):
        #print(f"Picker {self.id} payload state: ", self.payload_state, "Robot's battery state: ", self.battery_state, "Robot's battery: ", self.battery, "Robot's position: ", self.x, self.y)
        if self.is_low_battery:  # Return to the charging station if the battery is low
            if self.mode == "Novel":
                self.signal_charger_robot() # Ensure battery never goes to 0
                return "wait"
            else:
                if self.battery <= 0:
                    return "wait" # Stay in the same position if the battery is empty
                #print("Battery is low, move to battery station")
                return "move_to_charging_station"
        elif self.is_charging:  # Charge the battery until full
            #print("Battery is charging")
            return "recharge"
        elif self.is_Payload_Full: # Drop off the strawberries if the payload is full
            #print("Payload is full, drop off")
            return "drop_off"
        elif self.is_enough_battery and self.is_Signaled: # If the robot is signaled by the drone
            return "move_to_target"
        elif self.is_Payload_Not_Full and self.model.check_all_collected():
            #print("All strawberries are collected, needs to drop off")
            return "drop_off"
        elif self.is_enough_battery and self.is_Free: # If the battery is enough
            if self.mode == "Extended" and self.is_in_the_house():
                return "wait"  # Stay in the house and wait, not visible
            if(self.is_Payload_Not_Full or self.is_Payload_Empty) and not self.model.check_all_collected():
                if self.check_surroundings():
                    #print("Found a strawberry, try to pick")
                    # Check if another robot with a smaller ID is also trying to pick strawberries
                    cell_contents = self.model.grid.get_cell_list_contents((self.x, self.y))
                    for agent in cell_contents:
                        if isinstance(agent, PickerRobot) and agent.id < self.id: # Give priority to the robot with the smaller ID
                            #print(f"Robot {self.id} found another robot with a smaller ID trying to pick strawberries.")
                            if self.mode == "Novel":
                                self.update_rewards()
                            return "move_randomly"
                    #print("No other robot found, try to pick")
                    return "pick"
                else:
                    if self.mode == "Novel":
                        self.update_rewards()
                    #print("No strawberry found, move randomly")
                    return "move_randomly"

    def bid(self):
        if self.tokens <= 0:
            #print(f"Robot {self.id} has no tokens to bid.")
            bid = 0
        elif self.is_charging:
            #print(f"Robot {self.id} is charging and cannot bid.")
            bid = 0
        elif self.is_Payload_Full:
            #print(f"Robot {self.id} has a full payload and cannot bid.")
            bid = 0
        elif self.is_Busy:
            #print(f"Robot {self.id} is busy and cannot bid.")
            bid = 0
        elif self.is_Signaled:
            #print(f"Robot {self.id} is already signaled and cannot bid.")
            bid = 0
        else:
            battery_factor = math.ceil((self.battery / MAX_BATTERY)) # Bidding factor based on the battery level
            path = self.bfs((self.x, self.y), self.target_tree_pos, within_range=True)
            distance_factor = int(len(path)) if path else 500 # Bidding factor based on the distance to the target tree
            bid = math.ceil((battery_factor / distance_factor)) # Bid based on the factors
        return bid

    def pay(self, payment):
        self.tokens -= payment

    def receive_signal(self, tree_pos):
        self.target_tree = self.model.grid.get_cell_list_contents(tree_pos)[0]
        self.target_tree_pos = tree_pos
        self.work_state = SIGNALED
        #print(f"Robot {self.id} received a signal from the drone for tree at {tree_pos}.")

    def move_to_target(self):
        if self.crossing_river:
            self.cross_river()
            return

        if self.target_tree and self.target_tree_pos:
            path = self.bfs((self.x, self.y), self.target_tree_pos, within_range=True)
            #print(f"Path to the target tree: {path}")
            if path:
                # Check if the target tree still exists and is ready for picking
                cell_contents = self.model.grid.get_cell_list_contents(self.target_tree_pos)
                if self.target_tree not in cell_contents or not self.target_tree.is_strawberry_ready:
                    #print(f"Robot {self.id} found that the target tree is no longer available.")
                    self.target_tree = None
                    self.target_tree_pos = None
                    self.work_state = FREE
                    return

                if (self.x, self.y) == path[-1]: # If the robot reached the target tree
                    #print(f"Robot {self.id} reached the target tree.")
                    self.pick() # Pick the strawberry
                else:
                    next_position = path[1] if len(path) > 1 else path[0]  # Move to the next position
                    self.next_x, self.next_y = next_position
                    cell_contents = self.model.grid.get_cell_list_contents(next_position)
                    if any(isinstance(agent, River) for agent in cell_contents):
                        self.crossing_river = True
                        self.crossing_direction = (self.next_x - self.x, self.next_y - self.y)
                        self.advance()
                    else:
                        self.consume_battery(BATTERY_CONSUMPTION_RATE)  # Consume battery
                        self.crossing_river = False
                        self.advance()
            #else:
                #print(f"Robot {self.id} could not find a path to the target tree.")
        #else:
            #print(f"Robot {self.id} has no target tree to move to.")

    def is_cell_empty(self, cell_contents):
        """Check if the cell is empty. It does not check for Explorer drones, drones fly over the robots."""
        if self.mode in ["Novel", "Extended"]:
            return all(not isinstance(agent, (Tree, PickerRobot, House, ChargingStation, CollectionPoint)) for agent in cell_contents)
        else:
            return all(not isinstance(agent, (Tree, PickerRobot, ChargingStation, CollectionPoint)) for agent in cell_contents)


    def check_surroundings(self):
        """Check for strawberries in the next 3 trees around the robot."""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Up, down, right, left
        for dx, dy in directions:
            for dist in range(1, self.max_distance + 1):
                tree_x, tree_y = self.x + dx * dist, self.y + dy * dist
                if self.model.grid.out_of_bounds((tree_x, tree_y)):
                    continue  # Skip positions out of bounds
                cell_contents = self.model.grid.get_cell_list_contents((tree_x, tree_y))
                for agent in cell_contents:
                    if isinstance(agent, Tree) and agent.is_strawberry_ready:
                        self.target_tree = agent
                        self.target_tree_pos = (tree_x, tree_y)
                        return True
        return False

    def pick(self):
        """Pick a strawberry from the target tree."""
        if self.target_tree and self.target_tree_pos:
            self.work_state = BUSY
            self.target_tree.state = NOT_READY_STRAWBERRY  # Pick the strawberry
            self.payload.append(self.target_tree.id)  # Add the strawberry to the payload
            self.model.update_collected_strawberries(1)  # Update the collected strawberries in the model
            # Update the payload state
            if len(self.payload) >= MAX_PAYLOAD:
                self.payload_state = PAYLOAD_FULL
            else:
                self.payload_state = PAYLOAD_NOT_FULL
            #print(f"Robot {self.id} picked a strawberry from tree {self.target_tree.id}.")
            # Remove the robt, tree pair from the assigned list
            if (self.id, self.target_tree_pos) in self.model.assigned_trees:
                self.model.assigned_trees.remove((self.id, self.target_tree_pos))
            # Replace the tree with a normal tree
            self.model.grid.remove_agent(self.target_tree)
            new_tree_id = self.generate_unique_id()
            new_tree = Tree(new_tree_id, self.target_tree_pos, self.model, 0)
            self.model.grid.place_agent(new_tree, self.target_tree_pos)
            self.model.schedule.add(new_tree)
            self.target_tree = None
            self.target_tree_pos = None
            self.work_state = FREE

    def drop_off(self):
        """Drop off the picked strawberries at the collection point."""
        if self.crossing_river:
            self.cross_river()
            return

        collection_point = self.find_nearest_point("collection")
        self.work_state = BUSY

        if self.x != collection_point.x or self.y != collection_point.y:
            path = self.bfs((self.x, self.y), (collection_point.x, collection_point.y))
            if path:
                # Move to the first step in the path toward the collection point
                next_position = path[1] if len(path) > 1 else path[0]
                #print("Path: ", path)
                #print("Next position: ", next_position)
                self.next_x, self.next_y = next_position
                cell_contents = self.model.grid.get_cell_list_contents(next_position)
                if any(isinstance(agent, River) for agent in cell_contents):
                        self.crossing_river = True
                        self.crossing_direction = (self.next_x - self.x, self.next_y - self.y)
                        self.advance()
                else:
                    self.consume_battery(BATTERY_CONSUMPTION_RATE)  # Consume battery
                    self.crossing_river = False
                    self.advance()
            #else:
                #print(f"Robot {self.id} could not find a path to the collection point.")
        else:
            # Robot is at the collection point
            self.model.update_dropped_off_strawberries(len(self.payload))  # Update the collected and dropped off strawberries in the model
            self.payload = []  # Empty the payload
            self.payload_state = PAYLOAD_EMPTY  # Update the payload state
            self.work_state = FREE
            #print("Payload dropped off at collection point.")

class ExplorerDrone(FarmRobot):
    """Represents the explorer drone of the farm."""
    def __init__(self, id, pos, model):
        super().__init__(id,pos, model)

    def deliberate(self):
        #print(f"Drone {self.id}, battery: {self.battery}, battery state: {self.battery_state})")
        if self.is_low_battery:  # Return to the charging station if the battery is low
            if self.mode == "Novel":
                self.signal_charger_robot() # Ensure battery never goes to 0
                return "wait"
            else:
                if self.battery <= 0:
                    return "wait" # Stay in the same position if the battery is empty
                #print("Battery is low, move to battery station")
                return "move_to_charging_station"
        elif self.is_charging:  # Charge the battery until full
            return "recharge"
        elif self.is_enough_battery: # If the battery is enough
            if self.mode in ["Novel","Extended"]:
                if self.check_strawberry():
                    return "send_signal"
                else:
                    if self.mode == "Novel":
                        self.update_rewards()
                    return "move_randomly"
            else: # Basic Mode
                if self.check_strawberry(): # Check if there is a strawberry underneath
                    return "wait"
                else:
                    return "move_randomly"

    def check_strawberry(self):
        """Check if there is a strawberry underneath and wait if there is."""
        cell_contents = self.model.grid.get_cell_list_contents((self.x, self.y))
        for agent in cell_contents:
            if isinstance(agent, Tree) and agent.is_strawberry_ready: # Check if the tree has strawberries
                if any(isinstance(agent, ExplorerDrone) and agent.id != self.id for agent in cell_contents): # Check if another drone is already waiting
                    #print(f"Drone {self.id} found a strawberry but another drone is already waiting.")
                    return False
                return True
        #print(f"No agent found at position ({self.x}, {self.y})")
        return False

    def is_cell_empty(self, cell_contents):
        """Check if the cell is empty. Does not check for Picker robots, it flies over them."""
        if self.mode in ["Novel", "Extended"]:
            return all(not isinstance(agent, (House, ExplorerDrone, ChargingStation, CollectionPoint)) for agent in cell_contents)
        else:
            return all(not isinstance(agent, (ExplorerDrone, ChargingStation, CollectionPoint)) for agent in cell_contents)

    def send_signal(self):
        self.wait()
        winner = None
        if self.mode == "Novel": # Get the winner of the auction
            winner = self.model.run_auction((self.x, self.y))
            #print(f"Drone {self.id} triggered the auction for tree at {(self.x, self.y)}.")
        if self.mode == "Extended": # Get the first free robot
            for robot in self.model.picker_robots:
                if robot.is_Free and not any(assigned_tree == (self.x, self.y) for _, assigned_tree in self.model.assigned_trees):
                    winner = robot.id
                    self.model.assigned_trees.append((robot.id, (self.x, self.y)))
                    break

        # Send the signal to the winner
        if winner:
            #print("Winner: ", winner)
            for robot in self.model.picker_robots:
                if robot.id == winner:
                    robot.receive_signal((self.x, self.y))
                    break
        #else:
            #print(f"No robot found to send the signal for tree at {(self.x, self.y)}.")

    def get_signaled_robot_id(self):
        for robot in self.model.picker_robots:
            if robot.is_Signaled and robot.target_tree_pos == (self.x, self.y):
                return robot.id
        return None

class Tree(mesa.Agent):
    """Represents a tree in the farm."""
    def __init__(self, id, pos, model, has_strawberries):
        super().__init__(id, model)
        self.mode = self.model.mode # Mode of the model
        self.id = id # ID of the tree
        self.x, self.y = pos  # Position of the tree
        self.state = READY_STRAWBERRY if has_strawberries else NOT_READY_STRAWBERRY # State of the tree
        if self.mode in ["Extended", "Novel"]:
            self.growth_delay = random.randint(20, 500)  # Random delay to slow down growth

    @property
    def is_strawberry_ready(self): # Check if the tree has strawberries
        return self.state == READY_STRAWBERRY

    @property
    def is_strawberry_not_ready(self): # Check if the tree does not have strawberries
        return self.state == NOT_READY_STRAWBERRY

    def step(self):
        if self.mode in ["Extended", "Novel"]:
            if self.is_strawberry_not_ready: # If the tree does not have strawberries
                self.growth_delay -= 1  # Decrease the step counter
                if self.growth_delay <= 0: # Check if the tree is ready to grow strawberries
                    self.grow() # Grow strawberries

    def grow(self):
        if self.is_strawberry_not_ready: # If the tree does not have strawberries
            self.state = READY_STRAWBERRY
            self.growth_delay = random.randint(20, 500)  # Reset the step counter for the next growth


class River(mesa.Agent):
    """Represents a river in the farm."""
    def __init__(self, id, pos, model):
        super().__init__(id, model)
        self.id = id # ID of the river
        self.x, self.y = pos # Position of the river

class ChargingStation(mesa.Agent):
    """Represents a charging station for the robots."""
    def __init__(self, id, pos, model):
        super().__init__(id, model)
        self.id = id # ID of the charging station
        self.x, self.y = pos # Position of the charging station
        self.reserved_by = None # Robot reserved by the charging station

    def is_free(self):
        """Check if the charging station is free."""
        return not any(station == self for station, _ in self.model.reserved_charging_stations)

class CollectionPoint(mesa.Agent):
    """Represents a collection point for the strawberries."""
    def __init__(self, id, pos, model):
        super().__init__(id, model)
        self.id = id # ID of the collection point
        self.x, self.y = pos # Position of the collection point

class House(mesa.Agent):
    """Represents a house where picker robots start from."""
    def __init__(self, id, pos, model):
        super().__init__(id, model)
        self.id = id # ID of the house station
        self.x, self.y = pos  # Position of the house station

class ChargerRobot(FarmRobot):
    """Represents a charger robot that acts like a mobile power bank."""
    def __init__(self, id, pos, model):
        super().__init__(id, pos, model)
        self.battery = 4 * MAX_BATTERY  # Charger robots have higher battery capacity
        self.work_state = FREE  # Work State of the robot
        self.signaled_by = None  # Robot that signaled this charger robot

    def deliberate(self):
        if self.is_low_battery:  # Return to the charging station if the battery is low
            return "move_to_charging_station"
        elif self.is_charging:  # Charge the battery until full
            return "recharge"
        elif self.is_Signaled:  # Move to the picker robot if battery is enough and signaled
            #print(f"Charger robot {self.id} signaled by picker robot {self.signaled_by.id}.")
            return "move_to_picker"
        else:
            return "wait"

    def move_to_picker(self):
        if self.signaled_by:
            #print(f"Charger robot at {(self.x, self.y)} moving to signaled picker robot {self.signaled_by.x, self.signaled_by.y}.")
            path = self.bfs((self.x, self.y), (self.signaled_by.x, self.signaled_by.y))
            if path:
                next_position = path[1] if len(path) > 1 else path[0]
                self.next_x, self.next_y = next_position
                self.consume_battery(BATTERY_CONSUMPTION_RATE)  # Consume battery
                self.advance()
                if (self.x, self.y) == (self.signaled_by.x, self.signaled_by.y):
                    self.charge(self.signaled_by)

    def charge(self, robot):
        """Charge the battery of the robot."""
        #print(f"Charger robot {self.id} charging robot {picker_robot.id}.")
        self.battery -= MAX_BATTERY - robot.battery
        robot.battery = MAX_BATTERY
        robot.battery_state = ENOUGH_BATTERY
        self.signaled_by = None  # Reset the signaled robot
        self.model.robots_need_charging.remove(robot)
        self.work_state = FREE
        #print(f"Robot {robot.id} battery is full.")


