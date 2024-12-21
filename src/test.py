import unittest
import random
from farm.model import StrawberryFarm
from farm.agents import *

class TestStrawberryFarm(unittest.TestCase):
    test_counter = 1

    def setUp(self):
        print(f"\n\nRunning Test : {self._testMethodName}")

    def tearDown(self):
        print(f"Test {TestStrawberryFarm.test_counter} completed.")
        TestStrawberryFarm.test_counter += 1

    def test_model_creation(self):
        farm = StrawberryFarm(n_picker=5, n_explorer=5, width=50, height=50, strawberry_percentage=10, mode="Basic")
        # Test if the model is created
        self.assertIsNotNone(farm)
        print("Model created")

        # Test if the number of agents is correct
        self.assertEqual(len([agent for agent in farm.schedule.agents if isinstance(agent, PickerRobot)]), 5)
        self.assertEqual(len([agent for agent in farm.schedule.agents if isinstance(agent, ExplorerDrone)]), 5)
        print(f"Number of Picker Robots: {len([agent for agent in farm.schedule.agents if isinstance(agent, PickerRobot)])}, Number of Explorer Drones: {len([agent for agent in farm.schedule.agents if isinstance(agent, ExplorerDrone)])}")

        # Test if trees are placed
        self.assertGreater(len([agent for agent in farm.schedule.agents if isinstance(agent, Tree)]), 0)

        # Test if rivers are placed
        self.assertGreater(len([agent for agent in farm.schedule.agents if isinstance(agent, River)]), 0)

        # Test if charging stations are placed
        self.assertGreater(len([agent for agent in farm.schedule.agents if isinstance(agent, ChargingStation)]), 0)

        # Test if collection point is placed
        self.assertGreater(len([agent for agent in farm.schedule.agents if isinstance(agent, CollectionPoint)]), 0)

    def test_robots_make_decision(self):
        farm = StrawberryFarm(n_picker=1, n_explorer=1, width=50, height=50, strawberry_percentage=10, mode="Basic")
        picker = next(agent for agent in farm.schedule.agents if isinstance(agent, PickerRobot))
        drone = next(agent for agent in farm.schedule.agents if isinstance(agent, ExplorerDrone))

        # Test when pickers's payload is full and battery is low
        picker.battery_state = LOW_BATTERY
        picker.payload_state = PAYLOAD_NOT_FULL
        action = picker.deliberate()
        self.assertEqual(action, "move_to_charging_station")
        print("Picker's payload is full, battery low =>", action)

        # Test when drone is low on battery
        drone.battery = MAX_BATTERY
        drone.battery_state = ENOUGH_BATTERY
        tree = next(agent for agent in farm.schedule.agents if isinstance(agent, Tree))
        tree.state = READY_STRAWBERRY
        drone.x, drone.y = tree.x, tree.y
        action = drone.deliberate()
        self.assertEqual(action, "wait")
        print("There are strawberries under the drone =>", tree.is_strawberry_ready, "=>", action)

    def test_robot_wait(self):
        farm = StrawberryFarm(n_picker=1, n_explorer=1, width=50, height=50, strawberry_percentage=10, mode="Basic")
        picker = next(agent for agent in farm.schedule.agents if isinstance(agent, PickerRobot))
        drone = next(agent for agent in farm.schedule.agents if isinstance(agent, ExplorerDrone))

        # Test robot waiting
        picker_initial_battery = picker.battery
        picker.wait()
        self.assertEqual((picker.x, picker.y), (picker.next_x, picker.next_y))
        self.assertLess(picker.battery, picker_initial_battery)
        print("Picker waited at position: ", (picker.x, picker.y))

        drone_initial_battery = drone.battery
        drone.wait()
        self.assertEqual((drone.x, drone.y), (drone.next_x, drone.next_y))
        self.assertLess(drone.battery, drone_initial_battery)
        print("Drone waited at position: ", (drone.x, drone.y))

        print(f"Picker's Battery Consumption: -{picker_initial_battery- picker.battery}, Drone's Battery Consumption: -{drone_initial_battery - drone.battery}")

    def test_charger_robot(self):
        farm = StrawberryFarm(n_picker=1, n_explorer=1, width=50, height=50, strawberry_percentage=10, mode="Novel")
        robot = next(agent for agent in farm.schedule.agents if isinstance(agent, PickerRobot))
        charger = next(agent for agent in farm.schedule.agents if isinstance(agent, ChargerRobot))

        # Test robot recharge battery
        robot.x, robot.y = charger.x, charger.y
        charger.work_state = SIGNALED
        farm.robots_need_charging.append(robot)
        robot.battery = 20
        initial_battery = robot.battery
        while robot.battery < MAX_BATTERY:
            charger.charge(robot)
        self.assertGreater(robot.battery, initial_battery)
        self.assertEqual((robot.x, robot.y), (charger.x, charger.y))
        print(f"Robot Energy Loading: %{robot.battery - initial_battery}")

    def test_robot_recharge_battery(self):
        farm = StrawberryFarm(n_picker=1, n_explorer=1, width=50, height=50, strawberry_percentage=10, mode="Basic")
        robot = next(agent for agent in farm.schedule.agents if isinstance(agent, PickerRobot))
        drone = next(agent for agent in farm.schedule.agents if isinstance(agent, ExplorerDrone))
        charging_station = farm.charging_stations[0]

        # Test robot recharge battery
        robot.x, robot.y = charging_station.x, charging_station.y
        robot.battery_state = CHARGING
        robot.battery = 20
        initial_battery = robot.battery
        while robot.battery < MAX_BATTERY:
            robot.recharge()
        self.assertGreater(robot.battery, LOW_BATTERY)
        self.assertEqual((robot.x, robot.y), (charging_station.x, charging_station.y))
        print(f"Picker Energy Loading: %{robot.battery - initial_battery}")

        # Place drone at charging station
        drone.x, drone.y = charging_station.x, charging_station.y
        drone.battery_state = CHARGING
        drone.battery = 20
        initial_battery = drone.battery
        while drone.battery < MAX_BATTERY:
            drone.recharge()
        self.assertGreater(drone.battery, LOW_BATTERY)
        self.assertEqual((drone.x, drone.y), (charging_station.x, charging_station.y))
        print(f"Drone Energy Loading: %{drone.battery - initial_battery}")

    def test_robot_move_randomly(self):
        farm = StrawberryFarm(n_picker=1, n_explorer=1, width=50, height=50, strawberry_percentage=10, mode="Basic")
        picker = next(agent for agent in farm.schedule.agents if isinstance(agent, PickerRobot))
        drone = next(agent for agent in farm.schedule.agents if isinstance(agent, ExplorerDrone))

        # Test picker move around
        picker.x, picker.y = 0, 0
        picker_initial_battery = picker.battery
        picker.move_randomly()
        picker.advance()
        self.assertNotEqual((picker.x, picker.y), (0, 0))
        self.assertLess(picker.battery, picker_initial_battery)
        print("Picker at (0, 0) moved to", (picker.x, picker.y))

        # Test drone move around
        drone.x, drone.y = 0, 0
        drone_initial_battery = drone.battery
        drone.move_randomly()
        drone.advance()
        self.assertNotEqual((drone.x, drone.y), (0, 0))
        self.assertLess(drone.battery, drone_initial_battery)
        print("Drone at (0, 0) moved to", (drone.x, drone.y))

        print(f"Picker's Battery Consumption: -{picker_initial_battery- picker.battery}, Drone's Battery Consumption: -{drone_initial_battery - drone.battery}")

    def test_move_picker_robot_to_target(self):
        farm = StrawberryFarm(n_picker=1, n_explorer=0, width=50, height=50, strawberry_percentage=10, mode="Basic")
        picker = next(agent for agent in farm.schedule.agents if isinstance(agent, PickerRobot))
        tree = next(agent for agent in farm.schedule.agents if isinstance(agent, Tree) and agent.state == READY_STRAWBERRY)
        picker.target_tree = tree
        picker.target_tree_pos = (tree.x, tree.y)

        # Find a free cell for the robot
        free_cell_found = False
        for x in range(farm.width):
            for y in range(farm.height):
                if not farm.grid.get_cell_list_contents((x, y)):
                    picker.x, picker.y = x, y
                    free_cell_found = True
                    break
            if free_cell_found:
                break

        # Ensure a free cell was found
        self.assertTrue(free_cell_found, "No free cell found for the robot")
        print("Position: ", (picker.x, picker.y))
        print("Target tree position: ", picker.target_tree_pos)

        # Test robot move to tree
        initial_battery = picker.battery
        while picker.target_tree:
            picker.move_to_target()

        self.assertEqual( picker.target_tree_pos, None)
        print("Robot reached the target tree: ", "NO" if picker.target_tree else "YES")

        self.assertLess(picker.battery, initial_battery)
        print(f"Battery Consumption: -{initial_battery - picker.battery}")

    def test_picker_robot_check_surroundings(self):
        farm = StrawberryFarm(n_picker=1, n_explorer=0, width=50, height=50, strawberry_percentage=10, mode="Basic")
        robot = next(agent for agent in farm.schedule.agents if isinstance(agent, PickerRobot))
        tree = next(agent for agent in farm.schedule.agents if isinstance(agent, Tree))

        # Test robot check surroundings
        robot.x, robot.y = tree.x, tree.y - 1
        tree.state = READY_STRAWBERRY

        is_there_strawberries = robot.check_surroundings()
        print("There are strawberries around robot =>", is_there_strawberries)
        self.assertTrue(is_there_strawberries)

    def test_picker_robot_pick(self):
        farm = StrawberryFarm(n_picker=1, n_explorer=0, width=50, height=50, strawberry_percentage=10, mode="Basic")
        robot = next(agent for agent in farm.schedule.agents if isinstance(agent, PickerRobot))
        tree = next(agent for agent in farm.schedule.agents if isinstance(agent, Tree))

        # Test robot pick
        robot.x, robot.y = tree.x, tree.y - 1
        tree.state = READY_STRAWBERRY
        robot.target_tree = tree
        robot.target_tree_pos = (tree.x, tree.y)

        print("Robot's payload: ", len(robot.payload))
        print("There are strawberries around robot =>", tree.is_strawberry_ready, "=> Robot is picking strawberries")
        robot.pick()
        print("Robot's payload: ", len(robot.payload))
        self.assertIn(tree.id, robot.payload)
        self.assertFalse(tree.is_strawberry_ready)

    def test_picker_robot_drop_off(self):
        farm = StrawberryFarm(n_picker=1, n_explorer=0, width=50, height=50, strawberry_percentage=10, mode="Basic")
        robot = next(agent for agent in farm.schedule.agents if isinstance(agent, PickerRobot))
        collection_point = next(agent for agent in farm.schedule.agents if isinstance(agent, CollectionPoint))

        # Fill the robot's payload to trigger drop off state
        robot.payload_state = PAYLOAD_FULL
        robot.payload = [1] * MAX_PAYLOAD
        initial_payload = len(robot.payload)

        # Move robot to collection point
        robot.x, robot.y = collection_point.x, collection_point.y

        # Test robot drop off
        robot.drop_off()
        self.assertEqual(robot.payload, [])
        print(f"Robot's initial payload: {initial_payload} => Robot's payload after dropping off: {len(robot.payload)}")

    def test_picker_robot_cross_river(self):
        farm = StrawberryFarm(n_picker=1, n_explorer=0, width=50, height=50, strawberry_percentage=10, mode="Basic")
        robot = next(agent for agent in farm.schedule.agents if isinstance(agent, PickerRobot))
        river = next(agent for agent in farm.schedule.agents if isinstance(agent, River))

        # Test robot crossing river
        robot.x, robot.y = river.x, river.y - 1
        initial_battery = robot.battery
        robot.crossing_river = True
        robot.crossing_direction = (0, 1)
        robot.cross_river()
        self.assertTrue(robot.crossing_river)
        print("Robot crossing river =>", robot.crossing_river)
        self.assertLess(robot.battery, initial_battery)
        print(f"Battery Consumption: -{initial_battery - robot.battery}")

    def test_strawberry_growth(self):
        farm = StrawberryFarm(n_picker=0, n_explorer=0, width=50, height=50, strawberry_percentage=10, mode="Extended")
        tree = next(agent for agent in farm.schedule.agents if isinstance(agent, Tree))
        tree.state = NOT_READY_STRAWBERRY
        self.assertTrue(tree.is_strawberry_not_ready)
        print("Strawberry =>", "READY" if tree.is_strawberry_ready else "NOT READY")

        # Test if strawberries grow in the tree
        tree.growth_delay = 0
        print("Strawberry grew in the tree.")
        tree.step()
        self.assertTrue(tree.is_strawberry_ready)
        print("Strawberry =>", "READY" if tree.is_strawberry_ready else "NOT READY")

    def test_model_step(self):
        # Test the step function
        farm = StrawberryFarm(n_picker=2, n_explorer=2, width=50, height=50, strawberry_percentage=10, mode="Basic")

        # Set all strawberries to collected to test if the model stops running
        for agent in farm.schedule.agents:
            if isinstance(agent, Tree):
                agent.state = NOT_READY_STRAWBERRY
            if isinstance(agent, PickerRobot):
                agent.payload = []

        # Check if any strawberries left
        strawberries_left = False
        for agent in farm.schedule.agents:
            if isinstance(agent, Tree) and agent.is_strawberry_ready:
                strawberries_left = True
                break

        if not strawberries_left:
            print("No strawberries left to collect")


        farm.step()
        print("Model Running =>", farm.running)
        self.assertFalse(farm.running)

        # Reset the model and test if it continues running with strawberries left
        print("Resetting the model...")
        farm = StrawberryFarm(n_picker=2, n_explorer=2, width=50, height=50, strawberry_percentage=10, mode="Basic")
        farm.step()
        print("Model Running =>", farm.running)
        self.assertTrue(farm.running)

    def test_model_with_different_sizes(self):
        # Test with smaller grid size
        farm1 = StrawberryFarm(n_picker=2, n_explorer=2, width=10, height=10, strawberry_percentage=10, mode="Basic")
        print(f"Farm 1 Grid sizes => {farm1.grid.width}x{farm1.grid.height}" )
        self.assertIsNotNone(farm1)
        self.assertEqual(len([agent for agent in farm1.schedule.agents if isinstance(agent, PickerRobot)]), 2)
        self.assertEqual(len([agent for agent in farm1.schedule.agents if isinstance(agent, ExplorerDrone)]), 2)

        # Test with larger grid size
        farm2 = StrawberryFarm(n_picker=2, n_explorer=2, width=100, height=100, strawberry_percentage=10, mode="Basic")
        print(f"Farm 2 Grid sizes => {farm2.grid.width}x{farm2.grid.height}")
        self.assertIsNotNone(farm2)
        self.assertEqual(len([agent for agent in farm2.schedule.agents if isinstance(agent, PickerRobot)]), 2)
        self.assertEqual(len([agent for agent in farm2.schedule.agents if isinstance(agent, ExplorerDrone)]), 2)

        # Test if the grids are different sizes
        self.assertNotEqual(farm1.grid.width, farm2.grid.width)
        self.assertNotEqual(farm1.grid.height, farm2.grid.height)
        if farm1.grid.width != farm2.grid.width and farm1.grid.height != farm2.grid.height:
            print("Grid sizes are different.")


    def test_model_with_different_number_of_robots(self):
        # Test with fewer robots
        farm1 = StrawberryFarm(n_picker=2, n_explorer=2, width=30, height=30, strawberry_percentage=10, mode="Basic")
        print(f"Farm 1 => Number of Picker Robots: {len([agent for agent in farm1.schedule.agents if isinstance(agent, PickerRobot)])}, Number of Explorer Drones: {len([agent for agent in farm1.schedule.agents if isinstance(agent, ExplorerDrone)])}")
        self.assertIsNotNone(farm1)
        self.assertEqual(len([agent for agent in farm1.schedule.agents if isinstance(agent, PickerRobot)]), 2)
        self.assertEqual(len([agent for agent in farm1.schedule.agents if isinstance(agent, ExplorerDrone)]), 2)

        # Test with more robots
        farm2 = StrawberryFarm(n_picker=5, n_explorer=5, width=30, height=30, strawberry_percentage=10, mode="Basic")
        print(f"Farm 2 => Number of Picker Robots: {len([agent for agent in farm2.schedule.agents if isinstance(agent, PickerRobot)])}, Number of Explorer Drones: {len([agent for agent in farm2.schedule.agents if isinstance(agent, ExplorerDrone)])}")
        self.assertIsNotNone(farm2)
        self.assertEqual(len([agent for agent in farm2.schedule.agents if isinstance(agent, PickerRobot)]), 5)
        self.assertEqual(len([agent for agent in farm2.schedule.agents if isinstance(agent, ExplorerDrone)]), 5)

        # Test if the number of robots are different
        self.assertNotEqual(len([agent for agent in farm1.schedule.agents if isinstance(agent, PickerRobot)]), len([agent for agent in farm2.schedule.agents if isinstance(agent, PickerRobot)]))
        self.assertNotEqual(len([agent for agent in farm1.schedule.agents if isinstance(agent, ExplorerDrone)]), len([agent for agent in farm2.schedule.agents if isinstance(agent, ExplorerDrone)]))
        if len([agent for agent in farm1.schedule.agents if isinstance(agent, PickerRobot)]) != len([agent for agent in farm2.schedule.agents if isinstance(agent, PickerRobot)]) and len([agent for agent in farm1.schedule.agents if isinstance(agent, ExplorerDrone)]) != len([agent for agent in farm2.schedule.agents if isinstance(agent, ExplorerDrone)]):
            print("Number of robots are different.")


    def test_model_with_different_strawberry_percentage(self):
        farm1 = StrawberryFarm(n_picker=2, n_explorer=2, width=30, height=30, strawberry_percentage=20, mode="Basic")
        print(f"Model 1 created with {farm1.strawberry_percentage}% strawberries")
        trees_with_strawberries_in_farm1 = len([agent for agent in farm1.schedule.agents if isinstance(agent, Tree) and agent.state == READY_STRAWBERRY])
        trees_without_strawberries_in_farm1 = len([agent for agent in farm1.schedule.agents if isinstance(agent, Tree) and not agent.state == READY_STRAWBERRY])
        print(f"Farm 1 => Trees with strawberries: {trees_with_strawberries_in_farm1}, Trees without strawberries: {trees_without_strawberries_in_farm1}")

        farm2 = StrawberryFarm(n_picker=2, n_explorer=2, width=30, height=30, strawberry_percentage=100, mode="Basic")
        print(f"Model 2 created with {farm2.strawberry_percentage}% strawberries")
        trees_with_strawberries_in_farm2 = len([agent for agent in farm2.schedule.agents if isinstance(agent, Tree) and agent.state == READY_STRAWBERRY])
        trees_without_strawberries_in_farm2 = len([agent for agent in farm2.schedule.agents if isinstance(agent, Tree) and not agent.state == READY_STRAWBERRY])
        print(f"Farm 2 => Trees with strawberries: {trees_with_strawberries_in_farm2}, Trees without strawberries: {trees_without_strawberries_in_farm2}")

    def test_communication(self):
        farm = StrawberryFarm(n_picker=1, n_explorer=1, width=50, height=50, strawberry_percentage=10, mode="Extended")
        drone = next(agent for agent in farm.schedule.agents if isinstance(agent, ExplorerDrone))
        picker = next(agent for agent in farm.schedule.agents if isinstance(agent, PickerRobot))

        # Test drone signaling picker robot
        tree = next(agent for agent in farm.schedule.agents if isinstance(agent, Tree) and agent.state is READY_STRAWBERRY)
        tree_pos = (tree.x, tree.y)
        drone.x, drone.y = tree_pos

        # Find a free cell for the robot
        free_cell_found = False
        for x in range(farm.width):
            for y in range(farm.height):
                if not farm.grid.get_cell_list_contents((x, y)):
                    picker.x, picker.y = x, y
                    free_cell_found = True
                    break
            if free_cell_found:
                break

        # Ensure a free cell was found
        self.assertTrue(free_cell_found, "No free cell found for the robot")

        # Initial battery of the robot
        initial_picker_battery = picker.battery
        initial_drone_battery = drone.battery

        drone.send_signal()
        print("Drone signaled the robot")

        while picker.target_tree:
            drone.step()
            picker.step()

        self.assertTrue(picker.is_Free)
        self.assertTrue(tree.is_strawberry_not_ready)
        self.assertTrue(picker.target_tree == None)
        self.assertTrue(picker.target_tree_pos == None)
        self.assertLess(picker.battery, initial_picker_battery, "Robot's battery should be consumed during the movement")
        self.assertLess(drone.battery, initial_drone_battery, "Drone's battery should be consumed during the movement")

        print(f"Picker's Battery Consumption: -{initial_picker_battery- picker.battery}, Drone's Battery Consumption: -{initial_drone_battery - drone.battery}")

    def test_auction_mechanism(self):
        farm = StrawberryFarm(n_picker=5, n_explorer=1, width=50, height=50, strawberry_percentage=10, mode="Novel")
        tree = next(agent for agent in farm.schedule.agents if isinstance(agent, Tree) and agent.state is READY_STRAWBERRY)
        tree_pos = (tree.x, tree.y)

        # Place robots in different, free cells
        for picker in farm.picker_robots:
            free_cell_found = False
            for x in range(farm.width):
                for y in range(farm.height):
                    if not farm.grid.get_cell_list_contents((x, y)):
                        picker.x, picker.y = x, y
                        picker.battery = int(random.uniform(0, MAX_BATTERY))
                        picker.update_battery_state()
                        farm.grid.place_agent(picker, (x, y))
                        free_cell_found = True
                        break
                if free_cell_found:
                    break

        initial_farm_tokens = farm.farm_tokens
        winner = farm.run_auction(tree_pos) # Run the auction

        print(f"Farm tokens => before auction: {initial_farm_tokens}, after auction: {farm.farm_tokens}")
        self.assertGreater(farm.farm_tokens, initial_farm_tokens) # Check if the farm tokens increased

        for picker in farm.picker_robots:
            if picker.id == winner:
                print(f"Winner's tokens => before auction: {MAX_TOKEN}, after auction: {picker.tokens}")
                self.assertLess(picker.tokens, MAX_TOKEN) # Check if the winner's tokens decreased
                break

        self.assertIsNotNone(winner)

    def test_q_table_initialisation(self):
        farm = StrawberryFarm(n_picker=1, n_explorer=1, width=50, height=50, strawberry_percentage=10, mode="Novel")
        picker = next(agent for agent in farm.schedule.agents if isinstance(agent, PickerRobot))
        self.assertIsNotNone(picker.q_table, "Q-table should be initialised")
        print("Q-table initialised.")

    def test_epsilon_greedy_action_selection(self):
        farm = StrawberryFarm(n_picker=1, n_explorer=1, width=50, height=50, strawberry_percentage=10, mode="Novel")
        picker = next(agent for agent in farm.schedule.agents if isinstance(agent, PickerRobot))
        state = picker.get_state()
        action = picker.get_action()
        print(f"State: {state}, Action: {action}")
        self.assertIn(action, range(picker.get_action_space_size()), "Action should be within the valid action space")
        print(f"Epsilon-greedy action selected {action} for {range(picker.get_action_space_size())}.")

    def test_q_learning_updates(self):
        farm = StrawberryFarm(n_picker=1, n_explorer=1, width=50, height=50, strawberry_percentage=10, mode="Novel")
        picker = next(agent for agent in farm.schedule.agents if isinstance(agent, PickerRobot))
        initial_q_table = np.copy(picker.q_table)
        initial_epsilon = picker.exploration_rate

        # Simulate updating rewards
        picker.update_rewards()

        updated_q_table = picker.q_table
        self.assertTrue(np.any(updated_q_table != initial_q_table), "Q-Table should be updated")
        self.assertLess(picker.exploration_rate, initial_epsilon, "Epsilon should be decremented")

        print(f"Epsilon => before update: {initial_epsilon}, after update: {picker.exploration_rate}")
        # Print the differences
        differences = np.where(initial_q_table != updated_q_table)
        for diff in zip(*differences):
            print(f"Q-Table reward difference at {diff}: {initial_q_table[diff]} -> {updated_q_table[diff]}")

    def test_battery_never_below_zero(self):
        farm = StrawberryFarm(n_picker=5, n_explorer=5, width=50, height=50, strawberry_percentage=10, mode="Novel")

        # Run the simulation for a specified number of steps
        num_steps = 2000
        for _ in range(num_steps):
            farm.step()

            # Check battery levels of all robots
            for agent in farm.schedule.agents:
                if isinstance(agent, (PickerRobot)):
                    self.assertGreaterEqual(agent.battery, 0, f"Robot {agent.id}'s battery went below 0")

        print("All robots' batteries remained above 0 during the simulation.")


    def run_simulation_with_seed(self,  num_steps, mode):
        farm = StrawberryFarm(n_picker=1, n_explorer=1, width=50, height=50, strawberry_percentage=50, mode=mode)
        for step in range(num_steps):
            if not farm.running:
                break
            farm.step()
        return farm.datacollector.get_model_vars_dataframe()

    def test_seed_consistency(self):
        num_steps = 100
        modes = ["Basic", "Extended", "Novel"]
        results = {}

        # Run the simulation twice for each mode with the same seed
        for mode in modes:
            results[f"{mode}_1"] = self.run_simulation_with_seed( num_steps, mode)
            results[f"{mode}_2"] = self.run_simulation_with_seed( num_steps, mode)
            results[f"{mode}_3"] = self.run_simulation_with_seed( num_steps, mode)
            results[f"{mode}_4"] = self.run_simulation_with_seed( num_steps, mode)
            results[f"{mode}_5"] = self.run_simulation_with_seed( num_steps, mode)

       # Compare the results for each mode
        for mode in modes:
            print(f"Mode: {mode}")
            base_data = results[f"{mode}_1"]
            for i in range(2, 6):
                current_data = results[f"{mode}_{i}"]
                self.assertTrue(base_data.equals(current_data), f"Results are not consistent across runs with the same seed for mode {mode}")
            print(f"Results are consistent across runs with the same seed for mode {mode}.")

if __name__ == '__main__':
    unittest.main()