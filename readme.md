# 🚀 Getting Started

This repository contains a simulation of a strawberry farm with different types of robots. The robots include picker robots and explorer drones, each with specific behaviors and tasks. The simulation is implemented using the Mesa framework.

## 📋 Activities

- Code -> [Strawberry Farm](/src/)
- Concept  Slides -> [Novel Feature](/TheStrawberryFarmNovelties.pdf)
- Performance Analysis -> [KPI](/kpi.ipynb)

### 🖥️  How to Run

1. **Install Dependencies:**
   - Install the required libraries using the following command:

     ```sh
     pip install -r requirements.txt
     ```

2. **Run the Simulation:**
   - Navigate to the `src` directory.
   - Run the simulation using the following command:

     ```sh
     python run.py
     ```

3. **Run Tests:**
   - Navigate to the `src` directory.
   - Run the code using the following command:

     ```sh
     python test.py
     ```

4. **Run KPIs:**
   - Run the `kpi.ipynb` file.

### 💡 FarmRobot Class Explanation

This is a parent class for farm agents with common functionalities.

**Battery Management:** When the robot's battery level falls below `LOW_BATTERY_LEVEL`, the robot's battery state is be changed to `LOW_BATTERY`. In the Basic and Extended mode, the robot prioritises returning to the nearest free charging station to recharge using the Breadth-First Search algorithm to find the shortest path. When there is no free station, it waits until it finds one. In the Novel mode, the robot will signal a _Charger Robot_ to come and charge it. The robot consumes less battery while `CHARGING` or moving to charging station at the rate of(`LOW_BATTERY_CONSUMPTION_RATE`). The battery recharges its battery until it reaches `MAX_BATTERY` level and then the robot's battery state is changed to `ENOUGH_BATTERY`. In the basic and extended modes of the model, charger robots do not exist. Consequently, when a robot does not reach a charging station before its battery level drops below 0, it stops moving and cannot continue its tasks. Therefore, model stops.

**Movement:** The robot moves randomly to locate strawberries and consumes battery (`BATTERY_CONSUMPTION_RATE`).  However, when a picker robot encounters a river, it takes longer to cross due to the increased difficulty of movement. This is simulated by introducing a delay (`crossing_delay`) and increasing the battery consumption rate as(`RIVER_CROSSING_BATTERY_CONSUMPTION_RATE`) while crossing the river. The robot continues moving in the same direction until it exits the river. Unlike picker robots, explorer drones and charger robots are not affected by rivers since they fly over them. In Novel Mode, this random movement is used to determine to rewards for reinforcement learning.

**Q-Learning:** In the Novel mode, the FarmRobot class implements Q-Learning to enable reinforcement learning for the robots random moves.

- **Initialisation:** The Q-Table is initialised with zeros. Parameters such as `learning_rate`, `discount_factor`, `exploration_rate`, `min_exploration_rate`, and `exploration_decay_rate` are set.
- **State Representation:** The state is represented by the robot's position on the grid. The state space size is determined by the width and height of the grid.
- **Action Space:** The action space consists of four possible actions: moving up, down, left, or right. Therefore, the action space size is set to 4.
- **Epsilon-Greedy Action Selection:** The robot selects actions using an epsilon-greedy strategy. With probability `exploration_rate`, the robot chooses a random action (exploration). Otherwise, it chooses the action with the highest Q-value for the current state (exploitation).
- **Q-Table Update:** After taking an action, the robot receives a reward and transitions to a new state. The Q-Table is updated using the Temporal Difference (TD) update rule to allow the robots to learn from their experiences in real-time, without requiring a complete model of the environment. In the formula, `learning_rate` determines how much of the new information is used to update the Q-Value and `discount_factor` determines the importance of future rewards to scale the estimated next state-action pair.
- **Reward Function:** The reward function uses `move_randomly` function to determine rewards based on the robot's actions. For example, moving to a empty cell result in a negative reward, while reaching a tree with strawberry result in a positive reward.
- **Exploration Decay:** The `exploration_rate` decays over time to encourage the robot to exploit learned actions more as it gains experience. This is done by gradually reducing epsilon after each episode, using `exploration_decay_rate`. This ensures that the robot starts with a high exploration rate to discover new actions and states, but over time, it shifts towards exploiting the knowledge it has gained to maximize rewards.

### 💡 Picker Robot Class Explanation

**Payload Management:** The robot can carry a maximum of `MAX_PAYLOAD` strawberries. When the payload is full, the robot returns to the nearest collection point to drop off the strawberries using the Breadth-First Search algorithm to find the shortest available path.

**Strawberry Collection:** The robot checks for strawberries in the next 3 trees around it. If strawberries are found, the robot picks them. When multiple robots are trying to pick the same strawberries, priority is given to the robot with the lower ID. This ensures that there is no conflict and that the robot with the lower ID gets to pick the strawberries first.

**Communication with Drones:** In the Extended mode, picker robots receive a signal of a ready-to-pick strawberry's location from the drones and then move to the target tree to collect the strawberry.

**Signaling Charger Robot:** In the Novel mode, when a picker robot is in the `LOW_BATTERY` state, it will send a signal to a free charger robot to come and charge itself.

**Bidding For Auction:** In the Novel mode, available robots will bid for the auction to collect a strawberry. The bidding depends on the distance to the tree and the battery left in the robot. The winner of the auction has the pay the payment to the farm.

### 💡 ExplorerDrone Class Explanation

**Strawberry Detection:** The drone checks if there is a strawberry underneath it. If a strawberry is found, the drone waits for the picker robot to pick the strawberries. When there is already a drone waiting for the same strawberries, the drone skips the strawberries and looks for others. This ensures that there is no conflict and that there is only one drone waiting for one strawberry tree at a time.

**Communication with Picker Robots:** In the Extended Mode, the drone will send the location to the picker robot to come and pick the strawberry when it detects a strawberry. In the Novel Mode, the drone will trigger the model to run an auction to determine the winning bid for collecting the strawberries.

### 💡 Tree Class Explanation

**Strawberry Growth:** In the Extended and Novel modes, trees continuously grow strawberries over time to simulate real-life growth cycles.

### 💡 Charger Robot Class Explanation

**Charging Picker Robots:** In the Novel mode, when a robot is in the `LOW_BATTERY`, it sends a signal to a free charger robot to come and charge it. The charger robot moves to the robot and charges its battery. Once the picker robot's battery is full, the charger robot updates its state to `FREE` and waits for the next signal. In the basic and extended modes of the model, charger robots do not exist. Consequently, when a robot does not reach a charging station before its battery level drops below 0, it stops moving and cannot continue its tasks.Therefore, model stops.

### 💡 Strawberry Farm Model Explanation

**Agent Placement:**

- **Rivers:** Rivers are placed in the first 3 columns of every 15 columns block.
- **Trees:** Trees are placed in the first 3 columns of every 5 columns block, with a certain percentage of trees having strawberries given by the user.
- **Collection Point:** A collection point is placed at a random border position.
- **Charging Stations:** Charging stations are placed at random border positions, with the number of stations based on the total number of robots.
- **Picker Robots, Explorer Drones and Charger Robots:** Robots are placed at random positions on the grid. Charger Robots only exist in the Novel mode fo operation.
- **House:** In Novel and Extended mode, house is placed at random border positions.

**Auction:** The model uses an First-come sealed auction mechanism to allocate strawberry-picking tasks to picker robots. When a tree with ripe strawberries is detected, the drone triggers to model to run an auction to allocate the task of picking the strawberries. Then the farm evaluates the bids and selects the picker robot with the highest bid as the winner of the auction. The winning picker robot is assigned the task of picking the strawberries from the specified tree and it pays the bid amount to the farm. This payment is deducted from the robot's tokens, and the farm's tokens are increased accordingly.

**Completion:** The model checks if all strawberries are collected and dropped off. If so, the model stops running. This will ensure that all the strawberries are in collection point for the model to stop.

**Seed:** A seed has been implemented to ensure reproducibility in the model's simulations. By setting a fixed seed, the random elements in the model (such as agent placement and actions) are consistent across different runs. This allows for accurate and reliable performance analysis, as the results can be replicated and compared under the same conditions.

**Data Collector:** A data collector from Mesa has been implemented to gather and analyse KPIs (Key Performance Indicators) during the simulation. This tool allows for the systematic collection of data such as the number of strawberries picked, battery consumption, and distance traveled by robots.

### 💡 Server and Grid Parameters Explanation

**Model Parameters:**

- **Number of Picker Robots (`n_picker`):** Users can choose the number of picker robots in the simulation.
- **Number of Explorer Drones (`n_explorer`):** Users can choose the number of explorer drones in the simulation.
- **Percentage of Strawberries (`strawberry_percentage`):** Users can choose the percentage of trees that have strawberries.
- **Width of Grid (`width`):** Users can choose the width of the grid which ensures that the farm model works with any size.
- **Height of Grid (`height`):** Users can choose the height of the grid which ensures that the farm model works with any size.
- **Mode of Operation(`mode`):** Users can choose mode of operation to run the model with.

**Canvas Grid Creation:**
   Each cell in the canvas is 30 pixels. The total width of the canvas is calculated as `width * cell_size` and the total height of the canvas is calculated as `height * cell_size`.

## 🎯 Novel Feature Slides

The novel features of the strawberry farm simulation are detailed in the slides. Open the [Novel Features](/TheStrawberryFarmNovelties.pdf) file to view it.

## 🎯 Performance Analysis

The performance analysis of the strawberry farm model is conducted using various Key Performance Indicators (KPIs). These KPIs include:

- **Strawberry Collection:** Measures the number of strawberries collected within a specified number of steps.
- **Strawberries Dropped-Off:** Measures the number of strawberries collected and successfully delivered to the collection points.
- **Picker Robots Distance Traveled Efficiency:** Measures the efficiency rate as the ratio of strawberries picked to the total distance traveled by the picker robots.
- **Picker Robots Battery Efficiency:** Measures the efficiency rate as the ratio of strawberries picked to the total battery usage.

The KPIs are analysed using data collected during the simulation. A data collector from Mesa has been implemented to systematically gather and analyse these KPIs.

The detailed analysis and visualisations of these KPIs can be found in the [KPI](/kpi.ipynb) file.

## 🤖 Use of Generative AI Tools

**ChatGPT:** Used to optimise the `breadth_first_search()` algorithm in [agents](/src/farm/agents.py), as the initial implementation was slow due to using initially `queue()` function instead `deque()`. I have learnt that `deque()` provides O(1) time complexity for append and pop operations from both ends, making it more efficient for single-threaded application compared to `queue()`, which has additional overhead due to its thread safety mechanism. Additionally, ChatGPT assisted in implementing `get_image_path()` function in model [portrayal](/src/farm/portrayal.py), resolving issues with full path usage.

**GitHub Copilot:** Used to assist in writing code comments for the simulation and generating the initial structure of the `solution.md` files as well as [KPI](/kpi.ipynb) file.

**Grammarly:** Used to check the grammar of the PDF slides for [Novel Feature Slides](/Part2Solutions/TheStrawberryFarmNovelties.pdf).

## 🤓 **Maintainers**

Sude Fidan ([@sudefidan](https://github.com/sudefidan))

## 📖  **Referencing**

- Model images are taken from [FreeSVG](https://freesvg.org)
- Diagrams created with [Drawio](https://www.drawio.com)
- [Visualization Tutorial](https://mesa.readthedocs.io/stable/tutorials/visualization_tutorial.html)
- [Introduction to Mesa: Agent-based Modeling in Python](https://towardsdatascience.com/introduction-to-mesa-agent-based-modeling-in-python-bcb0596e1c9a)
- [Q-Learning](https://www.geeksforgeeks.org/q-learning-in-python/)
