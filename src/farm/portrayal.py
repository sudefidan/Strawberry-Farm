from farm.agents import *
import os


def farm_portrayal(agent):
    """
    Determine which portrayal to use according to the type of agent.
    """
    if isinstance(agent, PickerRobot) and not agent.is_in_the_house():
        return get_portrayal(agent, "robot")
    elif isinstance(agent, ChargerRobot) and not agent.is_in_the_house():
        return get_portrayal(agent, "charger_robot")
    elif isinstance(agent, ExplorerDrone):
        return get_portrayal(agent, "explorer_drone")
    elif isinstance(agent, Tree):
        return get_portrayal(agent, "strawberry") if agent.is_strawberry_ready else get_portrayal(agent, "tree")
    elif isinstance(agent, River):
        return get_portrayal(agent, "river")
    elif isinstance(agent, ChargingStation):
        return get_portrayal(agent, "charging_station")
    elif isinstance(agent, CollectionPoint):
        return get_portrayal(agent, "collection_point")
    elif isinstance(agent, House):
        return get_portrayal(agent, "house")
    else:
        return None  # For other agents or background

def get_portrayal(agent, img_name):
    if agent == None:
        raise AssertionError
    portrayal = {
        "Shape": get_image_path(img_name),
        "Filled": "true",
        "Layer": 0,
        "x": agent.x,
        "y": agent.y,
    }
    if isinstance(agent, PickerRobot) and agent.is_Signaled:
        portrayal["text"] = str(agent.id)  # Display the robot's ID
        portrayal["text_size"] = 20
    if isinstance(agent, ExplorerDrone) and agent.get_signaled_robot_id() != None:
        portrayal["text"] = str(agent.get_signaled_robot_id())  # Display the robot's ID
        portrayal["text_size"] = 20
    if isinstance(agent, (PickerRobot, ExplorerDrone, ChargerRobot)) and agent.is_low_battery:
        portrayal["text"] = "LOW"  # Display the battery state of the robot
        portrayal["text_size"] = 20
    if isinstance(agent, (PickerRobot,ExplorerDrone, ChargerRobot)) and agent.is_charging:
        portrayal["text"] = str(int(agent.battery))  # Display the battery of the robot
        portrayal["text_size"] = 20
    if isinstance(agent, ChargerRobot) and agent.is_Signaled:
        portrayal["text"] = str(agent.signaled_by.id)
        portrayal["text_size"] = 20
    return portrayal


def get_image_path(img_name):
    current_dir = os.path.dirname(__file__)
    img_dir = os.path.join(current_dir, "img")
    return os.path.join(img_dir, "{}.svg".format(img_name))
    #return "{}/{}.svg".format("/home/sude2.fidan/AA-MAS/aamas2324-portfolio-sp1-sudefidan/img", img_name)
