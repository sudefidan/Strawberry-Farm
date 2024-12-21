import mesa
from mesa.visualization import CanvasGrid
from farm.model import StrawberryFarm
from farm.portrayal import farm_portrayal

# Function to create a CanvasGrid with dynamic width and height
def create_canvas_grid(width, height):
    cell_size = 30  # Define the size of each cell in pixels
    canvas_width = width * cell_size
    canvas_height = height * cell_size
    print("Creating canvas grid with width:", canvas_width)
    return CanvasGrid(farm_portrayal, width, height, canvas_width, canvas_height)

# Initial grid size
initial_width = 30
initial_height = 25

# Create the initial CanvasGrid
canvas_element = create_canvas_grid(initial_width, initial_height)

model_params = {
    "n_picker": mesa.visualization.Slider("Number of Picker Robots", 1, 1, 20, 1),
    "n_explorer": mesa.visualization.Slider("Number of Explorer Drones", 1, 1, 20, 1),
    "strawberry_percentage": mesa.visualization.Slider("Percentage of Strawberries", 10, 0, 100, 1),
    "width": mesa.visualization.Slider("Width of Grid", initial_width, 10, 30, 5),
    "height": mesa.visualization.Slider("Height of Grid", initial_height, 10, 25, 1),
    "mode": mesa.visualization.Choice("Mode", "Extended", ["Basic", "Extended", "Novel"]),
}

def model_gen(n_picker, n_explorer, width, height, strawberry_percentage, mode):
    return StrawberryFarm(n_picker, n_explorer, width, height, strawberry_percentage, mode)

class CustomModularServer(mesa.visualization.ModularServer):
    def __init__(self, model_cls, visualization_elements, name, model_params):
        super().__init__(model_cls, visualization_elements, name, model_params)
        self.model_params = model_params

    def reset_model(self):
        super().reset_model()
        new_canvas = update_canvas_grid(model_params)
        self.visualization_elements[0] = new_canvas

# Function to update the canvas grid whenever the model parameters change
def update_canvas_grid(params):
    print("Updating canvas grid with new width:", params["width"].value)
    width = params["width"].value
    height = params["height"].value
    return create_canvas_grid(width, height)

server = CustomModularServer(
    model_gen,
    [canvas_element],
    "Strawberry Farm",
    model_params,
)


server.static_path = "img"