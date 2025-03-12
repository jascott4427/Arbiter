import sys
import time
import random
import math
import numpy as np
from shapely.geometry import Point, Polygon
import threading

# -------------------------------
# Pygame Initialization
# -------------------------------
import pygame
pygame.init()
SIM_WIDTH, SIM_HEIGHT = 800, 800  # Offscreen simulation surface size

# -------------------------------
# Matplotlib and Tkinter Imports
# -------------------------------
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# -------------------------------
# Simulation Constants
# -------------------------------
GRID_SIZE = 25
FPS = 120
FRAME_DELAY = 1 / FPS
MAX_ITER = 1000
SPEED = 5
TIME_STEP = 0.1                # Time resolution for 3D grid
TIME_HORIZON = GRID_SIZE * 3 / SPEED  # Total time horizon in seconds
TIME_RESOLUTION = int(TIME_HORIZON / TIME_STEP)  # Number of time steps
FLAG_POSITION = Point(GRID_SIZE - 2, GRID_SIZE - 2)
START_POSITION = Point(2, 2)
PLAYER_RADIUS = 0.5

# Wall configurations
WALL_CONFIGS = {
    "simple": [
        Polygon([(0, 0), (GRID_SIZE, 0), (GRID_SIZE, 1), (0, 1)]),
        Polygon([(0, GRID_SIZE - 1), (GRID_SIZE, GRID_SIZE - 1), (GRID_SIZE, GRID_SIZE), (0, GRID_SIZE)]),
        Polygon([(0, 0), (1, 0), (1, GRID_SIZE), (0, GRID_SIZE)]),
        Polygon([(GRID_SIZE - 1, 0), (GRID_SIZE, 0), (GRID_SIZE, GRID_SIZE), (GRID_SIZE - 1, GRID_SIZE)])
    ],
    "obstacles": [
        Polygon([(0, 0), (GRID_SIZE, 0), (GRID_SIZE, 1), (0, 1)]),
        Polygon([(0, GRID_SIZE - 1), (GRID_SIZE, GRID_SIZE - 1), (GRID_SIZE, GRID_SIZE), (0, GRID_SIZE)]),
        Polygon([(0, 0), (1, 0), (1, GRID_SIZE), (0, GRID_SIZE)]),
        Polygon([(GRID_SIZE - 1, 0), (GRID_SIZE, 0), (GRID_SIZE, GRID_SIZE), (GRID_SIZE - 1, GRID_SIZE)]),
        Polygon([(10, 10), (15, 10), (15, 15), (10, 15)]),
        Polygon([(5, 20), (10, 20), (10, 25), (5, 25)])
    ],
    "maze": [
        Polygon([(0, 0), (GRID_SIZE, 0), (GRID_SIZE, 1), (0, 1)]),
        Polygon([(0, GRID_SIZE - 1), (GRID_SIZE, GRID_SIZE - 1), (GRID_SIZE, GRID_SIZE), (0, GRID_SIZE)]),
        Polygon([(0, 0), (1, 0), (1, GRID_SIZE), (0, GRID_SIZE)]),
        Polygon([(GRID_SIZE - 1, 0), (GRID_SIZE, 0), (GRID_SIZE, GRID_SIZE), (GRID_SIZE - 1, GRID_SIZE)]),
        Polygon([(5, 5), (10, 5), (10, 10), (5, 10)]),
        Polygon([(15, 5), (20, 5), (20, 10), (15, 10)]),
        Polygon([(5, 15), (10, 15), (10, 20), (5, 20)]),
        Polygon([(15, 15), (20, 15), (20, 20), (15, 20)])
    ]
}

# -------------------------------
# Node and RRT3D Classes
# -------------------------------
class Node:
    def __init__(self, point, parent=None, time=None):
        self.point = point
        self.parent = parent
        self.time = time

class RRT3D:
    def __init__(self, start, goal, obstacles, max_iter=MAX_ITER, step_size=1.0, max_speed=SPEED, goal_bias_probability=0.05):
        self.start = Node(start, None, 0)
        self.goal = Node(goal)
        self.obstacles = obstacles
        self.max_iter = max_iter
        self.step_size = step_size
        self.nodes = [self.start]
        self.occupancy_grid = np.zeros((GRID_SIZE, GRID_SIZE, TIME_RESOLUTION))
        self.lock = threading.Lock()
        self.max_speed = max_speed
        self.goal_bias_probability = goal_bias_probability

    def generate_random_point(self):
        if random.random() < self.goal_bias_probability:
            direction = np.arctan2(self.goal.point.y - self.start.point.y, self.goal.point.x - self.start.point.x)
            distance = self.step_size * random.uniform(0, 1)
            x = self.goal.point.x - distance * np.cos(direction)
            y = self.goal.point.y - distance * np.sin(direction)
            t = random.uniform(0, TIME_HORIZON)
            return Point(x, y), t
        else:
            x = random.uniform(0, GRID_SIZE)
            y = random.uniform(0, GRID_SIZE)
            t = random.uniform(0, TIME_HORIZON)
            return Point(x, y), t

    def nearest_node(self, point, time):
        return min(self.nodes, key=lambda node: np.sqrt((node.point.x - point.x)**2 +
                                                         (node.point.y - point.y)**2 +
                                                         (node.time - time)**2))

    def new_point(self, nearest, random_point, random_time):
        direction = math.atan2(random_point.y - nearest.point.y, random_point.x - nearest.point.x)
        new_x = nearest.point.x + self.step_size * math.cos(direction)
        new_y = nearest.point.y + self.step_size * math.sin(direction)
        new_time = nearest.time + self.step_size / self.max_speed
        distance = math.sqrt((new_x - nearest.point.x)**2 + (new_y - nearest.point.y)**2)
        travel_time = distance / self.max_speed
        if new_time > nearest.time + travel_time:
            new_time = nearest.time + travel_time
        return Point(new_x, new_y), new_time

    def collision_free(self, point, time):
        for wall in self.obstacles:
            if wall.distance(point) < PLAYER_RADIUS:
                return False
        x_idx = int(point.x)
        y_idx = int(point.y)
        t_idx = int(time / TIME_STEP)
        if x_idx < 0 or x_idx >= GRID_SIZE or y_idx < 0 or y_idx >= GRID_SIZE or t_idx < 0 or t_idx >= TIME_RESOLUTION:
            return False
        if self.occupancy_grid[x_idx, y_idx, t_idx] == 1:
            return False
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dt in range(-1, 2):
                    if (0 <= x_idx + dx < GRID_SIZE and
                        0 <= y_idx + dy < GRID_SIZE and
                        0 <= t_idx + dt < TIME_RESOLUTION):
                        if math.sqrt((dx * GRID_SIZE)**2 + (dy * GRID_SIZE)**2 + (dt * TIME_STEP)**2) <= 0.5:
                            if self.occupancy_grid[x_idx + dx, y_idx + dy, t_idx + dt] == 1:
                                return False
        return True

    def plan(self):
        for i in range(self.max_iter):
            random_point, random_time = self.generate_random_point()
            nearest = self.nearest_node(random_point, random_time)
            new_point, new_time = self.new_point(nearest, random_point, random_time)
            if self.collision_free(new_point, new_time):
                new_node = Node(new_point, nearest, new_time)
                self.nodes.append(new_node)
                if new_point.distance(self.goal.point) <= self.step_size:
                    self.goal.parent = new_node
                    self.goal.time = new_node.time
                    path = self.reconstruct_path(self.goal)
                    print(f"Path found after {i+1} iterations!")
                    return path
        print("No path found")
        return None

    def reconstruct_path(self, node):
        path = []
        while node:
            t = node.time if node.time is not None else (node.parent.time if node.parent else 0)
            path.append((node.point, t))
            node = node.parent
        return path[::-1]

# -------------------------------
# MovingEntity Base Class
# -------------------------------
class MovingEntity:
    def __init__(self, init_pos, init_speed, path):
        self.position = init_pos
        self.speed = init_speed
        self.path = path
        self.current_target = 0
        self.forward = True

    def move(self, delta_time):
        # Exit if no path or all targets have been reached
        if not self.path or self.current_target >= len(self.path):
            return

        # Get current target and compute direction
        target = self.path[self.current_target]
        target_point = target[0] if isinstance(target, tuple) else target
        direction = np.arctan2(target_point.y - self.position.y, target_point.x - self.position.x)

        # Calculate new position based on speed and delta time
        distance_to_move = self.speed * delta_time
        new_x = self.position.x + distance_to_move * np.cos(direction)
        new_y = self.position.y + distance_to_move * np.sin(direction)
        new_position = Point(new_x, new_y)

        # Check if new position overshoots target, and handle excess distance
        distance_to_target = self.position.distance(target_point)
        if new_position.distance(target_point) > distance_to_target:
            self.position = target_point
            excess_distance = new_position.distance(target_point) - distance_to_target

            # Move to next target, handling forward/backward movement
            if self.forward:
                self._move_to_next_target(excess_distance)
            else:
                self._move_to_previous_target(excess_distance)
        else:
            self.position = new_position

    def _move_to_next_target(self, excess_distance):
        # Move to the next target along the path
        if self.current_target + 1 < len(self.path):
            self.current_target += 1
            self._update_position(excess_distance)
        else:
            self.forward = False

    def _move_to_previous_target(self, excess_distance):
        # Move to the previous target along the path
        if self.current_target - 1 >= 0:
            self.current_target -= 1
            self._update_position(excess_distance)
        else:
            self.forward = True

    def _update_position(self, excess_distance):
        next_target = self.path[self.current_target]
        next_target_point = next_target[0] if isinstance(next_target, tuple) else next_target
        direction_to_next = np.arctan2(next_target_point.y - self.position.y, next_target_point.x - self.position.x)
        self.position = Point(self.position.x + excess_distance * np.cos(direction_to_next),
                              self.position.y + excess_distance * np.sin(direction_to_next))

# -------------------------------
# Arbiter Class (Original set_path and move)
# -------------------------------
class Arbiter(MovingEntity):
    def __init__(self, init_pos, init_speed):
        super().__init__(init_pos, init_speed, [])
        self.current_time = 0  # Track elapsed time
        self.reset_time = 0

    def set_path(self, path):
        self.path = path  # Expecting list of (Point, time) tuples
        self.current_target = 0
        self.reset_time += self.current_time
        self.current_time = 0  # Reset time when a new path is set

    def move(self, delta_time):
        if not self.path or self.current_target >= len(self.path):
            return
        target = self.path[self.current_target]
        if isinstance(target, tuple):
            target_point, target_time = target
        else:
            target_point = target
            target_time = None
        direction = np.arctan2(target_point.y - self.position.y, target_point.x - self.position.x)
        distance_to_target = self.position.distance(target_point)
        if target_time is not None:
            time_difference = target_time - self.current_time
            if time_difference > 0:
                required_speed = distance_to_target / time_difference
                self.speed = min(required_speed, SPEED)
                print('speeds: ', self.speed, required_speed, "\n")
            else:
                self.speed = SPEED
        else:
            self.speed = SPEED
        distance_to_move = self.speed * delta_time
        new_x = self.position.x + distance_to_move * np.cos(direction)
        new_y = self.position.y + distance_to_move * np.sin(direction)
        new_position = Point(new_x, new_y)
        if new_position.distance(target_point) > distance_to_target:
            excess_distance = new_position.distance(target_point) - distance_to_target
            self.position = target_point
            if self.forward:
                if self.current_target + 1 < len(self.path):
                    self.current_target += 1
                    next_target = self.path[self.current_target]
                    if isinstance(next_target, tuple):
                        next_target_point, next_target_time = next_target
                    else:
                        next_target_point = next_target
                    direction_to_next = np.arctan2(next_target_point.y - self.position.y,
                                                next_target_point.x - self.position.x)
                    self.position = Point(self.position.x + excess_distance * np.cos(direction_to_next),
                                            self.position.y + excess_distance * np.sin(direction_to_next))
                else:
                    self.forward = False
            else:
                if self.current_target - 1 >= 0:
                    self.current_target -= 1
                    next_target = self.path[self.current_target]
                    if isinstance(next_target, tuple):
                        next_target_point, next_target_time = next_target
                    else:
                        next_target_point = next_target
                    direction_to_next = np.arctan2(next_target_point.y - self.position.y,
                                                next_target_point.x - self.position.x)
                    self.position = Point(self.position.x + excess_distance * np.cos(direction_to_next),
                                            self.position.y + excess_distance * np.sin(direction_to_next))
                else:
                    self.forward = True
        else:
            self.position = new_position
        if target_time is not None:
            self.current_time += delta_time  # Update the current time

# -------------------------------
# Environment Class
# -------------------------------
class Environment:
    def __init__(self, map_name="simple", plot_update_callback=None):
        self.walls = WALL_CONFIGS.get(map_name, WALL_CONFIGS["simple"])
        self.enemy_path = [
            Point(GRID_SIZE // 2, GRID_SIZE // 2),
            Point(GRID_SIZE // 2, GRID_SIZE - 2), 
            Point(GRID_SIZE // 2, 2)
        ]
        self.enemy = MovingEntity(Point(GRID_SIZE//2, GRID_SIZE//2), SPEED, self.enemy_path)
        self.arbiter = Arbiter(START_POSITION, SPEED)
        self.rrt3d = RRT3D(START_POSITION, FLAG_POSITION, self.walls)
        path = self.rrt3d.plan()  # List of (Point, time) tuples.
        self.arbiter.set_path(path)
        self.last_time = time.time()
        self.state = "to_flag"  # "to_flag" or "to_start"
        self.pausing = False  # Flag to pause before replanning.
        self.screen = pygame.Surface((SIM_WIDTH, SIM_HEIGHT))
        self.font = pygame.font.Font(None, 36)
        self.plot_update_callback = plot_update_callback  # Callback for plot updates

    def update(self):
        self.update_occupancy_grid()
        current_time = time.time()
        delta_time = current_time - self.last_time
        self.last_time = current_time

        self.enemy.move(delta_time)
        if not self.pausing:
            self.arbiter.move(delta_time)

        if not self.pausing:
            if self.state == "to_flag" and self.arbiter.position.distance(FLAG_POSITION) < 0.001:
                self.pausing = True
                if self.plot_update_callback:
                    self.plot_update_callback()  # Trigger plot update
                self.root.after(0, self.replan_to_start)
            elif self.state == "to_start" and self.arbiter.position.distance(START_POSITION) < 0.001:
                self.pausing = True
                if self.plot_update_callback:
                    self.plot_update_callback()  # Trigger plot update
                self.root.after(0, self.replan_to_flag)

    def predict_enemy_position(self, time, last_time=0):
        """
        Predict the enemy's position at a given time based on its speed and path.
        """
        total_distance = 0  # Total distance traveled so far
        current_time = 0  # Current time elapsed
        current_segment = 0  # Current segment of the path
        path = self.enemy_path
        # Iterate through the path segments
        while current_segment < len(path) - 1:
            start = path[current_segment]
            end = path[current_segment + 1]
            segment_distance = start.distance(end)  # Distance of the current segment
            segment_time = segment_distance / self.enemy.speed  # Time to traverse the segment

            # Check if the predicted time falls within this segment
            if current_time + segment_time >= time+last_time:
                # Calculate the fraction of the segment traveled
                fraction = (time+last_time - current_time) / segment_time
                # Interpolate the position
                x = start.x + fraction * (end.x - start.x)
                y = start.y + fraction * (end.y - start.y)
                return Point(x, y)

            # Move to the next segment
            current_time += segment_time
            current_segment += 1

            if current_segment == len(path) - 1:
                path = path[::-1]
                current_segment = 0

        # If the time exceeds the total path time, return the last point
        return path[-1]

    def update_occupancy_grid(self):
        self.rrt3d.occupancy_grid = np.zeros((GRID_SIZE, GRID_SIZE, TIME_RESOLUTION))
        current_time = self.arbiter.reset_time + self.arbiter.current_time  # Get the current time from the arbiter
        #print(current_time)
        for t in range(TIME_RESOLUTION):
            # Shift the time reference by the current time
            shifted_time = current_time + t * TIME_STEP
            enemy_pos = self.predict_enemy_position(shifted_time)
            x_idx = int(enemy_pos.x)
            y_idx = int(enemy_pos.y)
            if 0 <= x_idx < GRID_SIZE and 0 <= y_idx < GRID_SIZE:
                with self.rrt3d.lock:
                    self.rrt3d.occupancy_grid[x_idx, y_idx, t] = 1

    def replan_to_start(self):
        self.update_occupancy_grid()
        self.rrt3d = RRT3D(FLAG_POSITION, START_POSITION, self.walls)
        path = self.rrt3d.plan()
        if path:
            self.arbiter.set_path(path)
        self.state = "to_start"
        self.pausing = False

    def replan_to_flag(self):
        self.update_occupancy_grid()
        self.rrt3d = RRT3D(START_POSITION, FLAG_POSITION, self.walls)
        path = self.rrt3d.plan()
        if path:
            self.arbiter.set_path(path)
        self.state = "to_flag"
        self.pausing = False

    def draw(self):
        self.screen.fill((255, 255, 255))
        for wall in self.walls:
            pts = [(int(x*20), int(y*20)) for x, y in wall.exterior.coords]
            pygame.draw.polygon(self.screen, (128, 128, 128), pts)
        pygame.draw.circle(self.screen, (0,255,0), (int(FLAG_POSITION.x*20), int(FLAG_POSITION.y*20)), 10)
        pygame.draw.circle(self.screen, (255,255,0), (int(START_POSITION.x*20), int(START_POSITION.y*20)), 10)
        pygame.draw.circle(self.screen, (255,0,0), (int(self.enemy.position.x*20), int(self.enemy.position.y*20)), 10)
        pygame.draw.circle(self.screen, (0,0,255), (int(self.arbiter.position.x*20), int(self.arbiter.position.y*20)), 10)
        speed_text = self.font.render(f"Speed: {self.arbiter.speed:.2f}", True, (0,0,0))
        self.screen.blit(speed_text, (10, 10))

    def get_rrt3d_data(self):
        tree_lines = []
        for node in self.rrt3d.nodes:
            if node.parent:
                tree_lines.append(((node.parent.point.x, node.parent.point.y, node.parent.time),
                                   (node.point.x, node.point.y, node.time)))
        path = self.arbiter.path
        return tree_lines, path

# -------------------------------
# Tkinter Widgets for GUI
# -------------------------------
class SimulationFrame(tk.Frame):
    def __init__(self, parent, env):
        super().__init__(parent)
        self.env = env
        self.label = tk.Label(self)
        self.label.pack(fill="both", expand=True)
        self.update_simulation()

    def update_simulation(self):
        self.env.update()
        self.env.draw()
        image_str = pygame.image.tostring(self.env.screen, "RGB")
        img = Image.frombytes("RGB", (self.env.screen.get_width(), self.env.screen.get_height()), image_str)
        photo = ImageTk.PhotoImage(img)
        self.label.configure(image=photo)
        self.label.image = photo
        self.after(int(1000/FPS), self.update_simulation)

class MatplotlibFrame(tk.Frame):
    def __init__(self, parent, env):
        super().__init__(parent)
        self.env = env
        self.figure = plt.figure(figsize=(7, 7))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.update_plot()  # Initial plot update

    def update_plot(self):
        self.figure.clf()
        ax = self.figure.add_subplot(111, projection="3d")
        ax.set_title("RRT3D Tree, Path, and Occupancy Grid")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Time (rel)")
        ax.set_ylim(GRID_SIZE, 0)
        with self.env.rrt3d.lock:
            occ_grid = self.env.rrt3d.occupancy_grid.copy()
        x_occ, y_occ, t_occ = np.where(occ_grid == 1)
        t_occ_seconds = t_occ * TIME_STEP
        #print("Occupied cells:", np.count_nonzero(occ_grid))
        ax.scatter(x_occ, y_occ, t_occ_seconds, c="red", marker="o", s=50, label="Occupied Cells")
        tree_lines, path = self.env.get_rrt3d_data()
        for (p0, p1) in tree_lines:
            if path:
                t0 = p0[2] - path[0][1]
                t1 = p1[2] - path[0][1]
            else:
                t0, t1 = p0[2], p1[2]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [t0, t1], c="black", linewidth=1)
        if path:
            start_time = path[0][1]
            xs = [pt[0].x for pt in path]
            ys = [pt[0].y for pt in path]
            ts = [t - start_time for (_, t) in path]
            ax.plot(xs, ys, ts, c="blue", linewidth=3, label="Chosen Path")
        rel_time = TIME_HORIZON
        t_vals = np.linspace(0, rel_time, 100)
        ax.plot([START_POSITION.x] * 100, [START_POSITION.y] * 100, t_vals, c="purple", linestyle="--", linewidth=2, label="Start")
        ax.plot([FLAG_POSITION.x] * 100, [FLAG_POSITION.y] * 100, t_vals, c="green", linestyle="--", linewidth=2, label="Goal")
        ax.legend()
        self.canvas.draw()

# -------------------------------
# Main Application Window (Tkinter)
# -------------------------------
class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulation with Embedded Pygame and Matplotlib (Tkinter)")
        self.geometry("1800x1000")
        self.env = Environment("simple")
        self.env.root = self
        sim_frame = SimulationFrame(self, self.env)
        sim_frame.pack(side="left", fill="both", expand=True)
        mpl_frame = MatplotlibFrame(self, self.env)
        mpl_frame.pack(side="left", fill="both", expand=True)
        # Pass the plot update callback to the environment
        self.env.plot_update_callback = mpl_frame.update_plot

# -------------------------------
# Main Function
# -------------------------------
def main():
    app = MainWindow()
    app.mainloop()

if __name__ == "__main__":
    main()
