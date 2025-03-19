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
SPEED = 10
TIME_STEP = 0.1                # Time resolution for 3D grid
TIME_HORIZON = GRID_SIZE * 3 / SPEED  # Total time horizon in seconds
FLAG_POSITION = Point(GRID_SIZE - 2, GRID_SIZE - 2)
START_POSITION = Point(2, 2)
PLAYER_RADIUS = 0.5
OCCUPANCY_RADIUS = 2

# -------------------------------
# Helper Function for Clamping
# -------------------------------
def clamp_point(point, min_val=0, max_val=GRID_SIZE):
    clamped_x = max(min_val, min(point.x, max_val))
    clamped_y = max(min_val, min(point.y, max_val))
    return Point(clamped_x, clamped_y)

# -------------------------------
# Wall Configurations
# -------------------------------
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
# Collision Checking Functions for PRM
# -------------------------------
def is_point_collision_free(point, t, obstacles, enemy):
    for wall in obstacles:
        if wall.distance(point) < PLAYER_RADIUS:
            return False
    if enemy is not None:
        pred_pos, pred_facing = enemy.predict_state(t)
        dx = point.x - pred_pos.x
        dy = point.y - pred_pos.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist <= enemy.fov_range:
            angle_to_point = math.degrees(math.atan2(dy, dx))
            facing_deg = math.degrees(pred_facing)
            angle_diff = (angle_to_point - facing_deg + 180) % 360 - 180
            if abs(angle_diff) <= enemy.fov_angle/2:
                return False
    return True

def is_edge_collision_free(n1, n2, obstacles, enemy, n_steps=10):
    p1, t1 = n1
    p2, t2 = n2
    for i in range(n_steps+1):
        fraction = i / n_steps
        interp_x = p1.x + fraction * (p2.x - p1.x)
        interp_y = p1.y + fraction * (p2.y - p1.y)
        interp_t = t1 + fraction * (t2 - t1)
        if not is_point_collision_free(Point(interp_x, interp_y), interp_t, obstacles, enemy):
            return False
    return True

# -------------------------------
# PRM Planner (Spatio-Temporal)
# -------------------------------
class PRMPlanner:
    @staticmethod
    def plan(start, goal, obstacles, enemy, n_samples=300, connection_radius=10, tolerance=1.0):
        samples = []
        samples.append((start, 0))
        samples.append((goal, TIME_HORIZON))
        while len(samples) < n_samples + 2:
            x = random.uniform(0, GRID_SIZE)
            y = random.uniform(0, GRID_SIZE)
            t = random.uniform(0, TIME_HORIZON)
            p = Point(x, y)
            if is_point_collision_free(p, t, obstacles, enemy):
                samples.append((p, t))
        graph = {i: [] for i in range(len(samples))}
        for i in range(len(samples)):
            for j in range(len(samples)):
                if i == j:
                    continue
                p1, t1 = samples[i]
                p2, t2 = samples[j]
                if t2 <= t1:
                    continue
                spatial_dist = p1.distance(p2)
                expected_dt = spatial_dist / SPEED
                if abs((t2 - t1) - expected_dt) < tolerance and spatial_dist < connection_radius:
                    if is_edge_collision_free(samples[i], samples[j], obstacles, enemy):
                        graph[i].append((j, spatial_dist))
        num_nodes = len(samples)
        dist = [float('inf')] * num_nodes
        prev = [None] * num_nodes
        dist[0] = 0
        unvisited = set(range(num_nodes))
        while unvisited:
            u = min(unvisited, key=lambda idx: dist[idx])
            unvisited.remove(u)
            if u == 1:
                break
            for v, w in graph[u]:
                alt = dist[u] + w
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
        if dist[1] == float('inf'):
            print("PRM: No path found.")
            return None
        path_indices = []
        u = 1
        while u is not None:
            path_indices.insert(0, u)
            u = prev[u]
        path = [samples[idx] for idx in path_indices]
        print(f"PRM: Path found with {len(path)} nodes.")
        return path

# -------------------------------
# MovingEntity Base Class (with FOV)
# -------------------------------
class MovingEntity:
    def __init__(self, init_pos, init_speed, path, fov_angle=60, fov_range=5):
        self.position = init_pos
        self.speed = init_speed
        self.path = path  # list of (Point, t) tuples
        self.current_target = 0
        self.forward = True
        self.fov_angle = fov_angle  # in degrees
        self.fov_range = fov_range  # simulation units
        self.facing = 0             # in radians
        self.current_time = 0       # simulation time

    def move(self, delta_time):
        if not self.path or self.current_target >= len(self.path):
            return
        target = self.path[self.current_target]
        target_point = target[0]
        direction = math.atan2(target_point.y - self.position.y, target_point.x - self.position.x)
        self.facing = direction
        distance_to_target = self.position.distance(target_point)
        if isinstance(target, tuple):
            target_time = target[1]
            time_difference = target_time - self.current_time
            if time_difference > 0:
                required_speed = distance_to_target / max(time_difference, 0.001)
                self.speed = min(required_speed, SPEED)
            else:
                self.speed = SPEED
        else:
            self.speed = SPEED
        distance_to_move = self.speed * delta_time
        new_position = clamp_point(Point(self.position.x + distance_to_move * math.cos(direction),
                                          self.position.y + distance_to_move * math.sin(direction)))
        if new_position.distance(target_point) > distance_to_target:
            excess = new_position.distance(target_point) - distance_to_target
            self.position = target_point
            if self.forward:
                if self.current_target + 1 < len(self.path):
                    self.current_target += 1
                    next_target = self.path[self.current_target]
                    next_point = next_target[0]
                    dnext = math.atan2(next_point.y - self.position.y, next_point.x - self.position.x)
                    self.position = clamp_point(Point(self.position.x + excess * math.cos(dnext),
                                                      self.position.y + excess * math.sin(dnext)))
                else:
                    self.forward = False
            else:
                if self.current_target - 1 >= 0:
                    self.current_target -= 1
                    next_target = self.path[self.current_target]
                    next_point = next_target[0]
                    dprev = math.atan2(next_point.y - self.position.y, next_point.x - self.position.x)
                    self.position = clamp_point(Point(self.position.x + excess * math.cos(dprev),
                                                      self.position.y + excess * math.sin(dprev)))
                else:
                    self.forward = True
        else:
            self.position = new_position
        if isinstance(target, tuple):
            self.current_time += delta_time

    def predict_state(self, t, last_time=0):
        if not self.path or len(self.path) == 0:
            return self.position, self.facing
        current_time = 0
        for i in range(len(self.path) - 1):
            p0, t0 = self.path[i]
            p1, t1 = self.path[i+1]
            if t0 <= t <= t1:
                fraction = (t - t0) / (t1 - t0)
                x = p0.x + fraction * (p1.x - p0.x)
                y = p0.y + fraction * (p1.y - p0.y)
                facing = math.atan2(p1.y - p0.y, p1.x - p0.x)
                return clamp_point(Point(x, y)), facing
        return self.path[-1][0], 0

# -------------------------------
# Arbiter Class (Robot Controller)
# -------------------------------
class Arbiter(MovingEntity):
    def __init__(self, init_pos, init_speed):
        super().__init__(init_pos, init_speed, [])
        self.current_time = 0
        self.time_error = 0

    def set_path(self, path):
        if path:
            self.current_time = path[0][1]
            self.position = path[0][0]
            self.path = path
            self.current_target = 0
        self.time_error = 0

    def move(self, delta_time):
        super().move(delta_time)

# -------------------------------
# Environment Class
# -------------------------------
class Environment:
    def __init__(self, map_name="simple", plot_update_callback=None):
        self.walls = WALL_CONFIGS.get(map_name, WALL_CONFIGS["simple"])
        # Time-stamp the enemy's path.
        start_enemy = Point(GRID_SIZE/2, GRID_SIZE/2)
        enemy_second = Point(GRID_SIZE/2, GRID_SIZE - 2)
        enemy_third = Point(GRID_SIZE/2, 2)
        enemy_speed = SPEED/2
        time_to_second = start_enemy.distance(enemy_second) / enemy_speed
        time_to_third = enemy_second.distance(enemy_third) / enemy_speed
        self.enemy_path = [
            (start_enemy, 0),
            (enemy_second, time_to_second),
            (enemy_third, time_to_second + time_to_third)
        ]
        self.enemy = MovingEntity(clamp_point(start_enemy), enemy_speed, self.enemy_path, fov_angle=60, fov_range=5)
        self.arbiter = Arbiter(START_POSITION, SPEED)
        path = PRMPlanner.plan(START_POSITION, FLAG_POSITION, self.walls, self.enemy,
                                n_samples=300, connection_radius=10, tolerance=1.0)
        if path:
            self.arbiter.set_path(path)
        self.last_time = time.time()
        self.state = "to_flag"  # "to_flag" or "to_start"
        self.pausing = False
        self.screen = pygame.Surface((SIM_WIDTH, SIM_HEIGHT))
        self.font = pygame.font.Font(None, 36)
        self.plot_update_callback = plot_update_callback
        self.frame_counter = 0

    def predict_enemy_position(self, t, last_time=0):
        current_time = 0
        current_segment = 0
        path = self.enemy_path
        while current_segment < len(path) - 1:
            start, _ = path[current_segment]
            end, _ = path[current_segment + 1]
            seg_dist = start.distance(end)
            seg_time = seg_dist / self.enemy.speed
            if current_time + seg_time >= t + last_time:
                fraction = (t + last_time - current_time) / seg_time
                x = start.x + fraction * (end.x - start.x)
                y = start.y + fraction * (end.y - start.y)
                return clamp_point(Point(x, y))
            current_time += seg_time
            current_segment += 1
            if current_segment == len(path) - 1:
                path = path[::-1]
                current_segment = 0
        return clamp_point(path[-1][0])

    def update_occupancy_grid(self):
        self.frame_counter = (self.frame_counter + 1) % 5
        if self.frame_counter != 0:
            return
        self.rrt_occupancy = np.zeros((GRID_SIZE, GRID_SIZE, TIME_RESOLUTION))
        current_time = self.arbiter.current_time
        for t in range(TIME_RESOLUTION):
            shifted_time = current_time + t * TIME_STEP
            enemy_pos = self.predict_enemy_position(shifted_time)
            x_idx = int(enemy_pos.x)
            y_idx = int(enemy_pos.y)
            if 0 <= x_idx < GRID_SIZE and 0 <= y_idx < GRID_SIZE:
                self.rrt_occupancy[x_idx, y_idx, t] = 1

    def replan_to_start(self):
        new_start = self.arbiter.position
        def planning_thread():
            path = PRMPlanner.plan(new_start, START_POSITION, self.walls, self.enemy,
                                     n_samples=300, connection_radius=10, tolerance=1.0)
            if path:
                self.root.after(0, lambda: self.arbiter.set_path(path))
        threading.Thread(target=planning_thread, daemon=True).start()
        self.state = "to_start"
        self.pausing = False

    def replan_to_flag(self):
        new_start = self.arbiter.position
        def planning_thread():
            path = PRMPlanner.plan(new_start, FLAG_POSITION, self.walls, self.enemy,
                                     n_samples=300, connection_radius=10, tolerance=1.0)
            if path:
                self.root.after(0, lambda: self.arbiter.set_path(path))
        threading.Thread(target=planning_thread, daemon=True).start()
        self.state = "to_flag"
        self.pausing = False

    def update(self):
        self.update_occupancy_grid()
        current_time = time.time()
        delta_time = current_time - self.last_time
        self.last_time = current_time

        self.enemy.move(delta_time)
        if not self.pausing:
            self.arbiter.move(delta_time)

        if not self.pausing:
            if self.state == "to_flag" and self.arbiter.position.distance(FLAG_POSITION) < 0.1:
                self.pausing = True
                if self.plot_update_callback:
                    self.plot_update_callback()
                self.root.after(0, self.replan_to_start)
            elif self.state == "to_start" and self.arbiter.position.distance(START_POSITION) < 0.1:
                self.pausing = True
                if self.plot_update_callback:
                    self.plot_update_callback()
                self.root.after(0, self.replan_to_flag)

    def draw(self):
        self.screen.fill((255, 255, 255))
        for wall in self.walls:
            pts = [(int(x * 20), int(y * 20)) for x, y in wall.exterior.coords]
            pygame.draw.polygon(self.screen, (128, 128, 128), pts)
        pygame.draw.circle(self.screen, (0, 255, 0), (int(FLAG_POSITION.x * 20), int(FLAG_POSITION.y * 20)), 10)
        pygame.draw.circle(self.screen, (255, 255, 0), (int(START_POSITION.x * 20), int(START_POSITION.y * 20)), 10)
        pygame.draw.circle(self.screen, (255, 0, 0), (int(self.enemy.position.x * 20), int(self.enemy.position.y * 20)), 10)
        facing = self.enemy.facing
        left_angle = facing - math.radians(self.enemy.fov_angle / 2)
        right_angle = facing + math.radians(self.enemy.fov_angle / 2)
        left_point = (self.enemy.position.x + self.enemy.fov_range * math.cos(left_angle),
                      self.enemy.position.y + self.enemy.fov_range * math.sin(left_angle))
        right_point = (self.enemy.position.x + self.enemy.fov_range * math.cos(right_angle),
                       self.enemy.position.y + self.enemy.fov_range * math.sin(right_angle))
        enemy_screen = (int(self.enemy.position.x * 20), int(self.enemy.position.y * 20))
        left_screen = (int(left_point[0] * 20), int(left_point[1] * 20))
        right_screen = (int(right_point[0] * 20), int(right_point[1] * 20))
        pygame.draw.polygon(self.screen, (255, 200, 200), [enemy_screen, left_screen, right_screen])
        pygame.draw.circle(self.screen, (0, 0, 255), (int(self.arbiter.position.x * 20), int(self.arbiter.position.y * 20)), 10)
        speed_text = self.font.render(f"Speed: {self.arbiter.speed:.2f}", True, (0, 0, 0))
        self.screen.blit(speed_text, (10, 10))

    def get_rrt3d_data(self):
        tree_lines = []
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
        self.after(int(1000 / FPS), self.update_simulation)

class MatplotlibFrame(tk.Frame):
    def __init__(self, parent, env):
        super().__init__(parent)
        self.env = env
        self.figure = plt.figure(figsize=(7, 7))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.update_plot()

    def update_plot(self):
        self.figure.clf()
        ax = self.figure.add_subplot(111, projection="3d")
        ax.set_title("PRM Roadmap and Chosen Path")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Time (rel)")
        ax.set_ylim(GRID_SIZE, 0)
        tree_lines, path = self.env.get_rrt3d_data()
        if path:
            start_time = path[0][1]
            xs = [pt[0].x for pt in path]
            ys = [pt[0].y for pt in path]
            ts = [t - start_time for (_, t) in path]
            ax.plot(xs, ys, ts, c="blue", linewidth=3, label="Chosen Path")
        t_vals = np.linspace(0, TIME_HORIZON, 100)
        ax.plot([START_POSITION.x]*100, [START_POSITION.y]*100, t_vals, c="purple", linestyle="--", linewidth=2, label="Start")
        ax.plot([FLAG_POSITION.x]*100, [FLAG_POSITION.y]*100, t_vals, c="green", linestyle="--", linewidth=2, label="Goal")
        ax.legend()
        self.canvas.draw()

# -------------------------------
# Main Application Window (Tkinter)
# -------------------------------
class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PRM-based Simulation with Embedded Pygame and Matplotlib (Tkinter)")
        self.geometry("1800x1000")
        self.env = Environment("simple")
        self.env.root = self
        sim_frame = SimulationFrame(self, self.env)
        sim_frame.pack(side="left", fill="both", expand=True)
        mpl_frame = MatplotlibFrame(self, self.env)
        mpl_frame.pack(side="left", fill="both", expand=True)
        self.env.plot_update_callback = mpl_frame.update_plot
        self.bind("<Control-c>", lambda event: self.destroy())

# -------------------------------
# Main Function
# -------------------------------
def main():
    app = MainWindow()
    app.mainloop()

if __name__ == "__main__":
    main()
