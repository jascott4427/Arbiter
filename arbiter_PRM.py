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
SPEED = 10                # Maximum allowed robot speed.
PLANNING_SPEED = 8        # Base speed used for time-parameterization.
TIME_STEP = 0.1           # Time resolution for occupancy grid.
TIME_HORIZON = GRID_SIZE * 3 / SPEED  # Overall time horizon for initial PRM.
TIME_RESOLUTION = int(TIME_HORIZON / TIME_STEP)
# When replanning (upon reaching target), use a shorter horizon.
REPLAN_HORIZON = 5.0      
FLAG_POSITION = Point(GRID_SIZE - 2, GRID_SIZE - 2)
START_POSITION = Point(2, 2)
PLAYER_RADIUS = 0.5
OCCUPANCY_RADIUS = 2
DT_EPSILON = 0.01         # Minimum allowed time difference for a forward edge.
REPLAN_THRESHOLD = 0.2    # When robot is within this distance of its target, switch targets.
NUM_NODES = 150           # Base number of nodes.
# Parameters for enemy FOV avoidance (used in cost function during planning):
ENEMY_BUFFER = 2.0        # Extra distance buffer.
FOV_MARGIN = 15           # Extra angular margin in degrees.
PENALTY_FACTOR = 250       # Weight for enemy avoidance cost.
MAX_PLANNING_ATTEMPTS = 3  # Maximum planning attempts if no path is found.

# -------------------------------
# Helper: Clamp a Point to the Grid
# -------------------------------
def clamp_point(point, min_val=0, max_val=GRID_SIZE):
    return Point(max(min_val, min(point.x, max_val)), max(min_val, min(point.y, max_val)))

# -------------------------------
# Wall Configurations
# -------------------------------
WALL_CONFIGS = {
    "simple": [
        Polygon([(0,0), (GRID_SIZE,0), (GRID_SIZE,1), (0,1)]),
        Polygon([(0,GRID_SIZE-1), (GRID_SIZE,GRID_SIZE-1), (GRID_SIZE,GRID_SIZE), (0,GRID_SIZE)]),
        Polygon([(0,0), (1,0), (1,GRID_SIZE), (0,GRID_SIZE)]),
        Polygon([(GRID_SIZE-1,0), (GRID_SIZE,0), (GRID_SIZE,GRID_SIZE), (GRID_SIZE-1,GRID_SIZE)])
    ],
    "obstacles": [
        Polygon([(0,0), (GRID_SIZE,0), (GRID_SIZE,1), (0,1)]),
        Polygon([(0,GRID_SIZE-1), (GRID_SIZE,GRID_SIZE-1), (GRID_SIZE,GRID_SIZE), (0,GRID_SIZE)]),
        Polygon([(0,0), (1,0), (1,GRID_SIZE), (0,GRID_SIZE)]),
        Polygon([(GRID_SIZE-1,0), (GRID_SIZE,0), (GRID_SIZE,GRID_SIZE), (GRID_SIZE-1,GRID_SIZE)]),
        Polygon([(10,10), (15,10), (15,15), (10,15)]),
        Polygon([(5,20), (10,20), (10,25), (5,25)])
    ],
    "maze": [
        Polygon([(0,0), (GRID_SIZE,0), (GRID_SIZE,1), (0,1)]),
        Polygon([(0,GRID_SIZE-1), (GRID_SIZE,GRID_SIZE-1), (GRID_SIZE,GRID_SIZE), (0,GRID_SIZE)]),
        Polygon([(0,0), (1,0), (1,GRID_SIZE), (0,GRID_SIZE)]),
        Polygon([(GRID_SIZE-1,0), (GRID_SIZE,0), (GRID_SIZE,GRID_SIZE), (GRID_SIZE-1,GRID_SIZE)]),
        Polygon([(5,5), (10,5), (10,10), (5,10)]),
        Polygon([(15,5), (20,5), (20,10), (15,10)]),
        Polygon([(5,15), (10,15), (10,20), (5,20)]),
        Polygon([(15,15), (20,15), (20,20), (15,20)])
    ]
}

# -------------------------------
# 3D PRM Node Class (x, y, time)
# -------------------------------
class Node:
    def __init__(self, point, parent=None, time=None):
        self.point = point
        self.parent = parent
        self.time = time

# -------------------------------
# Helper: Compute Angle Between Two Points
# -------------------------------
def compute_angle(p1, p2):
    return math.atan2(p2.y - p1.y, p2.x - p1.x)

# -------------------------------
# Reparameterize Path (with curvature adjustments)
# -------------------------------
def reparameterize_path(path, planning_speed):
    new_path = []
    t = 0
    new_path.append((path[0][0], t))
    for i in range(1, len(path)):
        d = path[i-1][0].distance(path[i][0])
        seg_speed = planning_speed
        if i > 1:
            angle1 = compute_angle(path[i-2][0], path[i-1][0])
            angle2 = compute_angle(path[i-1][0], path[i][0])
            angle_diff = abs(angle2 - angle1)
            angle_diff = min(angle_diff, 2*math.pi - angle_diff)
            if angle_diff > math.pi/8:
                reduction = max(0.5, 1 - (angle_diff - math.pi/8) / (math.pi/2))
                seg_speed = planning_speed * reduction
        dt = d / seg_speed if seg_speed > 0 else 0
        t += dt
        new_path.append((path[i][0], t))
    return new_path

# -------------------------------
# PRM3D Class (3D: x, y, t) with Cost Function for Enemy Avoidance
# -------------------------------
import heapq

class PRM3D:
    def __init__(self, start, goal, obstacles, enemy=None, num_nodes=NUM_NODES, K=10, step_size=1.0, max_speed=SPEED/2, time_horizon=TIME_HORIZON):
        self.time_horizon = time_horizon
        self.start = Node(start, None, 0)
        self.goal = Node(goal, None, time_horizon)
        self.obstacles = obstacles
        self.enemy = enemy
        self.num_nodes = num_nodes
        self.K = K
        self.step_size = step_size
        self.max_speed = max_speed
        self.nodes = []
        # Use fixed global TIME_RESOLUTION for occupancy grid.
        self.occupancy_grid = np.zeros((GRID_SIZE, GRID_SIZE, TIME_RESOLUTION))
        self.lock = threading.Lock()
        self.graph = {}

    def generate_random_point(self):
        x = random.uniform(0, GRID_SIZE)
        y = random.uniform(0, GRID_SIZE)
        t = random.uniform(0, self.time_horizon)
        return Point(x, y), t

    def collision_free(self, point, time):
        for wall in self.obstacles:
            if wall.distance(point) < PLAYER_RADIUS:
                return False
        x_idx = round(point.x)
        y_idx = round(point.y)
        t_idx = round(time / TIME_STEP)
        if not (0 <= x_idx < GRID_SIZE and 0 <= y_idx < GRID_SIZE and 0 <= t_idx < TIME_RESOLUTION):
            return False
        if self.occupancy_grid[x_idx, y_idx, t_idx] == 1:
            return False
        radius_in_cells = math.ceil(OCCUPANCY_RADIUS / GRID_SIZE)
        for dx in range(-radius_in_cells, radius_in_cells+1):
            for dy in range(-radius_in_cells, radius_in_cells+1):
                for dt in range(-radius_in_cells, radius_in_cells+1):
                    d = math.sqrt((dx * GRID_SIZE)**2 + (dy * GRID_SIZE)**2 + (dt * TIME_STEP)**2)
                    if d <= OCCUPANCY_RADIUS:
                        if (0 <= x_idx+dx < GRID_SIZE and 0 <= y_idx+dy < GRID_SIZE and 0 <= t_idx+dt < TIME_RESOLUTION):
                            if self.occupancy_grid[x_idx+dx, y_idx+dy, t_idx+dt] == 1:
                                return False
        if self.enemy is not None:
            enemy_pos = self.enemy.position
            dx = point.x - enemy_pos.x
            dy = point.y - enemy_pos.y
            dist = math.sqrt(dx**2 + dy**2)
            if dist <= self.enemy.fov_range + ENEMY_BUFFER:
                angle_to_point = math.degrees(math.atan2(dy, dx))
                enemy_facing = math.degrees(self.enemy.facing)
                angle_diff = (angle_to_point - enemy_facing + 180) % 360 - 180
                if abs(angle_diff) <= self.enemy.fov_angle/2 + FOV_MARGIN:
                    return False
        return True

    def is_edge_collision_free(self, node1, node2):
        if node2.time < node1.time + DT_EPSILON:
            return False
        m = 10
        for i in range(1, m+1):
            fraction = i / (m+1)
            x = node1.point.x + fraction * (node2.point.x - node1.point.x)
            y = node1.point.y + fraction * (node2.point.y - node1.point.y)
            t = node1.time + fraction * (node2.time - node1.time)
            if not self.collision_free(Point(x, y), t):
                return False
        return True

    def edge_cost(self, node1, node2):
        base = math.sqrt((node1.point.x - node2.point.x)**2 +
                         (node1.point.y - node2.point.y)**2 +
                         (node1.time - node2.time)**2)
        mid_time = (node1.time + node2.time) / 2
        mid_x = (node1.point.x + node2.point.x) / 2
        mid_y = (node1.point.y + node2.point.y) / 2
        mid_point = Point(mid_x, mid_y)
        penalty = 0
        if self.enemy is not None:
            enemy_pred = self.enemy.predict_state(mid_time)[0]
            d_enemy = mid_point.distance(enemy_pred)
            threshold = self.enemy.fov_range + ENEMY_BUFFER
            if d_enemy < threshold:
                penalty = PENALTY_FACTOR * (threshold - d_enemy)**2
        return base + penalty

    def distance(self, node1, node2):
        return math.sqrt((node1.point.x - node2.point.x)**2 +
                         (node1.point.y - node2.point.y)**2 +
                         (node1.time - node2.time)**2)

    def build_graph(self):
        n = len(self.nodes)
        self.graph = { i: [] for i in range(n) }
        for i in range(n):
            neighbors = []
            for j in range(n):
                if i == j:
                    continue
                if self.nodes[j].time < self.nodes[i].time + DT_EPSILON:
                    continue
                cost = self.edge_cost(self.nodes[i], self.nodes[j])
                neighbors.append((j, cost))
            neighbors.sort(key=lambda x: x[1])
            count = 0
            for (j, cost) in neighbors:
                if count >= self.K:
                    break
                if self.is_edge_collision_free(self.nodes[i], self.nodes[j]):
                    self.graph[i].append((j, cost))
                    count += 1

    def a_star(self, start_index, goal_index):
        open_set = []
        heapq.heappush(open_set, (0, start_index))
        came_from = {}
        g_score = { i: float('inf') for i in range(len(self.nodes)) }
        g_score[start_index] = 0
        f_score = { i: float('inf') for i in range(len(self.nodes)) }
        f_score[start_index] = self.distance(self.nodes[start_index], self.nodes[goal_index])
        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == goal_index:
                return self.reconstruct_path(came_from, current)
            for neighbor, weight in self.graph.get(current, []):
                tentative_g = g_score[current] + weight
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.distance(self.nodes[neighbor], self.nodes[goal_index])
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path

    def plan(self):
        attempt = 0
        node_multiplier = 1.0
        while attempt < MAX_PLANNING_ATTEMPTS:
            self.nodes = []
            self.nodes.append(self.start)
            self.nodes.append(self.goal)
            target_nodes = int(NUM_NODES * node_multiplier)
            while len(self.nodes) < target_nodes:
                p, t = self.generate_random_point()
                if self.collision_free(p, t):
                    self.nodes.append(Node(p, None, t))
            self.nodes.sort(key=lambda node: node.time)
            self.build_graph()
            start_index = self.nodes.index(self.start)
            goal_index = self.nodes.index(self.goal)
            path_indices = self.a_star(start_index, goal_index)
            if path_indices is not None:
                base_path = [(self.nodes[i].point, self.nodes[i].time) for i in path_indices]
                new_path = reparameterize_path(base_path, planning_speed=PLANNING_SPEED)
                print("Path found with PRM3D on attempt", attempt+1)
                return new_path
            else:
                attempt += 1
                node_multiplier *= 2
        print("No path found using PRM3D after", MAX_PLANNING_ATTEMPTS, "attempts.")
        return None

# -------------------------------
# MovingEntity Class (Robot Controller)
# -------------------------------
class MovingEntity:
    def __init__(self, init_pos, init_speed, path, fov_angle=60, fov_range=5):
        self.position = init_pos
        self.speed = init_speed
        self.path = path  # List of (Point, time)
        self.current_target = 0
        self.fov_angle = fov_angle
        self.fov_range = fov_range
        self.facing = 0
        self.current_time = path[0][1] if path else 0

    def move(self, delta_time):
        if not self.path or self.current_target >= len(self.path):
            return
        target_pt, target_time = self.path[self.current_target]
        d = self.position.distance(target_pt)
        time_left = target_time - self.current_time
        desired_speed = d / time_left if time_left > 0 else 0
        self.speed = min(desired_speed, SPEED)
        move_dist = self.speed * delta_time
        if move_dist >= d:
            self.position = target_pt
            self.current_time = target_time
            self.current_target += 1
        else:
            direction = math.atan2(target_pt.y - self.position.y, target_pt.x - self.position.x)
            new_x = self.position.x + move_dist * math.cos(direction)
            new_y = self.position.y + move_dist * math.sin(direction)
            self.position = clamp_point(Point(new_x, new_y))
            self.current_time += delta_time

    def predict_state(self, t, last_time=0):
        if not self.path:
            return self.position, self.facing
        idx = min(self.current_target, len(self.path)-1)
        pt, _ = self.path[idx]
        return pt, self.facing

# -------------------------------
# Arbiter Class (Robot Controller)
# -------------------------------
class Arbiter(MovingEntity):
    def __init__(self, init_pos, init_speed):
        super().__init__(init_pos, init_speed, [])
        self.enemy = None
    def set_path(self, path):
        if path:
            self.current_time = path[0][1]
            self.position = path[0][0]
            self.path = path
            self.current_target = 0

# -------------------------------
# Enemy Class (Cyclic MovingEntity with FOV)
# -------------------------------
class Enemy(MovingEntity):
    def __init__(self, init_pos, init_speed, path, fov_angle=60, fov_range=5):
        super().__init__(init_pos, init_speed, path, fov_angle, fov_range)
        self.forward = True
        self.constant_speed = init_speed
    def move(self, delta_time):
        if not self.path or self.current_target >= len(self.path):
            return
        target_pt, target_time = self.path[self.current_target]
        d = self.position.distance(target_pt)
        move_dist = self.constant_speed * delta_time
        if move_dist >= d:
            self.position = target_pt
            self.current_time = target_time
            self.current_target += 1
            if self.current_target >= len(self.path):
                old_path = self.path
                T = old_path[-1][1]
                new_path = []
                for pt, t in reversed(old_path):
                    new_path.append((pt, T + (T - t)))
                self.path = new_path
                self.current_target = 1
        else:
            direction = math.atan2(target_pt.y - self.position.y, target_pt.x - self.position.x)
            new_x = self.position.x + move_dist * math.cos(direction)
            new_y = self.position.y + move_dist * math.sin(direction)
            self.position = clamp_point(Point(new_x, new_y))
            self.current_time += delta_time
        self.facing = math.atan2(target_pt.y - self.position.y, target_pt.x - self.position.x)

# -------------------------------
# Environment Class
# -------------------------------
class Environment:
    def __init__(self, map_name="simple", plot_update_callback=None):
        self.walls = WALL_CONFIGS.get(map_name, WALL_CONFIGS["simple"])
        start_enemy = Point(GRID_SIZE/2, GRID_SIZE/2)
        enemy_second = Point(GRID_SIZE/2, GRID_SIZE - 2)
        enemy_third = Point(GRID_SIZE/2, 2)
        enemy_speed = SPEED/2
        time_to_second = start_enemy.distance(enemy_second) / enemy_speed
        time_to_third = enemy_second.distance(enemy_third) / enemy_speed
        enemy_path = [
            (start_enemy, 0),
            (enemy_second, time_to_second),
            (enemy_third, time_to_second + time_to_third)
        ]
        self.enemy = Enemy(clamp_point(start_enemy), enemy_speed, enemy_path, fov_angle=60, fov_range=5)
        # Initially plan from START to FLAG.
        self.prm3d = PRM3D(START_POSITION, FLAG_POSITION, self.walls, enemy=self.enemy, num_nodes=NUM_NODES)
        path = self.prm3d.plan()
        if path is None:
            print("Warning: PRM3D failed, using direct connection.")
            path = [(START_POSITION, 0), (FLAG_POSITION, TIME_HORIZON)]
        self.arbiter = Arbiter(START_POSITION, SPEED)
        self.arbiter.set_path(path)
        self.arbiter.enemy = self.enemy
        self.last_time = time.time()
        # Initial state: moving from START to FLAG.
        self.state = "to_flag"
        self.target_reached = False
        self.screen = pygame.Surface((SIM_WIDTH, SIM_HEIGHT))
        self.font = pygame.font.Font(None, 36)
        self.plot_update_callback = plot_update_callback
        self.frame_counter = 0
        self.current_path = path

    def predict_enemy_position(self, t, last_time=0):
        current_time = 0
        current_segment = 0
        path = self.enemy.path
        if not path:
            return self.enemy.position
        while current_segment < len(path)-1:
            start, _ = path[current_segment]
            end, _ = path[current_segment+1]
            seg_dist = start.distance(end)
            seg_time = seg_dist / self.enemy.constant_speed if self.enemy.constant_speed > 0 else 1e9
            if current_time + seg_time >= t + last_time:
                fraction = (t + last_time - current_time) / seg_time
                x = start.x + fraction * (end.x - start.x)
                y = start.y + fraction * (end.y - start.y)
                return clamp_point(Point(x, y))
            current_time += seg_time
            current_segment += 1
        return clamp_point(path[-1][0])

    def update_occupancy_grid(self):
        # Use fixed global TIME_RESOLUTION for the occupancy grid.
        occ = np.zeros((GRID_SIZE, GRID_SIZE, TIME_RESOLUTION))
        current_time = self.enemy.current_time
        for t in range(TIME_RESOLUTION):
            shifted_time = current_time + t * TIME_STEP
            enemy_pos = self.predict_enemy_position(shifted_time)
            x_idx = int(enemy_pos.x)
            y_idx = int(enemy_pos.y)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    ix = x_idx + dx
                    iy = y_idx + dy
                    if 0 <= ix < GRID_SIZE and 0 <= iy < GRID_SIZE:
                        occ[ix, iy, t] = 1
        with self.prm3d.lock:
            self.prm3d.occupancy_grid = occ

    def update(self):
        self.update_occupancy_grid()
        current_time = time.time()
        delta_time = current_time - self.last_time
        self.last_time = current_time

        self.enemy.move(delta_time)
        self.arbiter.move(delta_time)

        # When near the target, trigger a one-time target switch.
        if self.state == "to_flag":
            if self.arbiter.position.distance(FLAG_POSITION) < REPLAN_THRESHOLD and not self.target_reached:
                self.replan(switch_target=True)
                self.target_reached = True
            elif self.arbiter.position.distance(FLAG_POSITION) > REPLAN_THRESHOLD * 2:
                self.target_reached = False
        elif self.state == "to_start":
            if self.arbiter.position.distance(START_POSITION) < REPLAN_THRESHOLD and not self.target_reached:
                self.replan(switch_target=True)
                self.target_reached = True
            elif self.arbiter.position.distance(START_POSITION) > REPLAN_THRESHOLD * 2:
                self.target_reached = False

    def replan(self, switch_target=False):
        if switch_target:
            if self.state == "to_flag":
                new_goal = START_POSITION
                self.state = "to_start"
            else:
                new_goal = FLAG_POSITION
                self.state = "to_flag"
            self.target_reached = False
        else:
            if self.state == "to_flag":
                new_goal = FLAG_POSITION
            else:
                new_goal = START_POSITION
        horizon = REPLAN_HORIZON
        prm = PRM3D(self.arbiter.position, new_goal, self.walls, enemy=self.enemy, num_nodes=NUM_NODES, time_horizon=horizon)
        new_path = prm.plan()
        if new_path is None or len(new_path) < 2:
            print("Warning: Replanning failed, using direct connection.")
            new_path = [(self.arbiter.position, self.arbiter.current_time), (new_goal, self.arbiter.current_time + horizon)]
        offset = self.arbiter.current_time
        new_path = [(pt, t + offset) for pt, t in new_path]
        self.arbiter.set_path(new_path)
        self.current_path = new_path

    def draw(self):
        self.screen.fill((255,255,255))
        for wall in self.walls:
            pts = [(int(x*20), int(y*20)) for x,y in wall.exterior.coords]
            pygame.draw.polygon(self.screen, (128,128,128), pts)
        pygame.draw.circle(self.screen, (255,255,0), (int(START_POSITION.x*20), int(START_POSITION.y*20)), 10)
        pygame.draw.circle(self.screen, (0,255,0), (int(FLAG_POSITION.x*20), int(FLAG_POSITION.y*20)), 10)
        pygame.draw.circle(self.screen, (255,0,0), (int(self.enemy.position.x*20), int(self.enemy.position.y*20)), 10)
        facing = self.enemy.facing
        left_angle = facing - math.radians(self.enemy.fov_angle/2)
        right_angle = facing + math.radians(self.enemy.fov_angle/2)
        left_point = (self.enemy.position.x + self.enemy.fov_range*math.cos(left_angle),
                      self.enemy.position.y + self.enemy.fov_range*math.sin(left_angle))
        right_point = (self.enemy.position.x + self.enemy.fov_range*math.cos(right_angle),
                       self.enemy.position.y + self.enemy.fov_range*math.sin(right_angle))
        enemy_screen = (int(self.enemy.position.x*20), int(self.enemy.position.y*20))
        left_screen = (int(left_point[0]*20), int(left_point[1]*20))
        right_screen = (int(right_point[0]*20), int(right_point[1]*20))
        pygame.draw.polygon(self.screen, (255,200,200), [enemy_screen, left_screen, right_screen])
        pygame.draw.circle(self.screen, (0,0,255), (int(self.arbiter.position.x*20), int(self.arbiter.position.y*20)), 10)
        speed_text = self.font.render(f"Speed: {self.arbiter.speed:.2f}", True, (0,0,0))
        self.screen.blit(speed_text, (10,10))
        if self.current_path:
            pts = [(int(pt[0].x*20), int(pt[0].y*20)) for pt in self.current_path]
            if len(pts) > 1:
                pygame.draw.lines(self.screen, (0,0,255), False, pts, 3)

    def get_prm3d_data(self):
        tree_lines = []
        for i, neighbors in self.prm3d.graph.items():
            for j, _ in neighbors:
                p0 = self.prm3d.nodes[i]
                p1 = self.prm3d.nodes[j]
                tree_lines.append(((p0.point.x, p0.point.y, p0.time),
                                   (p1.point.x, p1.point.y, p1.time)))
        return tree_lines, self.arbiter.path

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
        self.figure = plt.figure(figsize=(7,7))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.update_plot()
    def update_plot(self):
        self.figure.clf()
        ax = self.figure.add_subplot(111, projection="3d")
        ax.set_title("3D PRM Roadmap, Path, and Occupancy Grid")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Time")
        ax.set_ylim(GRID_SIZE, 0)
        occ = self.env.prm3d.occupancy_grid
        x_idx, y_idx, t_idx = np.where(occ == 1)
        t_vals = t_idx * TIME_STEP
        ax.scatter(x_idx, y_idx, t_vals, c="red", marker="o", s=50, label="Occupied Cells")
        tree_lines, path = self.env.get_prm3d_data()
        for (p0, p1) in tree_lines:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], c="black", linewidth=1)
        if path:
            xs = [pt[0].x for pt in path]
            ys = [pt[0].y for pt in path]
            ts = [pt[1] for pt in path]
            ax.plot(xs, ys, ts, c="blue", linewidth=3, label="Chosen Path")
        t_line = np.linspace(0, TIME_HORIZON, 100)
        ax.plot([START_POSITION.x]*100, [START_POSITION.y]*100, t_line,
                c="purple", linestyle="--", linewidth=2, label="Start")
        ax.plot([FLAG_POSITION.x]*100, [FLAG_POSITION.y]*100, t_line,
                c="green", linestyle="--", linewidth=2, label="Goal")
        ax.legend()
        self.canvas.draw()

# -------------------------------
# Main Application Window (Tkinter)
# -------------------------------
class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("3D PRM Path Planning with Fixed Target Switching")
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
