import pygame
import threading
import random
import time
import math
import numpy as np
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Constants
GRID_SIZE = 25
FPS = 60
FRAME_DELAY = 1 / FPS
MAX_ITER = 1000
SPEED = 5
TIME_STEP = 0.1  # Time resolution for 3D grid
TIME_HORIZON = 20  # Total time horizon in seconds
TIME_RESOLUTION = int(TIME_HORIZON / TIME_STEP)  # Number of time steps
FLAG_POSITION = Point(GRID_SIZE - 2, GRID_SIZE - 2)
START_POSITION = Point(2, 2)

# Define different wall configurations
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
        Polygon([(10, 10), (15, 10), (15, 15), (10, 15)]),  # Obstacle in the middle
        Polygon([(5, 20), (10, 20), (10, 25), (5, 25)])     # Another obstacle
    ],
    "maze": [
        Polygon([(0, 0), (GRID_SIZE, 0), (GRID_SIZE, 1), (0, 1)]),
        Polygon([(0, GRID_SIZE - 1), (GRID_SIZE, GRID_SIZE - 1), (GRID_SIZE, GRID_SIZE), (0, GRID_SIZE)]),
        Polygon([(0, 0), (1, 0), (1, GRID_SIZE), (0, GRID_SIZE)]),
        Polygon([(GRID_SIZE - 1, 0), (GRID_SIZE, 0), (GRID_SIZE, GRID_SIZE), (GRID_SIZE - 1, GRID_SIZE)]),
        Polygon([(5, 5), (10, 5), (10, 10), (5, 10)]),  # Maze-like walls
        Polygon([(15, 5), (20, 5), (20, 10), (15, 10)]),
        Polygon([(5, 15), (10, 15), (10, 20), (5, 20)]),
        Polygon([(15, 15), (20, 15), (20, 20), (15, 20)])
    ]
}

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((800, 800))  # Pygame window (800x800)
clock = pygame.time.Clock()

# Node and RRT classes (unchanged)
class Node:
    def __init__(self, point, parent=None):
        self.point = point
        self.parent = parent

class Node3D:
    def __init__(self, point, time, parent=None):
        self.point = point
        self.time = time
        self.parent = parent

class RRT:
    def __init__(self, start, goal, obstacles, max_iter=MAX_ITER, step_size=1.0):
        self.start = Node(start)
        self.goal = Node(goal)
        self.obstacles = obstacles
        self.max_iter = max_iter
        self.step_size = step_size
        self.nodes = [self.start]

    def generate_random_point(self):
        return Point(random.uniform(0, GRID_SIZE), random.uniform(0, GRID_SIZE))

    def nearest_node(self, point):
        return min(self.nodes, key=lambda node: node.point.distance(point))

    def new_point(self, nearest, random_point):
        direction = np.arctan2(random_point.y - nearest.point.y, random_point.x - nearest.point.x)
        new_x = nearest.point.x + self.step_size * np.cos(direction)
        new_y = nearest.point.y + self.step_size * np.sin(direction)
        return Point(new_x, new_y)

    def collision_free(self, point, buffer=0.5):
        for wall in self.obstacles:
            if wall.distance(point) < buffer:
                return False
        return True

    def plan(self):
        for _ in range(self.max_iter):
            random_point = self.generate_random_point()
            nearest = self.nearest_node(random_point)
            new_point = self.new_point(nearest, random_point)

            if self.collision_free(new_point):
                new_node = Node(new_point, nearest)
                self.nodes.append(new_node)

                if new_point.distance(self.goal.point) <= self.step_size:
                    self.goal.parent = new_node
                    return self.reconstruct_path(self.goal)

        return None

    def reconstruct_path(self, node):
        path = []
        while node:
            path.append(node.point)
            node = node.parent
        return path[::-1]

class RRT3D(RRT):
    def __init__(self, start, goal, obstacles, max_iter=MAX_ITER, step_size=1.0):
        super().__init__(start, goal, obstacles, max_iter, step_size)
        self.occupancy_grid = np.zeros((GRID_SIZE, GRID_SIZE, TIME_RESOLUTION))
        self.lock = threading.Lock()  # Thread-safe lock for occupancy grid

    def generate_random_point(self):
        x = random.uniform(0, GRID_SIZE)
        y = random.uniform(0, GRID_SIZE)
        t = random.uniform(0, TIME_HORIZON)
        return Point(x, y), t

    def nearest_node(self, point, time):
        return min(self.nodes, key=lambda node: np.sqrt((node.point.x - point.x)**2 + (node.point.y - point.y)**2 + (node.time - time)**2))

    def new_point(self, nearest, random_point, random_time):
        direction = np.arctan2(random_point.y - nearest.point.y, random_point.x - nearest.point.x)
        new_x = nearest.point.x + self.step_size * np.cos(direction)
        new_y = nearest.point.y + self.step_size * np.sin(direction)
        new_time = nearest.time + TIME_STEP
        return Point(new_x, new_y), new_time

    def collision_free(self, point, time):
        x_idx = int(point.x)
        y_idx = int(point.y)
        t_idx = int(time / TIME_STEP)
        if x_idx < 0 or x_idx >= GRID_SIZE or y_idx < 0 or y_idx >= GRID_SIZE or t_idx < 0 or t_idx >= TIME_RESOLUTION:
            return False
        return self.occupancy_grid[x_idx, y_idx, t_idx] == 0

    def plan(self):
        for _ in range(self.max_iter):
            random_point, random_time = self.generate_random_point()
            nearest = self.nearest_node(random_point, random_time)
            new_point, new_time = self.new_point(nearest, random_point, random_time)

            if self.collision_free(new_point, new_time):
                new_node = Node3D(new_point, new_time, nearest)
                self.nodes.append(new_node)

                if new_point.distance(self.goal.point) <= self.step_size and abs(new_time - TIME_HORIZON) <= TIME_STEP:
                    self.goal.parent = new_node
                    return self.reconstruct_path(self.goal)

        return None

    def reconstruct_path(self, node):
        path = []
        while node:
            path.append((node.point, node.time))
            node = node.parent
        return path[::-1]

# MovingEntity and FriendlyBot classes (unchanged)
class MovingEntity:
    def __init__(self, init_pos, init_speed, path):
        self.position = init_pos
        self.speed = init_speed
        self.path = path
        self.current_target = 0
        self.forward = True

    def move(self, delta_time):
        if not self.path or self.current_target >= len(self.path):
            return

        target = self.path[self.current_target]
        direction = np.arctan2(target.y - self.position.y, target.x - self.position.x)
        distance_to_move = self.speed * delta_time

        # Compute new position
        new_x = self.position.x + distance_to_move * np.cos(direction)
        new_y = self.position.y + distance_to_move * np.sin(direction)
        new_position = Point(new_x, new_y)

        # Calculate the distance to the target
        distance_to_target = self.position.distance(target)

        # If we overshoot the target
        if new_position.distance(target) > distance_to_target:
            # Calculate the excess distance
            excess_distance = new_position.distance(target) - distance_to_target

            # Move to the target
            self.position = target

            # If there's a next target, apply the excess distance to it
            if self.forward:
                if self.current_target + 1 < len(self.path):
                    self.current_target += 1
                    next_target = self.path[self.current_target]
                    direction_to_next = np.arctan2(next_target.y - self.position.y, next_target.x - self.position.x)
                    self.position = Point(
                        self.position.x + excess_distance * np.cos(direction_to_next),
                        self.position.y + excess_distance * np.sin(direction_to_next))
                else:
                    self.forward = False
            else:
                if self.current_target - 1 >= 0:
                    self.current_target -= 1
                    next_target = self.path[self.current_target]
                    direction_to_next = np.arctan2(next_target.y - self.position.y, next_target.x - self.position.x)
                    self.position = Point(
                        self.position.x + excess_distance * np.cos(direction_to_next),
                        self.position.y + excess_distance * np.sin(direction_to_next))
                else:
                    self.forward = True
        else:
            self.position = new_position

class FriendlyBot(MovingEntity):
    def __init__(self, init_pos, init_speed):
        super().__init__(init_pos, init_speed, [])

    def set_path(self, path):
        self.path = path
        self.current_target = 0

# Environment class with threading
class Environment:
    def __init__(self, map_name="simple"):
        self.walls = WALL_CONFIGS.get(map_name, WALL_CONFIGS["simple"])  # Default to "simple" if map_name is invalid
        self.enemy_path = [Point(GRID_SIZE // 2, GRID_SIZE // 2), Point(GRID_SIZE // 2, GRID_SIZE - 2), Point(GRID_SIZE // 2, GRID_SIZE // 2)]
        self.enemy = MovingEntity(Point(GRID_SIZE // 2, GRID_SIZE // 2), SPEED, self.enemy_path)
        self.friendly_bot = FriendlyBot(START_POSITION, SPEED)
        self.rrt = RRT(START_POSITION, FLAG_POSITION, self.walls)
        self.rrt3d = RRT3D(START_POSITION, FLAG_POSITION, self.walls)
        self.friendly_bot.set_path(self.rrt.plan())
        self.last_time = time.time()
        self.running = True
        self.state = "to_flag"

    def update_occupancy_grid(self):
        # Predict enemy positions and update the occupancy grid
        for t in range(TIME_RESOLUTION):
            enemy_pos = self.predict_enemy_position(t * TIME_STEP)
            x_idx = int(enemy_pos.x)
            y_idx = int(enemy_pos.y)
            if 0 <= x_idx < GRID_SIZE and 0 <= y_idx < GRID_SIZE:
                with self.rrt3d.lock:  # Thread-safe access
                    self.rrt3d.occupancy_grid[x_idx, y_idx, t] = 1

    def predict_enemy_position(self, time):
        # Simple linear prediction based on the enemy's path
        path_length = len(self.enemy_path)
        segment = int(time / (TIME_HORIZON / path_length))
        if segment >= path_length - 1:
            return self.enemy_path[-1]
        t_segment = time % (TIME_HORIZON / path_length)
        start = self.enemy_path[segment]
        end = self.enemy_path[segment + 1]
        direction = np.arctan2(end.y - start.y, end.x - start.x)
        distance = t_segment * SPEED
        return Point(start.x + distance * np.cos(direction), start.y + distance * np.sin(direction))

    def update(self):
        current_time = time.time()
        delta_time = current_time - self.last_time
        self.last_time = current_time

        self.enemy.move(delta_time)
        self.friendly_bot.move(delta_time)

        if self.state == "to_flag" and self.friendly_bot.position.distance(FLAG_POSITION) < 0.001:
            self.state = "to_start"
            self.rrt = RRT(FLAG_POSITION, START_POSITION, self.walls)
            self.friendly_bot.set_path(self.rrt.plan())
        elif self.state == "to_start" and self.friendly_bot.position.distance(START_POSITION) < 0.001:
            self.state = "to_flag"
            self.rrt = RRT(START_POSITION, FLAG_POSITION, self.walls)
            self.friendly_bot.set_path(self.rrt.plan())

    def draw(self):
        screen.fill((255, 255, 255))

        # Draw walls
        for wall in self.walls:
            points = [(int(x * 20), int(y * 20)) for x, y in wall.exterior.coords]
            pygame.draw.polygon(screen, (128, 128, 128), points)

        # Draw flag and start
        pygame.draw.circle(screen, (0, 255, 0), (int(FLAG_POSITION.x * 20), int(FLAG_POSITION.y * 20)), 10)
        pygame.draw.circle(screen, (255, 255, 0), (int(START_POSITION.x * 20), int(START_POSITION.y * 20)), 10)

        # Draw entities
        pygame.draw.circle(screen, (255, 0, 0), (int(self.enemy.position.x * 20), int(self.enemy.position.y * 20)), 10)
        pygame.draw.circle(screen, (0, 0, 255), (int(self.friendly_bot.position.x * 20), int(self.friendly_bot.position.y * 20)), 10)

        pygame.display.flip()

    def run_pygame(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.update_occupancy_grid()
            self.update()
            self.draw()
            clock.tick(FPS)

    # Update run_matplotlib to use 3D plotting
    def run_matplotlib(self):
        plt.ion()  # Interactive mode
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # Create a 3D axis
        ax.set_title("3D Occupancy Grid")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Time")

        while self.running:
            with self.rrt3d.lock:  # Thread-safe access
                occupancy_grid = self.rrt3d.occupancy_grid.copy()

            ax.clear()
            ax.set_title("3D Occupancy Grid")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Time")

            # Plot the 3D occupancy grid
            x, y, t = np.where(occupancy_grid == 1)
            ax.scatter(x, y, t, c="red", marker="o")

            plt.draw()
            plt.pause(0.1)


# Main function
if __name__ == "__main__":
    env = Environment(map_name="maze")

    # Start Pygame and Matplotlib in separate threads
    pygame_thread = threading.Thread(target=env.run_pygame)
    matplotlib_thread = threading.Thread(target=env.run_matplotlib)

    pygame_thread.start()
    matplotlib_thread.start()

    pygame_thread.join()
    matplotlib_thread.join()

    pygame.quit()