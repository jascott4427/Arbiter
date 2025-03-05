import pygame
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
screen = pygame.display.set_mode((1600, 800))  # Wider window (1600x800)
clock = pygame.time.Clock()

# Node and RRT classes
class Node:
    def __init__(self, point, parent=None):
        self.point = point
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
        # Check if the point is too close to any wall
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

# MovingEntity and FriendlyBot classes
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

# Environment class with tree plotting
class Environment:
    def __init__(self, map_name="simple"):
        self.walls = WALL_CONFIGS.get(map_name, WALL_CONFIGS["simple"])  # Default to "simple" if map_name is invalid
        self.enemy_path = [Point(GRID_SIZE // 2, GRID_SIZE // 2), Point(GRID_SIZE // 2, GRID_SIZE - 2), Point(GRID_SIZE // 2, GRID_SIZE // 2)]
        self.enemy = MovingEntity(Point(GRID_SIZE // 2, GRID_SIZE // 2), SPEED, self.enemy_path)
        self.friendly_bot = FriendlyBot(START_POSITION, SPEED)
        self.rrt = RRT(START_POSITION, FLAG_POSITION, self.walls)
        self.friendly_bot.set_path(self.rrt.plan())
        self.last_time = time.time()
        self.running = True
        self.state = "to_flag"

        # Matplotlib setup for tree plotting
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvas(self.fig)
        self.tree_surface = None

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

    def draw_tree(self):
        self.ax.clear()
        self.ax.set_xlim(0, GRID_SIZE)
        self.ax.set_ylim(GRID_SIZE, 0)

        # Draw walls
        for wall in self.walls:
            x, y = wall.exterior.xy
            self.ax.fill(x, y, alpha=0.5, fc='gray', ec='black')

        # Draw tree
        for node in self.rrt.nodes:
            if node.parent:
                self.ax.plot([node.point.x, node.parent.point.x], [node.point.y, node.parent.point.y], 'b-')

        # Draw start and goal
        self.ax.plot(START_POSITION.x, START_POSITION.y, 'go', markersize=10, label='Start')
        self.ax.plot(FLAG_POSITION.x, FLAG_POSITION.y, 'ro', markersize=10, label='Flag')

        # Render the figure to a Pygame surface
        self.canvas.draw()
        renderer = self.canvas.get_renderer()
        raw_data = renderer.buffer_rgba()
        size = self.canvas.get_width_height()
        self.tree_surface = pygame.image.fromstring(bytes(raw_data), size, 'RGBA')

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

        # Draw the tree
        if self.tree_surface:
            screen.blit(self.tree_surface, (500, -50))  # Blit the tree to the right side

        pygame.display.flip()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.update()
            self.draw_tree()  # Update the tree plot
            self.draw()
            clock.tick(FPS)

if __name__ == "__main__":
    # Change "simple" to "obstacles" or "maze" to switch maps
    env = Environment(map_name="maze")
    env.run()
    pygame.quit()