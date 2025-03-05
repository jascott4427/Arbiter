import pygame
import random
import time
import math
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import numpy as np


# Constants
GRID_SIZE = 25
FPS = 60  
FRAME_DELAY = 1 / FPS
MAX_ITER = 1000
SPEED = 5
FLAG_POSITION = Point(GRID_SIZE - 2, GRID_SIZE - 2)
START_POSITION = Point(2, 2)
WALLS = [
    Polygon([(0, 0), (GRID_SIZE, 0), (GRID_SIZE, 1), (0, 1)]),
    Polygon([(0, GRID_SIZE - 1), (GRID_SIZE, GRID_SIZE - 1), (GRID_SIZE, GRID_SIZE), (0, GRID_SIZE)]),
    Polygon([(0, 0), (1, 0), (1, GRID_SIZE), (0, GRID_SIZE)]),
    Polygon([(GRID_SIZE - 1, 0), (GRID_SIZE, 0), (GRID_SIZE, GRID_SIZE), (GRID_SIZE - 1, GRID_SIZE)])
]

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

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

    def collision_free(self, point):
        return not any(obstacle.contains(point) for obstacle in self.obstacles)

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

        # Fix for getting stuck: Check if we've passed the target
        old_dist = self.position.distance(target)
        new_dist = new_position.distance(target)
        if new_dist > old_dist:  # If we're further away after moving, we must have overshot
            new_position = target

            if self.forward:
                self.current_target += 1
                if self.current_target >= len(self.path):
                    self.current_target = len(self.path) - 1
                    self.forward = False
            else:
                self.current_target -= 1
                if self.current_target < 0:
                    self.current_target = 0
                    self.forward = True

        self.position = new_position

class FriendlyBot(MovingEntity):
    def __init__(self, init_pos, init_speed):
        super().__init__(init_pos, init_speed, [])

    def set_path(self, path):
        self.path = path
        self.current_target = 0


class Environment:
    def __init__(self):
        print("Initializing Environment...")
        self.enemy_path = [Point(GRID_SIZE // 2, GRID_SIZE // 2), Point(GRID_SIZE // 2, GRID_SIZE - 2), Point(GRID_SIZE // 2, GRID_SIZE // 2)]
        self.enemy = MovingEntity(Point(GRID_SIZE // 2, GRID_SIZE // 2), SPEED, self.enemy_path)
        self.friendly_bot = FriendlyBot(START_POSITION, SPEED)
        print("Creating RRT...")
        self.rrt = RRT(START_POSITION, FLAG_POSITION, WALLS)
        print("Setting path for friendly bot...")
        self.friendly_bot.set_path(self.rrt.plan())
        self.last_time = time.time()
        self.running = True
        self.state = "to_flag"
        print("Environment initialized.")

    def update(self):
        print("Updating...")
        current_time = time.time()
        delta_time = current_time - self.last_time
        self.last_time = current_time

        self.enemy.move(delta_time)
        self.friendly_bot.move(delta_time)

        if self.state == "to_flag" and self.friendly_bot.position.distance(FLAG_POSITION) < 0.1:
            print("Reached flag, repathing to start...")
            self.state = "to_start"
            self.rrt = RRT(FLAG_POSITION, START_POSITION, WALLS)
            self.friendly_bot.set_path(self.rrt.plan())
        elif self.state == "to_start" and self.friendly_bot.position.distance(START_POSITION) < 0.1:
            print("Reached start, repathing to flag...")
            self.state = "to_flag"
            self.rrt = RRT(START_POSITION, FLAG_POSITION, WALLS)
            self.friendly_bot.set_path(self.rrt.plan())

    def draw(self):
        print("Drawing...")
        screen.fill((255, 255, 255))

        for wall in WALLS:
            points = [(int(x * 20), int(y * 20)) for x, y in wall.exterior.coords]
            pygame.draw.polygon(screen, (128, 128, 128), points)

        pygame.draw.circle(screen, (0, 255, 0), (int(FLAG_POSITION.x * 20), int(FLAG_POSITION.y * 20)), 10)
        pygame.draw.circle(screen, (255, 255, 0), (int(START_POSITION.x * 20), int(START_POSITION.y * 20)), 10)
        pygame.draw.circle(screen, (255, 0, 0), (int(self.enemy.position.x * 20), int(self.enemy.position.y * 20)), 10)
        pygame.draw.circle(screen, (0, 0, 255), (int(self.friendly_bot.position.x * 20), int(self.friendly_bot.position.y * 20)), 10)

        pygame.display.flip()

    def run(self):
        print("Running simulation...")
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.update()
            self.draw()
            clock.tick(FPS)
        print("Simulation ended.")

if __name__ == "__main__":
    print("Starting script...")
    env = Environment()
    env.run()
    pygame.quit()
    print("Script finished.")