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
TIME_HORIZON = GRID_SIZE * 3 / SPEED  # Total time horizon in seconds
TIME_RESOLUTION = int(TIME_HORIZON / TIME_STEP)  # Number of time steps
FLAG_POSITION = Point(GRID_SIZE - 2, GRID_SIZE - 2)
START_POSITION = Point(2, 2)
PLAYER_RADIUS = 0.5  # Define the radius of the player

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
    def __init__(self, point, parent=None, time=None):
        self.point = point
        self.parent = parent
        self.time = time  # Set time when available

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
    """
    A 3D RRT planner that computes a path from a start point to a goal point while respecting obstacles,
    time constraints, and movement restrictions.
    
    Attributes:
        start (Point): The starting point for the robot.
        goal (Point): The goal point for the robot.
        obstacles (list): A list of obstacles in the environment.
        max_iter (int): The maximum number of iterations to try for pathfinding.
        step_size (float): The step size used when expanding the tree.
        max_speed (float): The maximum speed the robot can travel.
        occupancy_grid (np.array): A 3D grid representing the occupancy of space and time.
        lock (threading.Lock): A lock for thread safety during tree expansion.
    """

    def __init__(self, start, goal, obstacles, max_iter=MAX_ITER, step_size=1.0, max_speed=SPEED):
        """
        Initializes the RRT3D planner.
        
        Args:
            start (Point): The starting point for the robot.
            goal (Point): The goal point for the robot.
            obstacles (list): A list of obstacles in the environment.
            max_iter (int): The maximum number of iterations to try for pathfinding.
            step_size (float): The step size used when expanding the tree.
            max_speed (float): The maximum speed the robot can travel.
        """
        self.start = Node(start, None, 0)
        self.goal = Node(goal) # end time unknown
        self.obstacles = obstacles
        self.max_iter = max_iter
        self.step_size = step_size
        self.nodes = [self.start]
        self.occupancy_grid = np.zeros((GRID_SIZE, GRID_SIZE, TIME_RESOLUTION))  # Occupancy grid
        self.lock = threading.Lock()  # Lock for thread safety
        self.max_speed = max_speed  # Maximum speed for the robot


    def generate_random_point(self):
        """
        Generates a random point in the 2D space and time within the defined grid.
        
        Returns:
            tuple: A tuple containing a random 2D point and a random time.
        """
        x = random.uniform(0, GRID_SIZE)
        y = random.uniform(0, GRID_SIZE)
        t = random.uniform(0, TIME_HORIZON)  # Random time step within the time horizon
        return Point(x, y), t

    def nearest_node(self, point, time):
        """
        Finds the nearest node in the RRT tree to a given point and time.
        
        Args:
            point (Point): The target point to find the nearest node to.
            time (float): The target time to find the nearest node to.
        
        Returns:
            Node: The nearest node in the tree.
        """
        return min(self.nodes, key=lambda node: np.sqrt((node.point.x - point.x)**2 + 
                                                         (node.point.y - point.y)**2 + 
                                                         (node.time - time)**2))

    def new_point(self, nearest, random_point, random_time):
        """
        Generates a new point by moving a step from the nearest node towards the random point, 
        respecting the time constraints.
        
        Args:
            nearest (Node): The nearest node in the tree.
            random_point (Point): The random target point.
            random_time (float): The random target time.
        
        Returns:
            tuple: A tuple containing the new point and the new time after taking a step.
        """
        # Ensure we only move forward in time, and compute direction
        direction = np.arctan2(random_point.y - nearest.point.y, random_point.x - nearest.point.x)
        new_x = nearest.point.x + self.step_size * np.cos(direction)
        new_y = nearest.point.y + self.step_size * np.sin(direction)
        new_time = nearest.time + self.step_size / self.max_speed  # Ensure forward time movement

        # Calculate travel time and ensure it is within the max speed limit
        distance = np.sqrt((new_x - nearest.point.x)**2 + (new_y - nearest.point.y)**2)
        travel_time = distance / self.max_speed  # Time required to cover the distance

        if new_time > nearest.time + travel_time:
            new_time = nearest.time + travel_time  # Respect the travel time

        return Point(new_x, new_y), new_time

    def collision_free(self, point, time):
        """
        Checks if the given point and time are free from collisions in the occupancy grid.
        
        Args:
            point (Point): The point to check for collisions.
            time (float): The time associated with the point.
        
        Returns:
            bool: True if the point and time are free from collisions, False otherwise.
        """
        x_idx = int(point.x)
        y_idx = int(point.y)
        t_idx = int(time / TIME_STEP)

        # Ensure the point is within valid bounds
        if x_idx < 0 or x_idx >= GRID_SIZE or y_idx < 0 or y_idx >= GRID_SIZE or t_idx < 0 or t_idx >= TIME_RESOLUTION:
            return False
        
        # Check if the point is directly within an occupied space
        if self.occupancy_grid[x_idx, y_idx, t_idx] == 1:
            return False
        
        # Check for invalid points within 0.5 units (x, y, and time)
        for dx in range(-1, 2):  # Check neighboring grid points in x direction
            for dy in range(-1, 2):  # Check neighboring grid points in y direction
                for dt in range(-1, 2):  # Check neighboring grid points in time direction
                    if (0 <= x_idx + dx < GRID_SIZE and
                        0 <= y_idx + dy < GRID_SIZE and
                        0 <= t_idx + dt < TIME_RESOLUTION):
                        # Check distance within a radius of 0.5 units
                        if np.sqrt((dx * GRID_SIZE)**2 + (dy * GRID_SIZE)**2 + (dt * TIME_STEP)**2) <= 0.5:
                            if self.occupancy_grid[x_idx + dx, y_idx + dy, t_idx + dt] == 1:
                                return False
        return True

    def plan(self):
        """
        Executes the RRT path planning algorithm to find a path from the start to the goal.
        
        Returns:
            list: A list of points and times representing the path from start to goal, or None if no path is found.
        """
        for _ in range(self.max_iter):
            random_point, random_time = self.generate_random_point()
            nearest = self.nearest_node(random_point, random_time)
            new_point, new_time = self.new_point(nearest, random_point, random_time)

            if self.collision_free(new_point, new_time):
                new_node = Node(new_point, nearest, new_time)
                self.nodes.append(new_node)

                # Check if we reached the goal
                if new_point.distance(self.goal.point) <= self.step_size and abs(new_time - TIME_HORIZON) <= TIME_STEP:
                    self.goal.parent = new_node
                    return self.reconstruct_path(self.goal)

        return None

    def reconstruct_path(self, node):
        """
        Reconstructs the path from the start to the goal by following the parent pointers.
        
        Args:
            node (Node): The goal node from which the path reconstruction starts.
        
        Returns:
            list: A list of tuples containing points and times in the path from start to goal.
        """
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
        self.rrt3d = RRT3D(START_POSITION, FLAG_POSITION, self.walls)
        self.friendly_bot.set_path(self.rrt3d.plan())
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
            self.rrt3d = RRT3D(FLAG_POSITION, START_POSITION, self.walls)
            self.friendly_bot.set_path(self.rrt3d.plan())
        elif self.state == "to_start" and self.friendly_bot.position.distance(START_POSITION) < 0.001:
            self.state = "to_flag"
            self.rrt3d = RRT3D(START_POSITION, FLAG_POSITION, self.walls)
            self.friendly_bot.set_path(self.rrt3d.plan())

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
        fig = plt.figure(figsize=(12, 6))  # Larger figure for better visualization

        # Create a 3D axis for the occupancy grid
        ax1 = fig.add_subplot(121, projection='3d')  # Left subplot for occupancy grid
        ax1.set_title("Occupancy Grid")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Time (seconds)")

        # Create a 3D axis for the RRT3D tree and path
        ax2 = fig.add_subplot(122, projection='3d')  # Right subplot for RRT3D tree and path
        ax2.set_title("RRT3D Tree and Path")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Time (seconds)")

        while self.running:
            with self.rrt3d.lock:  # Thread-safe access
                occupancy_grid = self.rrt3d.occupancy_grid.copy()

            # Clear both subplots
            ax1.clear()
            ax2.clear()

            # Set titles and labels for both subplots
            ax1.set_title("Occupancy Grid")
            ax1.set_xlabel("X")
            ax1.set_ylabel("Y")
            ax1.set_zlabel("Time (seconds)")

            ax2.set_title("RRT3D Tree and Path")
            ax2.set_xlabel("X")
            ax2.set_ylabel("Y")
            ax2.set_zlabel("Time (seconds)")

            # Plot the occupancy grid in the first subplot
            x, y, t = np.where(occupancy_grid == 1)
            t_seconds = t * TIME_STEP  # Convert frame index to time in seconds
            ax1.scatter(x, y, t_seconds, c="red", marker="o", s=10, label="Occupied Cells")

            # Plot the RRT3D tree and path in the second subplot
            if self.rrt3d.nodes:
                # Plot the RRT3D tree
                for node in self.rrt3d.nodes:
                    if node.parent:
                        # Draw a line from the node to its parent
                        ax2.plot([node.point.x, node.parent.point.x],
                                [node.point.y, node.parent.point.y],
                                [node.time, node.parent.time],
                                c="gray", alpha=0.5, linewidth=0.5)

                # Plot the RRT3D path (if it exists)
                if self.friendly_bot.path:
                    path = self.friendly_bot.path
                    path_x = [p[0].x for p in path]  # Extract x values
                    path_y = [p[0].y for p in path]  # Extract y values
                    path_t = [p[1] for p in path]  # Extract time values
                    ax2.plot(path_x, path_y, path_t, c="blue", linewidth=2, label="RRT3D Path")

            # Add legends
            ax1.legend()
            ax2.legend()

            # Adjust layout and draw
            plt.tight_layout()
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