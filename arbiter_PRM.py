import sys
import time
import random
import math
import numpy as np
from shapely.geometry import Point, Polygon, LineString
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

# -------------------------------
# Simulation Constants
# -------------------------------
GRID_SIZE = 25
FPS = 120
FRAME_DELAY = 1 / FPS
MAX_ITER = 1000
SPEED = 10
TIME_STEP = 0.1                # Time resolution for 3D grid
TIME_HORIZON = GRID_SIZE * 3 / SPEED  # Total time horizon in seconds
TIME_RESOLUTION = int(TIME_HORIZON / TIME_STEP)  # Number of time steps
FLAG_POSITION = Point(GRID_SIZE - 2, GRID_SIZE - 2)
START_POSITION = Point(2, 2)
PLAYER_RADIUS = 0.5
OCCUPANCY_RADIUS = 2

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
    def __init__(self, point, parent=None, time=None, obs_cost=0):
        self.point = point
        self.parent = parent
        self.time = time
        self.obs_cost = obs_cost
    
    # Calculate distance to other node using Euclidean distance
    def distance(self, other):
        return np.sqrt((other.point.x - self.point.x)**2 + (other.point.y - self.point.y)**2)

class PRM:
    def __init__(self, start, goal, obstacles, max_iter=MAX_ITER, max_speed = SPEED/2):
        self.start = Node(start, None, 0)
        self.goal = Node(goal)
        self.obstacles = obstacles
        self.max_iter = max_iter
        self.max_speed = max_speed