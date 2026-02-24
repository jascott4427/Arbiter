# The Arbiter: A Dynamic Obstacle-Avoiding Bot for Flag Capture in 2D Space

**Authors:** James Scott & Kenadi Waymire 

---

## Overview
The **Arbiter** project focuses on developing a bot capable of navigating a 2D space containing static and dynamic obstacles to capture a flag and return home without being intercepted. The primary challenge is avoiding collisions with both static walls and a dynamically patrolling enemy.

This repository explores three path-planning algorithms:
1.  **2D RRT:** A baseline for static obstacle avoidance.
2.  **3D RRT (x, y, time):** Incorporates time to account for moving obstacles.
3.  **PRM with Cost Function:** Uses $A^*$ search to prioritize paths further from the enemy.

---

## Path Planning Approaches

### 1. 2D Rapidly-exploring Random Tree (RRT)
The 2D RRT serves as a base algorithm to navigate static obstacles.

![2D RRT Visualization](path_to_your_image_from_page_4.png)

* **Logic**: Incremental tree building by sampling random points and moving from the nearest node toward the sample by a fixed step size.
* **Pros**: Effectively avoids static obstacles using a buffer zone and is guaranteed to find a path if one exists.
* **Cons**: Does not account for dynamic obstacles (patrolling enemy).
* **Parameters**: A step size of 1.0 balances exploration speed and path smoothness.

### 2. 3D RRT (Space-Time)
This version extends the 2D RRT by treating time as a third dimension, allowing the bot to avoid predicted future positions of the enemy.

![3D RRT Tree and Occupancy Grid](path_to_your_image_from_page_5.png)

* **Logic**: The enemy's patrol is modeled as a 3D occupancy grid where each cell represents a specific time step.
* **Constraints**: The tree must move forward in time, and links are only allowed if the bot can traverse them within its maximum speed.
* **Optimization**: A **5% goal bias** is used to increase efficiency.
* **Challenges**: Discrepancies between planned and actual speeds can lead to "time synchronization errors" and collisions.

### 3. PRM with Cost Function (Final Implementation)
The final implementation utilizes a Probabilistic Road Map (PRM) to prioritize safety and distance from the enemy.

![3D PRM Roadmap and Path](path_to_your_image_from_page_10.png)

* **Safety weighting**: Edges closer to the enemy are weighted higher, forcing the bot to take longer but safer paths.
* **Cost Function**: A penalty is added to edges within the enemy's avoidance zone:
    $$Penalty = PENALTY\_FACTOR \times (threshold - d_{enemy})^2$$
* **Enemy FOV**: Nodes are verified against the enemy's field of view (FOV), defined as a cone based on range and angle.
* **Trajectory Smoothing**: Raw paths are processed into time-consistent trajectories by adjusting velocity between segments to ensure smoothness.

---

## Visualization & GUI
The project features a dual-view interface embedded into a Tkinter window:
* **Pygame Frame**: Renders a real-time 2D top-down simulation of the environment.
* **Matplotlib Frame**: Displays the underlying 3D graph (RRT or PRM) and the occupancy grid.

---

## Conclusion
The project demonstrated that while 3D RRT shows promise, it often produces paths too close to threats. The PRM with a cost function successfully prioritized safety by maintaining distance, though it was the most computationally intensive method.

---

## References
* **GitHub Repository**: [https://github.com/jascott4427/Arbiter](https://github.com/jascott4427/Arbiter)
