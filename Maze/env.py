import numpy as np

class MazeEnv:
    def __init__(self, size=(5, 5)):
        self.size = size
        self.maze = np.zeros(size)  # 0 = empty space, -1 = obstacle, 1 = goal
        self.maze[4, 4] = 1  # Goal position
        self.start_pos = (0, 0)
        self.current_pos = self.start_pos
        
    def reset(self):
        """Reset the environment to the initial state."""
        self.current_pos = self.start_pos
        return self.current_pos
    
    def step(self, action):
        """Take an action and return the new state, reward, and done flag."""
        # Define the actions
        actions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        
        # Calculate new position
        move = actions[action]
        new_pos = (self.current_pos[0] + move[0], self.current_pos[1] + move[1])
        
        # Check boundaries
        if 0 <= new_pos[0] < self.size[0] and 0 <= new_pos[1] < self.size[1]:
            self.current_pos = new_pos
        
        # Determine reward
        if self.maze[self.current_pos] == 1:  # Goal
            reward = 100
            done = True
        else:
            reward = -1  # Standard move penalty
            done = False
        
        return self.current_pos, reward, done
    
    def render(self):
        """Render the maze with the agent's current position."""
        maze_render = np.copy(self.maze)
        maze_render[self.current_pos] = 2  # Mark the agent's position
        print(maze_render)
    
    def close(self):
        """Close the environment."""
        pass

if __name__ == '__main__':
    # Example usage
    env = MazeEnv()
    state = env.reset()
    print("Initial state: ", state)
    print("Initial Maze:", )

    env.render()

    done = False
    print("Start random actions:")
    while not done:
        action = np.random.choice(4)  # Random action
        state, reward, done = env.step(action)
        # print the state, reward, and done flag
        print(state, reward, done)
        env.render()