import numpy as np
import random

class MC_DLA():
    """Random Walker DLA implementation"""
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)

        # Place the seed in the center at the bottom
        self.bottom = self.size - 1
        self.center = self.size // 2

        self.grid[self.bottom, self.center] = 1
        self.particles_count = 1

    def candidate(self, x, y):
        """Check for candidates"""
        neighbours = [(0,1) , (1,0), (0,-1), (-1,0)]

        for dx, dy in neighbours:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:                
                if self.grid[nx, ny] == 1:
                    return True
        return False

    def add_walker(self):
        """Add a random walker"""
        x = 0
        y = random.randint(0, self.size - 1) # Start at a random point at the top

        walking = True
        while walking:
            dx, dy = random.choice([(0,1) , (1,0), (0,-1), (-1,0)])

            new_x = x + dx
            new_y = y + dy

            # If walker goes out of bounds remove it and try again
            if not (0 <= new_x < self.size and 0 <= new_y < self.size):
                return False
            
            # Update positions
            x = new_x
            y = new_y

            # Check if the point neighbours the cluster
            if self.candidate(x,y):
                self.grid[x, y] = 1
                self.particles_count += 1
                return True
    
    def add_walker_ps(self, ps):
        """Add a random walker including a sticking probability"""
        x = 0
        y = random.randint(0, self.size - 1) # Start at a random point at the top

        walking = True
        while walking:
            dx, dy = random.choice([(0,1) , (1,0), (0,-1), (-1,0)])

            new_x = x + dx
            new_y = y + dy

            # If walker goes out of bounds remove it and try again
            if not (0 <= new_x < self.size and 0 <= new_y < self.size):
                return False
            
            # Update positions
            x = new_x
            y = new_y

            # Check if the point neighbours the cluster
            if self.candidate(x,y):
                if random.random() < ps:
                    self.grid[x, y] = 1
                    self.particles_count += 1
                    return True
    


