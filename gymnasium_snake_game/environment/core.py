import math
import numpy as np
from .utils import *
import random

class Snake:
    def __init__(
        self,
        fps=60,
        max_step=9999,
        init_length=4,
        food_reward=1.0,
        dist_reward=0.0,
        living_bonus=0.0,
        death_penalty=0.0,
        init_hunger=100,
        width=10,
        height=10,
        block_size=20,
        background_color=Color.black,
        food_color=Color.green,
        head_color=Color.grey,
        body_color=Color.white,
        fitness=0,
    ) -> None:

        self.episode = 0
        self.fps = fps
        self.max_step = max_step
        self.init_length = min(init_length, width//2)
        self.food_reward = food_reward
        self.dist_reward = dist_reward
        self.living_bonus = living_bonus
        self.death_penalty = death_penalty
        self.blocks_x = width
        self.blocks_y = height
        self.food_color = food_color
        self.head_color = head_color
        self.body_color = body_color
        self.background_color = background_color
        self.food = Food(self.blocks_x, self.blocks_y, food_color)
        self.visited = set()  # Track visited positions for loop detection
        self.init_hunger = init_hunger
        self.hunger=init_hunger
        Block.size = block_size

        self.screen = None
        self.clock = None
        self.human_playing = False
        self.seed= None
        self.fitness = Fitness(fitness)
        
    def init(self):           
        self.episode += 1
        self.score = 0
        # Initialize direction to a known value, e.g., Right (3) for the snake to move initially
        self.direction = 2 
        self.current_step = 0
        self.head = Block(self.blocks_x//2, self.blocks_y//2, self.head_color)
        self.body = [self.head.copy(i, 0, self.body_color)
                     for i in range(-self.init_length, 0)]
        self.blocks = [self.food.block, self.head, *self.body]
        self.food.new_food(self.blocks)
        self.hunger = self.init_hunger
        
    def set_seed(self, seed):
        self.seed = seed
        random.seed(self.seed)

    def close(self):
        pygame.quit()
        pygame.display.quit()
        self.screen = None
        self.clock = None

    def render(self):
        if self.screen is None:
            self.screen, self.clock = game_start(
                self.blocks_x*Block.size, self.blocks_y*Block.size)
        self.clock.tick(self.fps)
        update_screen(self.screen, self)
        handle_input()

    def step(self, action): # Renamed 'direction' to 'action' for clarity
        # action is expected to be an integer: 0 for Straight, 1 for Left, 2 for Right
        if action is None:
            # If no action is provided (e.g., at the very start or for human input)
            # just continue in the current direction.
            new_direction = self.direction 
        else:
            current_direction_vec = Direction.step(self.direction)
            
            if action == 0:  # Go Straight
                new_direction_vec = current_direction_vec
            elif action == 1:  # Turn Left
                new_direction_vec = Snake.left(current_direction_vec)
            elif action == 2:  # Turn Right
                new_direction_vec = Snake.right(current_direction_vec)
            else:
                # Handle unexpected action, e.g., default to straight or raise error
                print(f"Warning: Unexpected action value: {action}. Defaulting to straight.")
                new_direction_vec = current_direction_vec

            # Convert the new direction vector back to your internal direction integer (0-3)
            # You'll need to define a mapping for Direction.step() for this.
            # Assuming Direction.step(0)=(0,1), Direction.step(1)=(0,-1), Direction.step(2)=(-1,0), Direction.step(3)=(1,0)
            if new_direction_vec == (0, -1): # Up
                new_direction = 0
            elif new_direction_vec == (-1, 0): # Left
                new_direction = 1
            elif new_direction_vec == (1, 0): # Right
                new_direction = 2
            elif new_direction_vec == (0, 1): # Down
                new_direction = 3
            else:
                # This should ideally not happen if left/right are properly implemented
                #print("test")
                new_direction = self.direction # Fallback


        self.current_step += 1
        truncated = self.current_step == self.max_step

        (x, y) = (self.head.x, self.head.y)
        
        # Apply the new direction
        # The previous 'if (direction in [0, 1] ...)' block for preventing reversing is no longer needed
        # because the relative movement naturally prevents 180-degree turns.
        self.direction = new_direction # Update the internal direction

        step_vec = Direction.step(self.direction) # Get the (dx, dy) for the new direction

        self.head.x += step_vec[0]
        self.head.y += step_vec[1]
        
        dead = False
        
        # Eat food
        reward = self.fitness.next(self) 
        
        if self.head == self.food.block:
            self.score += 1
            self.grow(x, y)
            self.food.new_food(self.blocks)
            self.visited = set()
            self.hunger=self.init_hunger
        else:
            if self.hunger==0:
                dead = True 
                #print('starved')
            else:
                self.hunger-=1
                
            self.move(x, y)

            # Check collision with body or wall
            for block in self.body:
                if self.head == block:
                    dead = True
                    #print('wall')
            if self.head.x >= self.blocks_x or self.head.x < 0 or \
            self.head.y >= self.blocks_y or self.head.y < 0:
                dead = True
                #print('tail')
        
        return self.observation(), reward, dead, truncated

    def observation(self):
        obs = np.zeros((self.blocks_x+2, self.blocks_y+2, 4), dtype=np.float32)
        if 0 <= self.head.x < self.blocks_x and 0 <= self.head.y < self.blocks_y:
            obs[self.head.x+1][self.head.y+1][0] = 1
        for block in self.blocks:
            if 0 <= block.x < self.blocks_x and 0 <= block.y < self.blocks_y:
                obs[block.x+1][block.y+1][1] = 1
        obs[self.food.block.x+1][self.food.block.y+1][2] = 1
        # top and bottom rows (rows are first index)
        for x in range(self.blocks_y + 2):  # columns
            obs[0][x][3] = 1                # top row
            obs[self.blocks_x + 1][x][3] = 1  # bottom row

        # left and right columns (columns are second index)
        for y in range(self.blocks_x + 2):  # rows
            obs[y][0][3] = 1                # left column
            obs[y][self.blocks_y + 1][3] = 1  # right column

        return obs

    def calc_reward(self):
        if self.dist_reward == 0.0:
            return 0
        x = self.head.x - self.food.block.x
        y = self.head.y - self.food.block.y
        d = math.sqrt(x*x + y*y)
        return (self.dist_reward-d)/self.dist_reward

    def grow(self, x, y):
        body = Block(x, y, self.body_color)
        self.blocks.append(body)
        self.body.append(body)

    def move(self, x, y):
        tail = self.body.pop(0)
        tail.move_to(x, y)
        self.body.append(tail)

    def info(self):
        return {
            'head': (self.head.x, self.head.y),
            'food': (self.food.block.x, self.food.block.y),
        }

    def play(self, fps=10, acceleration=True, step=1, frep=10):
        self.max_step = 99999
        self.fps = fps
        self.food_reward = 1
        self.living_bonus = 0
        self.dist_reward = 0
        self.death_penalty = 0
        self.human_playing = True
        self.init()
        screen, clock = game_start(
            self.blocks_x*Block.size, self.blocks_y*Block.size)
        total_r = 0

        while pygame.get_init():
            clock.tick(self.fps)
            # For human playing, handle_input() should return the desired relative action (0, 1, or 2)
            # You might need to adjust handle_input() to return these values based on arrow keys.
            _, r, d, _ = self.step(handle_input()) 
            total_r += r
            if acceleration and total_r == frep:
                self.fps += step
                total_r = 0
            if d:
                self.init()
                total_r = 0
                self.fps = fps
            update_screen(screen, self, True)

    ### ADDED ###
    def get_inputs(self):
        matrix = self.observation() # This is (blocks_x+2, blocks_y+2, 4)
        MAX_DIST = self.blocks_x

        def look_to(direction_vec, head_pos_in_matrix, matrix):
            current_x, current_y = head_pos_in_matrix[0], head_pos_in_matrix[1]
            dx, dy = direction_vec  

            dist = 0
            food_found = False
            tail_found = False 
            wall_found = False

            max_x_idx, max_y_idx = matrix.shape[0], matrix.shape[1]

            wall = food = tail = None
            
            current_x += dx
            current_y += dy
            dist += 1 # Distance starts at 1 for the first cell checked

            # Loop while within the padded observation boundaries
            while 0 <= current_x < max_x_idx and 0 <= current_y < max_y_idx:
                # Channel 3: Walls
                if not wall_found and matrix[current_x, current_y, 3] == 1.0:
                    wall = dist
                    wall_found = True
                # Channel 2: Food
                if not food_found and matrix[current_x, current_y, 2] == 1.0:
                    food = dist
                    food_found = True
                # Channel 1: Body/Tai
                if not tail_found and matrix[current_x, current_y, 1] == 1.0:
                    tail = dist
                    tail_found = True
                
                # If all found, we can stop looking in this direction
                if wall_found and food_found and tail_found:
                    break

                # Move to the next cell in the current direction
                current_x += dx
                current_y += dy
                dist += 1

            # Assign MAX_DIST if nothing was found in that direction
            if wall is None:
                wall = MAX_DIST
            if food is None:
                food = MAX_DIST
            if tail is None:
                tail = MAX_DIST

            return wall, food, tail

        # The head's position in the padded observation matrix: (x+1, y+1)
        # So it directly maps to matrix[x_coord, y_coord]
        head_pos_in_matrix = (self.head.x+1, self.head.y+1) 

        # Get the current orientation vector (dx, dy)
        orientation = Direction.step(self.direction) 

        features = []
        # The 7 directions for 21 inputs:
        directions = [
            orientation,                              # Straight ahead
            self.left(orientation),                   # 90 degrees left
            self.right(orientation),                  # 90 degrees right
            self.semi_left(orientation),              # 45 degrees left diagonal
            self.semi_right(orientation),             # 45 degrees right diagonal
            self.semi_left(self.left(orientation)),   # 135 degrees left (back-left diagonal)
            self.semi_right(self.right(orientation)), # 135 degrees right (back-right diagonal)
        ]

        for dir_vec in directions:
            wall_dist, food_dist, tail_dist = look_to(dir_vec, head_pos_in_matrix, matrix)
            
            # Normalize distances to be between 0 and 1, where 1 means close, 0 means far
            # Using 1 - (dist / MAX_DIST) is common for this
            features.extend([
                wall_dist, 
                food_dist, 
                tail_dist
            ])
        
        return features

    @staticmethod
    def left(direction_vec):
        dx, dy = direction_vec
        return dy, -dx 

    @staticmethod
    def right(direction_vec):
        dx, dy = direction_vec
        return -dy, dx 
    
    @staticmethod
    def semi_left(direction_vec):
        dx, dy = direction_vec
        ldx, ldy = Snake.left(direction_vec)
        return (dx + ldx, dy + ldy)

    @staticmethod
    def semi_right(direction_vec):
        dx, dy = direction_vec
        rdx, rdy = Snake.right(direction_vec)
        return (dx + rdx, dy + rdy)
    
    def render_image(self):
        """
        Renders the current game state as a Pygame surface and returns it.
        The image will have the snake in white, food in green, and a black background.
        """
        # Ensure Pygame is initialized
        if not pygame.get_init():
            pygame.init()

        # Calculate screen dimensions
        screen_width = self.blocks_x * Block.size
        screen_height = self.blocks_y * Block.size

        # Create a surface for rendering
        image_surface = pygame.Surface((screen_width, screen_height))
        # FIX: Removed .value here
        image_surface.fill(self.background_color) # Fill background with black

        # Draw food
        food_rect = pygame.Rect(
            self.food.block.x * Block.size,
            self.food.block.y * Block.size,
            Block.size,
            Block.size
        )
        # FIX: Removed .value here
        pygame.draw.rect(image_surface, self.food_color, food_rect)

        # Draw snake body
        for i, block in enumerate(self.body):
            if block == self.head:
                continue 
            body_rect = pygame.Rect(
                block.x * Block.size,
                block.y * Block.size,
                Block.size,
                Block.size
            )
            # FIX: Removed .value here
            pygame.draw.rect(image_surface, self.body_color, body_rect)

        # Draw snake head (on top of body if overlap)
        head_rect = pygame.Rect(
            self.head.x * Block.size,
            self.head.y * Block.size,
            Block.size,
            Block.size
        )
        # FIX: Removed .value here
        pygame.draw.rect(image_surface, self.head_color, head_rect)

        return image_surface
    
# Define the Fitness class
class Fitness:
    

    def __init__(self, fitness):
        self.base = True
        self.antiloop = True
        self.eating = True
        self.nice_smell = True
        print(fitness)
        match fitness:
            case 0:
                return
            case 1:
                self.base = False
            case 2:
                self.antiloop = False
            case 3:
                self.eating = False
            case 4:
                self.nice_smell = False

    def next(self, snake) -> float:
        reward = 0
        if self.base:
            reward = 0.01  # baseline movement reward

        # Loop detection
        if self.antiloop:
            pos = (snake.head.x, snake.head.y)
            if pos in snake.visited:
                reward -= 0.25  # Penalize for looping
            else:
                snake.visited.add(pos)
        # Eat food
        if snake.head == snake.food.block:
            if self.eating:
                reward += 5.0
                snake.visited = set()
        elif self.nice_smell:
            # Near food bonus
            dist = abs(snake.head.x - snake.food.block.x) + abs(snake.head.y - snake.food.block.y)
            if dist == 1:
                reward += 0.2
        return reward
