import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()


# Define directions
class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

    @property
    def vector(self):
        vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        return vectors[self.value]


Point = namedtuple('Point', 'x y')

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 250  # Faster for more training iterations


class SnakeGameAI:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 25)
        self.reset()

        # Maximum number of steps without eating food before game ends
        self.max_steps_without_food = 100

    def reset(self):
        # Start in the middle
        self.direction = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.steps_without_food = 0

        # Track previous distance to food for reward calculation
        self.previous_distance = self._get_distance_to_food()

    def _place_food(self):
        x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)

        # Don't place food on snake
        if self.food in self.snake:
            self._place_food()

    def _get_distance_to_food(self):
        """Calculate Manhattan distance to food"""
        return abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

    def play_step(self, action):
        self.frame_iteration += 1
        self.steps_without_food += 1

        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move snake
        self._move(action)
        self.snake.insert(0, self.head)

        # Initialize reward
        reward = 0
        game_over = False

        # Check if game over
        # 1. Hit boundary or hit self
        if self._is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 2. Too many steps without food (stuck in a loop)
        if self.steps_without_food > self.max_steps_without_food * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Check if snake got food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.steps_without_food = 0
            self._place_food()
        else:
            self.snake.pop()

        # Calculate distance-based reward
        current_distance = self._get_distance_to_food()
        if current_distance < self.previous_distance:
            # Reward for moving closer to food (proportional to distance reduction)
            distance_improvement = (self.previous_distance - current_distance) / (self.width // BLOCK_SIZE)
            reward += 0.1 + distance_improvement
        else:
            # Small penalty for moving away from food
            reward -= 0.1

        # Additional small reward for staying alive
        reward += 0.01

        self.previous_distance = current_distance

        # Update UI
        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # Hit boundary
        if pt.x >= self.width or pt.x < 0 or pt.y >= self.height or pt.y < 0:
            return True

        # Hit self
        if pt in self.snake[1:]:
            return True

        return False

    def _move(self, action):
        # Action is [straight, right, left]

        # Get directions as integers for easier calculations
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = directions.index(self.direction)

        if action[0] == 1:  # Continue straight
            new_direction = directions[idx]
        elif action[1] == 1:  # Turn right
            new_direction = directions[(idx + 1) % 4]
        else:  # Turn left
            new_direction = directions[(idx - 1) % 4]

        self.direction = new_direction

        # Get movement vector
        x, y = self.direction.vector
        self.head = Point(self.head.x + (x * BLOCK_SIZE), self.head.y + (y * BLOCK_SIZE))

    def clone_state(self):
        """Create a copy of the current game state for reward calculation"""
        from collections import namedtuple
        GameState = namedtuple('GameState', ['head', 'food', 'steps_without_food', 'width', 'height', 'snake'])

        return GameState(
            head=self.head,
            food=self.food,
            steps_without_food=self.steps_without_food,
            width=self.width,
            height=self.height,
            snake=self.snake.copy()
        )

    def _update_ui(self):
        self.display.fill(BLACK)

        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        # Draw food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Draw score
        text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])

        pygame.display.flip()