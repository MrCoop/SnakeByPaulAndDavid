import pygame
import random
from collections import deque
import numpy as np

# Farben
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLACK = (0, 0, 0)

class SnakeGame:
    def __init__(self, grid_size=10, block_size=30, fps=10, render=True):
        pygame.init()
        self.grid_size = grid_size
        self.block_size = block_size
        self.width = self.height = self.grid_size * self.block_size
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake RL")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake = deque([[self.grid_size // 2, self.grid_size // 2]])
        self.direction = 1  # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self._place_food()
        self.done = False
        self.score = 0
        return self._get_state()

    def _place_food(self):
        empty_cells = [
            [x, y]
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if [x, y] not in self.snake
        ]
        self.food = random.choice(empty_cells)

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        if abs(action - self.direction) == 2:
            action = self.direction  # Keine 180°-Wende
        self.direction = action

        head_x, head_y = self.snake[0]
        move = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        new_head = [head_x + move[0], head_y + move[1]]

        if (
            new_head in self.snake
            or not (0 <= new_head[0] < self.grid_size)
            or not (0 <= new_head[1] < self.grid_size)
        ):
            self.done = True
            return self._get_state(), -1, self.done, {}

        self.snake.appendleft(new_head)
        reward = 0

        if new_head == self.food:
            reward = 1
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        self._render()
        return self._get_state(), reward, self.done, {}

    def _get_state(self):
        state = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for x, y in self.snake:
            state[y][x] = 0.5
        state[self.food[1]][self.food[0]] = 1.0
        return state.flatten()

    def _render(self):
        self.display.fill(BLACK)
        # Snake zeichnen
        for x, y in self.snake:
            pygame.draw.rect(
                self.display,
                GREEN,
                pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
            )
        # Food zeichnen
        fx, fy = self.food
        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(fx * self.block_size, fy * self.block_size, self.block_size, self.block_size)
        )

        pygame.display.flip()
        self.clock.tick(10)  # FPS

    def close(self):
        pygame.quit()




