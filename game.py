import pygame
import random
from enum import Enum
from collections import namedtuple

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
FT_BLUE = (0, 186, 188)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


BLOCK_SIZE = 20
SPEED = 10

GREEN_APPLES = 2
RED_APPLES = 1


class SnakeGame:
    def __init__(self, width=760, height=520):
        if width % (BLOCK_SIZE * 2) != 0 or height % (BLOCK_SIZE * 2) != 0:
            raise Exception(
                'Width and Height must be multiples of ' + str(BLOCK_SIZE * 2)
            )

        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Learn2Slither')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.width // 2, self.height // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        self.score = 0
        self.green_apples = []
        self.red_apples = []
        self._place_food()

    def _place_food(self):
        while len(self.green_apples) < GREEN_APPLES:
            x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) \
                * BLOCK_SIZE
            y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) \
                * BLOCK_SIZE

            if Point(x, y) in self.snake or Point(x, y) in self.green_apples \
                    or Point(x, y) in self.red_apples:
                continue

            self.green_apples.append(Point(x, y))

        while len(self.red_apples) < RED_APPLES:
            x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) \
                * BLOCK_SIZE
            y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) \
                * BLOCK_SIZE

            if Point(x, y) in self.snake or Point(x, y) in self.green_apples \
                    or Point(x, y) in self.red_apples:
                continue

            self.red_apples.append(Point(x, y))

    def play_step(self, action=None):
        if action is None:
            action = self.direction
        else:
            self.direction = action

        self._move(action)
        self.snake.insert(0, self.head)
        self.score = len(self.snake) - 3

        game_over = False
        reward = -1

        if self.head in self.green_apples:
            reward += 1
            self.green_apples.remove(self.head)
            self._place_food()
        elif self.head in self.red_apples:
            reward -= 1
            self.red_apples.remove(self.head)
            self._place_food()
            self.snake.pop()
            self.snake.pop()
        else:
            self.snake.pop()

        if self.score <= 0 or self._is_collision():
            reward -= 10
            game_over = True
            return reward, game_over, self.score

        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score

    def _is_collision(self):
        if self.head.x > self.width - BLOCK_SIZE or self.head.x < 0 \
                or self.head.y > self.height - BLOCK_SIZE or self.head.y < 0:
            return True
        if self.head in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(WHITE)

        for pt in self.snake:
            pygame.draw.rect(
                self.display, FT_BLUE,
                pygame.Rect(
                    pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE
                )
            )

        for pt in self.green_apples:
            pygame.draw.rect(
                self.display, GREEN,
                pygame.Rect(
                    pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE
                )
            )

        for pt in self.red_apples:
            pygame.draw.rect(
                self.display, RED,
                pygame.Rect(
                    pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE
                )
            )

        pygame.display.flip()

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)


if __name__ == '__main__':
    game = SnakeGame()

    while True:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = Direction.UP
                elif event.key == pygame.K_DOWN:
                    action = Direction.DOWN
                elif event.key == pygame.K_LEFT:
                    action = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    action = Direction.RIGHT

        reward, game_over, score = game.play_step(action)

        # print('Reward', reward)
        # print('Game Over', game_over)
        # print('Score', score)

        if game_over:
            break

    print('Final Score', score)

    pygame.quit()
